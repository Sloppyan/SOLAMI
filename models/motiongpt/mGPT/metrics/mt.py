import os
import torch
from torch import Tensor
from typing import List
from torchmetrics import Metric
import numpy as np
from .utils import *
from mGPT.metrics.features.kinetic_torch import normalize, cal_average_kinetic_energy_torch
from mGPT.data.humanml.scripts.ske_process import recover_from_ric, recover_from_smplx_feature
from mGPT.config import instantiate_from_config
sys.path.append('/mnt/AFS_jiangjianping/projects/tools/smplx')
import smplx
import spacy
# from bert_score import score as score_bert
import copy

class MTMetrics(Metric):
    def __init__(self, 
                cfg,
                bleu_k=4,
                max_text_len=40,
                force_in_meter: bool = True,
                align_root: bool = True,
                dist_sync_on_step=True,
                diversity_times=300,
                w_vectorizer=None,
                dataname='humanml3d',
                **kwargs):
        
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.cfg = cfg
        self.metrics = []
        self.w_vectorizer = w_vectorizer
        self.dataname = dataname
        self.max_text_len = max_text_len
        self.bleu_k = bleu_k
        # Fid
        self.add_state("FID", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.append("FID")

        self.add_state("pred_valid", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.append("pred_valid")

        self.add_state("code_frequency", default=torch.zeros((256, )), dist_reduce_fx="sum")
        self.metrics.append("code_frequency")

        # Diversity
        self.add_state("Diversity",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("gt_Diversity",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.metrics.extend(["Diversity", "gt_Diversity"])
        self.diversity_times = diversity_times

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        self.add_state("recmotion_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("gtmotion_embeddings", default=[], dist_reduce_fx=None)
        
        self.betas = torch.tensor([-0.06134899, -0.4861751 ,  0.8630473 , -3.07320443,  1.10772016,
                                    -1.44656493,  2.97690664, -1.12731489,  1.24817344, -1.4111463 ,
                                    -0.04035034, -0.29547926,  0.38509519,  0.13750311,  0.94445029,
                                    -0.47172116], dtype=torch.float32)
        
        self.t_root_J = torch.tensor([
            0, 0, 0
        ], dtype=torch.float32)
        
        self.model_path = '/mnt/AFS_jiangjianping/datasets/Assets/SMPL_MODELS/smplx/SMPLX_MALE.npz'
        self.smplx_model = smplx.create(self.model_path, 
                                        model_type='smplx', 
                                        gender='male', 
                                        ext='npz', 
                                        num_betas=len(self.betas), 
                                        use_pca=False, 
                                        flat_hand_mean=True)
        self.smplx_model.eval()
        
        
        self.align_root = align_root
        self.force_in_meter = force_in_meter
        self.add_state("MPJPE",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("PAMPJPE",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("ACCEL",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.metrics += ["MPJPE", "PAMPJPE", "ACCEL"]
        
        # NLG
        self._get_t2m_evaluator(cfg)
        self.pred_texts = []
        self.gt_texts = []
        self.nlp = spacy.load('en_core_web_sm')
        for k in [1, bleu_k]:
            self.add_state(
                f"Bleu_{str(k)}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.metrics.append(f"Bleu_{str(k)}")

        self.add_state("ROUGE_L",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.metrics.append("ROUGE_L")

        self.add_state("CIDEr",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.metrics.append("CIDEr")
        
        # self.add_state("Bert_F1",
        #                default=torch.tensor(0.0),
        #                dist_reduce_fx="sum")
        # self.metrics.append("Bert_F1")
        
        self.add_state("predtext_embeddings", default=[])
        self.add_state("gttext_embeddings", default=[])
        
        from nlgmetricverse import NLGMetricverse, load_metric
        metrics = [
            load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}),
            load_metric("bleu", resulting_name="bleu_4", compute_kwargs={"max_order": 4}),
            load_metric("rouge"),
            load_metric("cider"),
        ]
        self.nlg_evaluator = NLGMetricverse(metrics)
        pass
        
    @torch.no_grad()
    def compute(self, sanity_flag):
        count = self.count.item()
        count_seq = self.count_seq.item()
        pred_valid = self.pred_valid.item()
        metrics = {metric: getattr(self, metric) for metric in self.metrics}
        if pred_valid == 0:
            self.reset()
            self.gt_texts = []
            self.pred_texts = []
            metrics['code_frequency'] = torch.tensor(0.0).to(self.device)
            return {**metrics}
        
        # assert pred_valid == len(self.recmotion_embeddings)
        
        
        metrics['pred_valid'] = torch.tensor(pred_valid / count_seq).to(self.device)
        
        codes = torch.sum(self.code_frequency)
        if codes > 0:
            self.code_frequency /= codes
            epsilon = 1e-10
            prob_dist = self.code_frequency + epsilon
            
            # 计算熵 H(X) = -sum(p(x) * log(p(x)))
            entropy = -torch.sum(prob_dist * torch.log(prob_dist))
            metrics['code_frequency'] = entropy.to(self.device)
        else:
            metrics['code_frequency'] = torch.tensor(0.0).to(self.device)
        # if sanity_flag:
        #     return metrics
        
        # shuffle_idx = torch.randperm(count_seq)
        # all_genmotions = torch.cat(self.recmotion_embeddings,
        #                            axis=0).cpu()[shuffle_idx, :]
        # all_gtmotions = torch.cat(self.gtmotion_embeddings,
        #                           axis=0).cpu()[shuffle_idx, :]
        # all_genmotions = all_genmotions.numpy()
        # all_gtmotions = all_gtmotions.numpy()
        if not sanity_flag:
            assert len(self.recmotion_embeddings) == len(self.gtmotion_embeddings)
            shuffle_idx = np.random.permutation(len(self.gtmotion_embeddings))
            all_genmotions = np.concatenate(self.recmotion_embeddings, axis=0)[shuffle_idx, :]
            all_gtmotions = np.concatenate(self.gtmotion_embeddings, axis=0)[shuffle_idx, :]

            # Normalize
            all_genmotions = normalize(all_genmotions)
            all_gtmotions = normalize(all_gtmotions)
            # Compute fid
            mu, cov = calculate_activation_statistics_np(all_genmotions)
            gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)
            metrics["FID"] = torch.tensor(calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)).to(self.device)

            # Compute diversity
            if pred_valid > self.diversity_times:
                metrics["Diversity"] = torch.tensor(calculate_diversity_np(all_genmotions,
                                                            self.diversity_times)).to(self.device)
                metrics["gt_Diversity"] = torch.tensor(calculate_diversity_np(
                    all_gtmotions, self.diversity_times)).to(self.device)

        
        if self.force_in_meter:
            factor = 1000.0
        else:
            factor = 1.0

        count = count / (pred_valid / (len(self.gtmotion_embeddings)))
        pred_valid = pred_valid / (len(self.gtmotion_embeddings))

        mr_metrics = {}
        mr_metrics["MPJPE"] = self.MPJPE / count * factor
        mr_metrics["PAMPJPE"] = self.PAMPJPE / count * factor
        # accel error: joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
        # n-2 for each sequences
        mr_metrics["ACCEL"] = self.ACCEL / (count - 2 * pred_valid) * factor
        metrics.update(mr_metrics)
        
        
        
        # NLP scores
        pred_texts = copy.deepcopy(self.pred_texts)
        gt_texts = copy.deepcopy(self.gt_texts)
        assert len(pred_texts) == len(gt_texts)
        if len(gt_texts) != 0:
            
            scores = self.nlg_evaluator(predictions=gt_texts,
                                        references=gt_texts)
            for k in [1, self.bleu_k]:
                metrics[f"Bleu_{str(k)}"] = torch.tensor(scores[f'bleu_{str(k)}']['score'],
                                                        device=self.device)
                
            metrics["ROUGE_L"] = torch.tensor(scores["rouge"]["rougeL"],
                                            device=self.device)
            metrics["CIDEr"] = torch.tensor(scores["cider"]['score'],device=self.device)

        # Bert metrics
        # P, R, F1 = score_bert(pred_texts,
        #                       gt_texts,
        #                       lang='en',
        #                       rescale_with_baseline=True,
        #                       idf=True,
        #                       device=self.device,
        #                       verbose=False)

        # metrics["Bert_F1"] = F1.mean()
        
        
        # Reset
        self.reset()
        self.gt_texts = []
        self.pred_texts = []
        
        return {**metrics}
    
    @torch.no_grad()
    def update(self,
               feats_ref: Tensor = None,
               feats_rst: Tensor = None,
               lengths_ref: List[int] = [],
               lengths_rst: List[int] = [],
               pred_texts: List[str] = [],
               gt_texts: List[str] = [],
               pred_valid: List[bool] = [],
               code_frequency: Tensor = None,
               ):
        # self.count += sum(lengths_ref)
        self.count_seq += len(lengths_ref)
        self.pred_valid += sum(pred_valid)
        self.count += sum(lengths_ref)
        if sum(pred_valid) == 0:
            return
        #### use valid data to calculate the metrics
        if feats_ref is not None and feats_rst is not None:
            feats_ref = feats_ref[pred_valid]
            feats_rst = feats_rst[pred_valid]
            lengths_ref = [lengths_ref[i] for i in range(len(lengths_ref)) if pred_valid[i]]
            lengths_rst = [lengths_rst[i] for i in range(len(lengths_rst)) if pred_valid[i]]
        if pred_texts is not []:
            pred_texts = [pred_texts[i] for i in range(len(pred_texts)) if pred_valid[i]]
            gt_texts = [gt_texts[i] for i in range(len(gt_texts)) if pred_valid[i]]
        
        if code_frequency is not None:
            self.code_frequency += code_frequency
        
        if feats_ref is not None and feats_rst is not None:
            # align_idx = np.argsort(lengths_ref)[::-1].copy()
            # feats_ref = feats_ref[align_idx]
            # lengths_ref = np.array(lengths_ref)[align_idx]
            
            gtmotion_embeddings, ref = self.get_motion_embeddings(
                feats_ref, lengths_ref)
            cache = [0] * len(lengths_ref)
            for i in range(len(lengths_ref)):
                cache[i] = gtmotion_embeddings[i:i + 1]
            self.gtmotion_embeddings.extend(cache)
            
            # align_idx = np.argsort(lengths_rst)[::-1].copy()
            # feats_rst = feats_rst[align_idx]
            # lengths_rst = np.array(lengths_rst)[align_idx]
            recmotion_embeddings, rst = self.get_motion_embeddings(
                feats_rst, lengths_rst)
            cache = [0] * len(lengths_rst)
            for i in range(len(lengths_rst)):
                cache[i] = recmotion_embeddings[i:i + 1]
            self.recmotion_embeddings.extend(cache)
            
            assert rst.shape == ref.shape
            assert rst.dim() == 4
            if self.align_root:
                align_inds = [0]
            else:
                align_inds = None

            for i in range(len(lengths_ref)):
                # len_tmp = min(lengths_ref[i], lengths_rst[i])
                len_tmp = lengths_ref[i]
                self.MPJPE += torch.sum(
                    calc_mpjpe(rst[i][: len_tmp], ref[i][: len_tmp], align_inds=align_inds))
                self.PAMPJPE += torch.sum(calc_pampjpe(rst[i][: len_tmp], ref[i][: len_tmp]))
                self.ACCEL += torch.sum(calc_accel(rst[i][: len_tmp], ref[i][: len_tmp]))
        else:
            print('Invalid data')
        if pred_texts is not []:
            # predtext_emb = self._get_text_embeddings(pred_texts)
            # predtext_embeddings = torch.flatten(predtext_emb, start_dim=1).detach()

            # gttext_emb = self._get_text_embeddings(gt_texts)
            # gttext_embeddings = torch.flatten(gttext_emb, start_dim=1).detach()
            # self.gttext_embeddings.append(gttext_embeddings)
            # self.predtext_embeddings.append(predtext_embeddings)

            self.pred_texts.extend(pred_texts)
            self.gt_texts.extend(gt_texts)
        pass
    
    
    def revserse_ric_data(self, data):
        # index from all data to body and hands
        body_index = list(range(0, 4+21*3)) + list(range(4+51*3, 4+51*3+21*6)) + \
                list(range(4+51*9, 4+51*9+22*3)) + list(range(4+51*9+52*3, 4+51*9+52*3+4))
        hand_index = list(range(4+21*3, 4+51*3)) + list(range(4+51*3+21*6, 4+51*9)) + \
            list(range(4+51*9+22*3, 4+51*9+52*3))
            
        all_index_to_feat = body_index + hand_index
        ## get the index of each item in all_index_to_feat
        all_feat_to_ric = {}
        for i, item in enumerate(all_index_to_feat):
            all_feat_to_ric[item] = i
        # sort the dicts by key, ascending
        all_feat_to_ric = dict(sorted(all_feat_to_ric.items(), key=lambda item: item[0]))
        
        reverse_indices = list(all_feat_to_ric.values())
        return data[..., reverse_indices]
    
    
    def get_motion_embeddings(self, feats: Tensor, lengths: List[int]):
        # step 1 get original data
        njoints = 22 if self.cfg.EXPER.motion_part == 'body' else 52
        if self.cfg.EXPER.motion_repre == 'ske':
            if njoints > 25:
                feats = self.revserse_ric_data(feats)
            joints_ = recover_from_ric(feats, njoints)
            joints = self.process_smplx_joints(joints_)
        elif self.cfg.EXPER.motion_repre == 'global cont6d':
            res = recover_from_smplx_feature(feats, 'global')
            # res_test = process_smplx_feature(res, 'global')
            joints_ = self.smplx_infer(res)
            joints = self.process_smplx_joints(joints_)
        elif self.cfg.EXPER.motion_repre == 'local cont6d':
            res = recover_from_smplx_feature(feats, 'local')
            joints_ = self.smplx_infer(res)
            joints = self.process_smplx_joints(joints_)
            # res_test = process_smplx_feature(res.clone(), 'local')
            # reconstruction error is normal, considering the padding sequence
        else:
            raise ValueError('Unknown motion representation')        
        
        # step 3 cal kinetic feature
        # kinectic_features = []
        # for i in range(joints.shape[0]):
        #     joints_tmp = joints[i][:lengths[i]]
        #     kinectic_feature = extract_kinetic_features(joints_tmp.detach().cpu().numpy())
        #     kinectic_features.append(kinectic_feature)
        # kinectic_feats = np.array(kinectic_features, dtype=np.float32)
        
        kinectic_feats = cal_average_kinetic_energy_torch(joints.detach(), lengths)
        kinectic_feats_np = kinectic_feats.cpu().numpy()
        return kinectic_feats_np, joints_.detach().cpu()
    
    
    def smplx_infer(self, res):
        batch_size, seq_len, feat_len = res.shape
        
        global_orient = res[..., 3:6]
        
        transl = res[..., :3] - self.t_root_J.to(res.device)
        
        betas = self.betas.to(res.device)
        betas = betas.repeat(batch_size, seq_len, 1)
        
        expression = torch.zeros([batch_size, seq_len, 10], dtype=torch.float32).to(res.device)
        
        if feat_len in [6 + 21*3, 6+24*3]:
            jaw_pose = torch.zeros([batch_size, seq_len, 3], dtype=torch.float32).to(res.device)
            leye_pose = torch.zeros([batch_size, seq_len, 3], dtype=torch.float32).to(res.device)
            reye_pose = torch.zeros([batch_size, seq_len, 3], dtype=torch.float32).to(res.device)
            left_hand_pose = torch.zeros([batch_size, seq_len, 45], dtype=torch.float32).to(res.device)
            right_hand_pose = torch.zeros([batch_size, seq_len, 45], dtype=torch.float32).to(res.device)
            body_pose = res[..., 6:6 + 21*3]
        elif feat_len == 6 + 51 *3:
            body_pose = res[..., 6:6+21*3]
            left_hand_pose = res[..., 6+21*3: 6+(21+15)*3]
            right_hand_pose = res[..., 6+36*3:]
            jaw_pose = torch.zeros([batch_size, seq_len, 3], dtype=torch.float32).to(res.device)
            leye_pose = torch.zeros([batch_size, seq_len, 3], dtype=torch.float32).to(res.device)
            reye_pose = torch.zeros([batch_size, seq_len, 3], dtype=torch.float32).to(res.device)
        elif feat_len == 6 + 54 *3:
            body_pose = res[..., 6:6+21*3]
            jaw_pose = res[..., 6+21*3: 6+22*3]
            leye_pose = res[..., 6+22*3: 6+23*3]
            reye_pose = res[..., 6+23*3: 6+24*3]
            left_hand_pose = res[..., 6+24*3: 6+(24+15)*3]
            right_hand_pose = res[..., 6+39*3:]
        else:
            raise ValueError('Unknown feature length')
        
        body_parms = {
            'global_orient': global_orient,
            'body_pose': body_pose,
            'jaw_pose': jaw_pose,
            'leye_pose': leye_pose,
            'reye_pose': reye_pose,
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
            'transl': transl,
            'betas': betas,
            'expression': expression,
        }
        
        for key in body_parms:
            body_parms[key] = body_parms[key].reshape(-1, body_parms[key].shape[-1])
        
        self.smplx_model.to(res.device)
        with torch.no_grad():
            output = self.smplx_model(**body_parms)
            
        joints = output.joints
        joints = joints.reshape(batch_size, seq_len, -1, 3)
        return joints[:, :, :55]
    
    
    def process_smplx_joints(self, joints):
        if self.cfg.EXPER.motion_part == 'body':
            return joints[:, :, :22]
        elif self.cfg.EXPER.motion_repre == 'ske':
            index = [
                *range(0, 22),
                22, 25, 28, 31, 34, 37, 40, 43, 46, 49
            ]
            joints_new = joints[:, :, index]
            return joints_new
        else:
            index = [
                *range(0, 22),
                25, 28, 31, 34, 37, 40, 43, 46, 49, 52
            ]
            joints_new = joints[:, :, index]
            return joints
    
    
    def _process_text(self, sentence):
        sentence = sentence.replace('-', '')
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == 'NOUN'
                    or token.pos_ == 'VERB') and (word != 'left'):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list
    
    
    def _get_t2m_evaluator(self, cfg):
        """
        load T2M text encoder and motion encoder for evaluating
        """
        # init module
        self.t2m_textencoder = instantiate_from_config(cfg.METRIC.TM2T.t2m_textencoder)


        # load pretrianed
        if self.dataname == "kit":
            dataname = "kit"
        else:
            dataname = "t2m"

        t2m_checkpoint = torch.load(os.path.join(
            cfg.METRIC.TM2T.t2m_path, dataname, "text_mot_match/model/finest.tar"),
                                    map_location='cpu')
        self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])

        # freeze params
        self.t2m_textencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
    
    
    def _get_text_embeddings(self, texts):
        word_embs = []
        pos_ohot = []
        text_lengths = []
        for i, sentence in enumerate(texts):
            word_list, pos_list = self._process_text(sentence.strip())
            t_tokens = [
                '%s/%s' % (word_list[i], pos_list[i])
                for i in range(len(word_list))
            ]

            if len(t_tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'
                                   ] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = t_tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(torch.tensor(pos_oh).float()[None])
                word_embeddings.append(torch.tensor(word_emb).float()[None])
            text_lengths.append(sent_len)
            pos_ohot.append(torch.cat(pos_one_hots, dim=0)[None])
            word_embs.append(torch.cat(word_embeddings, dim=0)[None])

        word_embs = torch.cat(word_embs, dim=0).to(self.Bleu_1)
        pos_ohot = torch.cat(pos_ohot, dim=0).to(self.Bleu_1)
        text_lengths = torch.tensor(text_lengths).to(self.Bleu_1)

        align_idx = np.argsort(text_lengths.data.tolist())[::-1].copy()

        # get text embeddings
        text_embeddings = self.t2m_textencoder(word_embs[align_idx],
                                               pos_ohot[align_idx],
                                               text_lengths[align_idx])

        original_text_embeddings = text_embeddings.clone()

        for idx, sort in enumerate(align_idx):
            original_text_embeddings[sort] = text_embeddings[idx]

        return original_text_embeddings