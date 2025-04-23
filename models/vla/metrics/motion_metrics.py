import os
import torch
from torch import Tensor
from typing import List
from torchmetrics import Metric
import numpy as np
from .utils import *
from metrics.features.kinetic_torch import cal_average_kinetic_energy_torch, normalize
from motion.smplx_process import recover_from_smplx_feature
sys.path.append('tools/smplx')
import smplx
import spacy
# from bert_score import score as score_bert
import copy


class MTMetrics(Metric):
    def __init__(self, 
                device='cuda',
                task='t2m',
                bleu_k=4,
                max_text_len=40,
                force_in_meter: bool = True,
                align_root: bool = True,
                dist_sync_on_step=True,
                diversity_times=300,
                **kwargs):
        
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.default_device = device
        self.task = task
        self.metrics = []
        self.max_text_len = max_text_len
        self.bleu_k = bleu_k
        # Fid
        self.add_state("FID", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.append("FID")

        self.add_state("pred_valid", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.append("pred_valid")

        self.add_state("angle_error",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.metrics.append("angle_error")
        
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
        
        self.model_path = 'SOLAMI_data/SMPL_MODELS/smplx/SMPLX_MALE.npz'
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
        self.pred_texts = []
        self.gt_texts = []
        
        if self.task in ['m2t']:
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

            
            from nlgmetricverse import NLGMetricverse, load_metric
            metrics = [
                load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}),
                load_metric("bleu", resulting_name="bleu_4", compute_kwargs={"max_order": 4}),
                load_metric("rouge"),
                load_metric("cider"),
            ]
            self.nlg_evaluator = NLGMetricverse(metrics)
        pass
    
    def reset(self):
        self.count = torch.tensor(0.)
        self.count_seq = torch.tensor(0.)
        self.pred_valid = torch.tensor(0.)
        self.recmotion_embeddings = []
        self.gtmotion_embeddings = []
        self.angle_error = torch.tensor(0.)
        self.MPJPE = torch.tensor([0.0])
        self.PAMPJPE = torch.tensor([0.0])
        self.ACCEL = torch.tensor([0.0])
        if self.task in ['m2t']:
            self.pred_texts = []
            self.gt_texts = []
        pass
    
    
    @torch.no_grad()
    def compute(self):
        count = self.count.item()
        count_seq = self.count_seq.item()
        pred_valid = self.pred_valid.item()
        metrics = {metric: getattr(self, metric) for metric in self.metrics}
        if pred_valid == 0:
            self.reset()
            self.gt_texts = []
            self.pred_texts = []
            return {**metrics}
        
        # assert pred_valid == len(self.recmotion_embeddings)
        
        
        metrics['pred_valid'] = torch.tensor(pred_valid / count_seq).to(self.default_device)
        
        if self.task in ['t2m', 'm2m']:
        
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
            metrics["FID"] = torch.tensor(calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)).to(self.default_device)

            # Compute diversity
            if pred_valid > self.diversity_times:
                metrics["Diversity"] = torch.tensor(calculate_diversity_np(all_genmotions,
                                                            self.diversity_times)).to(self.default_device)
                metrics["gt_Diversity"] = torch.tensor(calculate_diversity_np(
                    all_gtmotions, self.diversity_times)).to(self.default_device)

            
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
            mr_metrics["angle_error"] = self.angle_error / count
            metrics.update(mr_metrics)
        
        if self.task in ['m2t']:
        
            # NLP scores
            pred_texts = copy.deepcopy(self.pred_texts)
            gt_texts = copy.deepcopy(self.gt_texts)
            assert len(pred_texts) == len(gt_texts)
            if len(gt_texts) != 0:
                
                scores = self.nlg_evaluator(predictions=gt_texts,
                                            references=gt_texts)
                for k in [1, self.bleu_k]:
                    metrics[f"Bleu_{str(k)}"] = torch.tensor(scores[f'bleu_{str(k)}']['score'],
                                                            device=self.default_device)
                    
                metrics["ROUGE_L"] = torch.tensor(scores["rouge"]["rougeL"],
                                                device=self.default_device)
                metrics["CIDEr"] = torch.tensor(scores["cider"]['score'],device=self.default_device)

        # Bert metrics
        # P, R, F1 = score_bert(pred_texts,
        #                       gt_texts,
        #                       lang='en',
        #                       rescale_with_baseline=True,
        #                       idf=True,
        #                       device=self.default_device,
        #                       verbose=False)

        # metrics["Bert_F1"] = F1.mean()
        
        
        # Reset
        self.reset()
        self.gt_texts = []
        self.pred_texts = []
        
        return {**metrics}
    
    def log_metrics(self, metrics):
        # for metric in metrics:
        ret_metrics = {}
        if self.task in ['t2m', 'm2m']:
            for key in ['MPJPE', 'PAMPJPE', 'ACCEL', 'FID', 'Diversity', 'gt_Diversity', 'pred_valid', 'angle_error']:
                print(f"{key}: {metrics[key].item()}")
                ret_metrics[key] = metrics[key].item()
        elif self.task in ['m2t']:
            for key in ['Bleu_1', 'Bleu_4', 'ROUGE_L', 'CIDEr', 'pred_valid']:
                print(f"{key}: {metrics[key].item()}")
                ret_metrics[key] = metrics[key].item()
        else:
            pass
        return ret_metrics
    
    @torch.no_grad()
    def update(self,
                feats_ref: Tensor = None,
                feats_rst: Tensor = None,
                gt_texts: str = None,
                pred_texts: str = None,
                task: str = 't2m',
               ):
        if task in ['t2m', 'm2m']:
            self.count_seq += 1
            
            if feats_rst is not None:
                self.count += len(feats_ref)
                if type(feats_rst) == np.ndarray:
                    feats_rst = torch.tensor(feats_rst, dtype=torch.float32).unsqueeze(0).to(self.default_device)
                if type(feats_ref) == np.ndarray:
                    feats_ref = torch.tensor(feats_ref, dtype=torch.float32).unsqueeze(0).to(self.default_device)
                self.pred_valid += 1
                gtmotion_embeddings, ref = self.get_motion_embeddings(feats_ref, [feats_ref.shape[1]])
                self.gtmotion_embeddings.append(gtmotion_embeddings)
                
                recmotion_embeddings, rst = self.get_motion_embeddings(feats_rst, [feats_rst.shape[1]])
                self.recmotion_embeddings.append(recmotion_embeddings)
                
                assert rst.dim() == 4
                if self.align_root:
                    align_inds = [0]
                else:
                    align_inds = None

                for i in range(1):
                    len_tmp = min(feats_ref.shape[1], feats_rst.shape[1])
                    self.MPJPE += torch.sum(
                        calc_mpjpe(rst[i][: len_tmp], ref[i][: len_tmp], align_inds=align_inds))
                    self.PAMPJPE += torch.sum(calc_pampjpe(rst[i][: len_tmp], ref[i][: len_tmp]))
                    self.ACCEL += torch.sum(calc_accel(rst[i][: len_tmp], ref[i][: len_tmp]))
                    self.angle_error += torch.sum(calculate_rotvec_error(rst[i][: len_tmp, 3:], ref[i][: len_tmp, 3:]))
                
        elif task in ['m2t']:      
            #### use valid data to calculate the metrics
            self.count_seq += 1
            if pred_texts is not [] or None:
                self.pred_texts.append(pred_texts)
                self.gt_texts.append(gt_texts)
                self.pred_valid += 1
        else:
            pass
        
    
    
    def get_motion_embeddings(self, feats: Tensor, lengths: List[int]):
        # step 1 get original data
        joints_ = self.smplx_infer(feats)
        joints = self.process_smplx_joints(joints_)        
           
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

        body_pose = res[..., 6:6+21*3]
        left_hand_pose = res[..., 6+21*3: 6+(21+15)*3]
        right_hand_pose = res[..., 6+36*3:]
        jaw_pose = torch.zeros([batch_size, seq_len, 3], dtype=torch.float32).to(res.device)
        leye_pose = torch.zeros([batch_size, seq_len, 3], dtype=torch.float32).to(res.device)
        reye_pose = torch.zeros([batch_size, seq_len, 3], dtype=torch.float32).to(res.device)
        
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
        index = [
            *range(0, 22),
            25, 28, 31, 34, 37, 40, 43, 46, 49, 52
        ]
        joints_new = joints[:, :, index]
        return joints_new
    
    
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
    
    