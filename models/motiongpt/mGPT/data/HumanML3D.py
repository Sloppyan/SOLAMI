import numpy as np
import torch
import os 
from os.path import join as pjoin
from .humanml.utils.word_vectorizer import WordVectorizer
from .humanml.scripts.motion_process import process_file
from .humanml.scripts.ske_process import recover_from_ric, recover_from_smplx_feature, process_smplx_feature
from . import BASEDataModule
from .humanml import (Text2MotionDatasetEval, 
                      Text2MotionDataset, 
                      Text2MotionDatasetCB, 
                      MotionDataset, 
                      MotionDatasetVQ,
                      Text2MotionDatasetToken,
                      Text2MotionDatasetM2T,
                      InterSynthDatasetCB,
                      Text2MotionVQDatasetEval,
                      Text2MotionVQDatasetToken)
from .utils import humanml3d_collate


class HumanML3DDataModule(BASEDataModule):
    def __init__(self, cfg, **kwargs):

        super().__init__(collate_fn=humanml3d_collate)
        self.cfg = cfg
        self.save_hyperparameters(logger=False)
        
        # Basic info of the dataset
        cfg.DATASET.JOINT_TYPE = 'humanml3d'
        self.name = "humanml3d"
        self.njoints = 22 if cfg.EXPER.motion_part == 'body' else 52
        
        # Path to the dataset
        data_root = cfg.DATASET.HUMANML3D.ROOT
        self.hparams.data_root = data_root
        self.hparams.text_dir = pjoin(data_root, "texts")
        self.hparams.motion_dir = pjoin(data_root, 'new_joint_vecs')
        
        mean_var_path = "SOLAMI_data/mean_variance/all_mean_variance_post.npz"
        mean_var = np.load(mean_var_path, allow_pickle=True)
        
        if self.cfg['EXPER']['motion_repre'] == 'ske':
            motion = mean_var['ske_feature'].item()
            # range(0, 4 + 21 * 3)
            # range(4+51*3, 4+51*3+21*6)
            # range (4+51*9, 4 + 51*9+22*3)
            # range (4 + 51 * 9 + 52*3, 4 + 51 * 9 + 52*3 + 4)
            body_index = list(range(0, 4+21*3)) + list(range(4+51*3, 4+51*3+21*6)) + \
                list(range(4+51*9, 4+51*9+22*3)) + list(range(4+51*9+52*3, 4+51*9+52*3+4))
            hand_index = list(range(4+21*3, 4+51*3)) + list(range(4+51*3+21*6, 4+51*9)) + \
                list(range(4+51*9+22*3, 4+51*9+52*3))
            if self.cfg['EXPER']['motion_part'] == 'body':
                self.hparams.mean = motion['mean'][body_index]
                self.hparams.std = motion['std'][body_index]
            elif self.cfg['EXPER']['motion_part'] in ['body_hand_sep', 'body_hand_bind']:
                self.hparams.mean = motion['mean'][body_index + hand_index]
                self.hparams.std = motion['std'][body_index + hand_index]
            else:
                raise ValueError('Unknown motion part')
        elif self.cfg['EXPER']['motion_repre'] == 'global cont6d':
            motion_smplx = mean_var['smplx_feature'].item()
            if self.cfg['EXPER']['motion_part'] == 'body':
                self.hparams.mean = np.concatenate([motion_smplx['root_velocity']['mean'], 
                                            motion_smplx['root_height']['mean'],
                                            motion_smplx['global_root_cont6d']['mean'],
                                            motion_smplx['cont6d_global']['mean'][:24].reshape(-1)], axis=0)
                self.hparams.std = np.concatenate([motion_smplx['root_velocity']['std'],
                                                    motion_smplx['root_height']['std'],
                                                    motion_smplx['global_root_cont6d']['std'],
                                                    motion_smplx['cont6d_global']['std'][:24].reshape(-1)], axis=0)
            elif self.cfg['EXPER']['motion_part'] in ['body_hand_sep', 'body_hand_bind']:
                self.hparams.mean = np.concatenate([motion_smplx['root_velocity']['mean'], 
                                                    motion_smplx['root_height']['mean'],
                                                    motion_smplx['global_root_cont6d']['mean'],
                                                    motion_smplx['cont6d_global']['mean'].reshape(-1)], axis=0)
                self.hparams.std = np.concatenate([motion_smplx['root_velocity']['std'],
                                                    motion_smplx['root_height']['std'],
                                                    motion_smplx['global_root_cont6d']['std'],
                                                    motion_smplx['cont6d_global']['std'].reshape(-1)], axis=0)
            else:
                raise ValueError('Unknown motion part')
        elif self.cfg['EXPER']['motion_repre'] == 'local cont6d':
            motion_smplx = mean_var['smplx_feature'].item()
            if self.cfg['EXPER']['motion_part'] == 'body':  
                self.hparams.mean = np.concatenate([motion_smplx['root_velocity']['mean'], 
                                            motion_smplx['root_height']['mean'],
                                            motion_smplx['global_root_cont6d']['mean'],
                                            motion_smplx['cont6d_local']['mean'][:21].reshape(-1)], axis=0)
                self.hparams.std = np.concatenate([motion_smplx['root_velocity']['std'],
                                                    motion_smplx['root_height']['std'],
                                                    motion_smplx['global_root_cont6d']['std'],
                                                    motion_smplx['cont6d_local']['std'][:21].reshape(-1)], axis=0)
            elif self.cfg['EXPER']['motion_part'] in ['body_hand_sep', 'body_hand_bind']:
                self.hparams.mean = np.concatenate([motion_smplx['root_velocity']['mean'], 
                                            motion_smplx['root_height']['mean'],
                                            motion_smplx['global_root_cont6d']['mean'],
                                            motion_smplx['cont6d_local']['mean'].reshape(-1)], axis=0)
                self.hparams.std = np.concatenate([motion_smplx['root_velocity']['std'],
                                                    motion_smplx['root_height']['std'],
                                                    motion_smplx['global_root_cont6d']['std'],
                                                    motion_smplx['cont6d_local']['std'].reshape(-1)], axis=0)
            else:
                raise ValueError('Unknown motion part')
        else:
            raise ValueError('Unknown motion representation')
        self.hparams.std = np.where(self.hparams.std == 0, 1e-9, self.hparams.std)
        
        
        if self.cfg['EXPER']['transform'] == True:
            transforms = mean_var['transforms'].item()
            if self.cfg['EXPER']['motion_repre'] == 'ske':
                self.hparams.transform_mean = np.concatenate([transforms['ske_relative_cont6d']['mean'], transforms['ske_relative_pos']['mean']], axis=0)
                self.hparams.transform_std = np.concatenate([transforms['ske_relative_cont6d']['std'], transforms['ske_relative_pos']['std']], axis=0)
            else:
                self.hparams.transform_mean = np.concatenate([transforms['smplx_relative_cont6d']['mean'], transforms['smplx_relative_pos']['mean']], axis=0)
                self.hparams.transform_std = np.concatenate([transforms['smplx_relative_cont6d']['std'], transforms['smplx_relative_pos']['std']], axis=0)
            # self.hparams.transform_std = np.where(self.hparams.transform_std == 0, 1e-9, self.hparams.transform_std)
            self.hparams.transform_std = self.hparams.transform_std[[0, 2, 6, 7, 8]]
            self.hparams.transform_mean_all = self.hparams.transform_mean
            self.hparams.transform_mean = self.hparams.transform_mean[[0, 2, 6, 7, 8]]
        else:
            self.hparams.transform_mean = None
            self.hparams.transform_std = None
        
        
        # Length of the dataset
        self.hparams.max_motion_length = cfg.DATASET.HUMANML3D.MAX_MOTION_LEN
        self.hparams.min_motion_length = cfg.DATASET.HUMANML3D.MIN_MOTION_LEN
        self.hparams.max_text_len = cfg.DATASET.HUMANML3D.MAX_TEXT_LEN
        self.hparams.unit_length = cfg.DATASET.HUMANML3D.UNIT_LEN

        # Additional parameters
        self.hparams.debug = cfg.DEBUG
        self.hparams.stage = cfg.TRAIN.STAGE
        self.hparams.w_vectorizer = None
        self.DatasetEval = Text2MotionDatasetEval

        if cfg.TRAIN.STAGE == "vae":
            if cfg.model.params.motion_vae.target.split('.')[-1].lower() == "vqvae":
                self.hparams.win_size = 64
                self.Dataset = MotionDatasetVQ
                self.DatasetEval = Text2MotionVQDatasetEval
            else:
                self.Dataset = MotionDataset
        elif 'lm' in cfg.TRAIN.STAGE:
            self.hparams.code_path = cfg.DATASET.CODE_PATH
            self.hparams.task_path = cfg.DATASET.TASK_PATH
            self.hparams.std_text = cfg.DATASET.HUMANML3D.STD_TEXT
            self.Dataset = Text2MotionDatasetCB
            if hasattr(self.hparams.cfg.TRAIN, "TASK") and self.hparams.cfg.TRAIN.TASK == "interaction":
                self.Dataset = InterSynthDatasetCB
                self.DatasetEval = InterSynthDatasetCB
                self.hparams.scripts_path = cfg.DATASET.HUMANML3D.SCRIPTS_PATH
        elif cfg.TRAIN.STAGE == "token":
            # self.Dataset = Text2MotionDatasetToken
            # self.DatasetEval = Text2MotionDatasetToken
            self.Dataset = Text2MotionVQDatasetToken
            self.DatasetEval = Text2MotionVQDatasetToken
        elif cfg.TRAIN.STAGE == "m2t":
            self.Dataset = Text2MotionDatasetM2T
            self.DatasetEval = Text2MotionDatasetM2T
        else:
            self.Dataset = Text2MotionDataset

        # Get additional info of the dataset
        self._sample_set = self.get_sample_set(overrides={"split": "test", "tiny": True})
        self.nfeats = self._sample_set.nfeats
        cfg.DATASET.NFEATS = self.nfeats
        
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

    def feats2joints(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        
        if self.cfg.EXPER.motion_repre == 'ske':
            if self.njoints > 25:
                features = self.revserse_ric_data(features)
            res = recover_from_ric(features, self.njoints)
        elif self.cfg.EXPER.motion_repre == 'global cont6d':
            res = recover_from_smplx_feature(features, 'global')
            # res_test = process_smplx_feature(res, 'global')
        elif self.cfg.EXPER.motion_repre == 'local cont6d':
            res = recover_from_smplx_feature(features, 'local')
            # res_test = process_smplx_feature(res.clone(), 'local')
            # reconstruction error is normal, considering the padding sequence
        else:
            raise ValueError('Unknown motion representation')

        return res

    def joints2feats(self, features):
        # TODO
        example_data = np.load(os.path.join(self.hparams.data_root, 'joints', '000021.npy'))
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)
        features = process_file(features, self.njoints, example_data, 't2m')[0]
        return features

    def normalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = (features - mean) / std
        return features

    def denormalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return features


    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return features

    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.METRIC.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
