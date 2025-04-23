import random
import numpy as np
from torch.utils import data
from .dataset_t2m import Text2MotionDataset
import codecs as cs
import os
from os.path import join as pjoin
import copy
import json
from rich.progress import track

class Text2MotionDatasetToken(data.Dataset):

    def __init__(
        self,
        cfg,
        data_root,
        split,
        mean,
        std,
        transform_mean,
        transform_std,
        max_motion_length=196,
        min_motion_length=40,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        self.cfg=cfg
        # Data mean and std
        self.mean = mean
        self.std = std
        self.transform_mean = transform_mean
        self.transform_std = transform_std
        # Data path
        # split_file = pjoin(data_root, split + '.txt')
        # motion_dir = pjoin(data_root, 'new_joint_vecs')
        # text_dir = pjoin(data_root, 'texts')

        # # Data id list
        # self.id_list = []
        # with cs.open(split_file, "r") as f:
        #     for line in f.readlines():
        #         self.id_list.append(line.strip())
                
        # new_name_list = []
        # length_list = []
        # data_dict = {}
        # for name in self.id_list:
        #     try:
        #         motion = np.load(pjoin(motion_dir, name + '.npy'))
        #         if (len(motion)) <  self.min_motion_length or (len(motion) >= 200):
        #             continue

        #         data_dict[name] = {'motion': motion,
        #                         'length': len(motion),
        #                         'name': name}
        #         new_name_list.append(name)
        #         length_list.append(len(motion))
        #     except:
        #         # Some motion may not exist in KIT dataset
        #         pass

        # self.length_arr = np.array(length_list)
        # self.data_dict = data_dict
        # self.name_list = new_name_list
        # self.nfeats = motion.shape[-1]
    
    
        data_root = "/mnt/AFS_jiangjianping/datasets/SEA_processed/tmp_data"
        
        configs = {
            'dlp': {
                'text_embeddings': "/mnt/AFS_jiangjianping/datasets/SEA_processed/DLP-MoCap/embeddings/embeddings.npz",
                'dataset_items': "/mnt/AFS_jiangjianping/datasets/SEA_processed/DLP-MoCap/dataset_items.json",
                'dataset_root_dir': "/mnt/AFS_jiangjianping/datasets/SEA_processed/DLP-MoCap",
            },
            'humanml3d': {
                'text_embeddings': "/mnt/AFS_jiangjianping/datasets/SEA_processed/HumanML3D/embeddings/embeddings.npz",
                'dataset_items': "/mnt/AFS_jiangjianping/datasets/SEA_processed/HumanML3D/dataset_items.json",
                'dataset_root_dir': "/mnt/AFS_jiangjianping/datasets/SEA_processed/HumanML3D",
            },
            'interx': {
                'text_embeddings': "/mnt/AFS_jiangjianping/datasets/SEA_processed/Inter-X/embeddings/embeddings.npz",
                'dataset_items': "/mnt/AFS_jiangjianping/datasets/SEA_processed/Inter-X/dataset_items.json",
                'dataset_root_dir': "/mnt/AFS_jiangjianping/datasets/SEA_processed/Inter-X",
            },
        }
        dataset_items = {}     
        for key in configs:
            with open(configs[key]['dataset_items'], 'r') as f:
                dataset_items.update(json.load(f))
        print(f'Original Loaded total {len(dataset_items)} dataset items from all datasets')
    
        self.id_list = []
        if split == 'train':
            for key in dataset_items.keys():
                if dataset_items[key]['split_in_original'] in ['train', 'val', 'test']:
                    self.id_list.append(key)
        elif split in ['val']:
            for key in dataset_items.keys():
                if dataset_items[key]['split_in_original'] == 'val':
                    self.id_list.append(key)
        elif split in ['test']:
            for key in dataset_items.keys():
                if dataset_items[key]['split_in_original'] == 'test':
                    self.id_list.append(key)
        enumerator = enumerate(
                track(
                    self.id_list,
                    f"Loading HumanML3D {split}",
                ))
        maxdata = 1e10
        subset = ''
    
        new_name_list = []
        length_list = []
        data_dict = {}

        motion_type = self.cfg['EXPER']['motion_repre']
        motion_part = self.cfg['EXPER']['motion_part']
        
        for idx, name in enumerator:
            if len(new_name_list) > maxdata:
                break
            
            data_item = dataset_items[name]
            
            dataset = data_item['dataset']
            dataset_root = configs[dataset]['dataset_root_dir']
            motion_path = os.path.join(dataset_root, data_item['motion_data_path'])
            motion_data = np.load(motion_path, allow_pickle=True)
            motion_data = dict(motion_data)
            
            # motion = motion_data['motion']
            start_frame = data_item['start_frame']
            end_frame = data_item['end_frame']
            
            # if self.cfg['EXPER']['transform'] == True and (start_frame != 0 or end_frame != -1):
            #     continue
            
            motion = self.get_motion_feature(motion_data)
            
            motion = motion[start_frame:end_frame]
            
            transform = None
            if self.cfg['EXPER']['transform'] == True:
                transforms = motion_data['transforms'].item()
                if data_item['dataset'] == 'interx':
                    if self.cfg['EXPER']['motion_repre'] == 'ske':
                        transform = np.concatenate([transforms['ske_relative_cont6d'], transforms['ske_relative_pos']], axis=0)
                    else:
                        transform = np.concatenate([transforms['smplx_relative_cont6d'], transforms['smplx_relative_pos']], axis=0)
                    transform = transform[[0, 2, 6, 7, 8]]
            
            # if (len(motion)) < self.min_motion_length or (len(motion)>= 300):
            #     continue

            # Read text
            text_data = []
            for i in range(len(data_item['text'])):
                text_dict = {}
                text_dict['caption'] = data_item['text'][i]
                if i < len(data_item['tokens']):
                    tokens = data_item['tokens'][i].split(' ')
                    text_dict['tokens'] = tokens
                else:
                    text_dict['tokens'] = None
                text_data.append(text_dict)
            
            data_dict[name] = {
                    'motion': motion,
                    "length": len(motion),
                    'text': text_data,
                    'transform': transform,
                    'partner_motion': data_item['next_partner_motion_name'],
                }
            new_name_list.append(name)
            length_list.append(len(motion))
    
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        # self.name_list = name_list[start_id:start_id + maxdata]
        self.nfeats = data_dict[name_list[0]]['motion'].shape[1] #263 # data_dict[name_list[0]]['motion'].shape[1]
        # self.reset_max_len(self.max_length)
    
    def get_motion_feature(self, motion_data):
        motion_len = len(motion_data['ske_feature'])
        if self.cfg['EXPER']['motion_repre'] == 'ske':
            motion = motion_data['ske_feature']
            # range(0, 4 + 21 * 3)
            # range(4+51*3, 4+51*3+21*6)
            # range (4+51*9, 4 + 51*9+22*3)
            # range (4 + 51 * 9 + 52*3, 4 + 51 * 9 + 52*3 + 4)
            body_index = list(range(0, 4+21*3)) + list(range(4+51*3, 4+51*3+21*6)) + \
                list(range(4+51*9, 4+51*9+22*3)) + list(range(4+51*9+52*3, 4+51*9+52*3+4))
            hand_index = list(range(4+21*3, 4+51*3)) + list(range(4+51*3+21*6, 4+51*9)) + \
                list(range(4+51*9+22*3, 4+51*9+52*3))
            if self.cfg['EXPER']['motion_part'] == 'body':
                motion = motion[:, body_index]
            elif self.cfg['EXPER']['motion_part'] in ['body_hand_sep', 'body_hand_bind']:
                motion = motion[:, body_index + hand_index]
            else:
                raise ValueError('Unknown motion part')
        elif self.cfg['EXPER']['motion_repre'] == 'global cont6d':
            motion_smplx = motion_data['smplx_feature'].item()
            if self.cfg['EXPER']['motion_part'] == 'body':
                motion = np.concatenate([motion_smplx['root_velocity'], 
                                            motion_smplx['root_height'],
                                            motion_smplx['global_root_cont6d'],
                                            motion_smplx['cont6d_global'][:, :24].reshape(motion_len, -1)], axis=1)
            elif self.cfg['EXPER']['motion_part'] in ['body_hand_sep', 'body_hand_bind']:
                motion = np.concatenate([motion_smplx['root_velocity'], 
                                            motion_smplx['root_height'],
                                            motion_smplx['global_root_cont6d'],
                                            motion_smplx['cont6d_global'].reshape(motion_len, -1)], axis=1)
            else:
                raise ValueError('Unknown motion part')
        elif self.cfg['EXPER']['motion_repre'] == 'local cont6d':
            motion_smplx = motion_data['smplx_feature'].item()
            if self.cfg['EXPER']['motion_part'] == 'body':  
                motion = np.concatenate([motion_smplx['root_velocity'], 
                                            motion_smplx['root_height'],
                                            motion_smplx['global_root_cont6d'],
                                            motion_smplx['cont6d_local'][:, :21].reshape(motion_len, -1)], axis=1)
            elif self.cfg['EXPER']['motion_part'] in ['body_hand_sep', 'body_hand_bind']:
                motion = np.concatenate([motion_smplx['root_velocity'], 
                                            motion_smplx['root_height'],
                                            motion_smplx['global_root_cont6d'],
                                            motion_smplx['cont6d_local'].reshape(motion_len, -1)], axis=1)
            else:
                raise ValueError('Unknown motion part')
        else:
            raise ValueError('Unknown motion representation')
        
        return motion
    
    
    
    def __len__(self):
        return len(self.data_dict)  
        
    def __getitem__(self, item):
        idx = item
        data_item = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list, transform, partner_name = \
            data_item['motion'], data_item['length'], data_item['text'], data_item['transform'], data_item['partner_motion']
        
        
        ### transform
        if partner_name is None or transform is None:
            # single person transform
            transform = np.random.normal(loc=self.transform_mean, scale=self.transform_std, size=self.transform_mean.shape)
            transform = (transform - self.transform_mean) / self.transform_std
        else:
            # two person transform
            transform = (transform - self.transform_mean) / self.transform_std
        
        
        ### next partner motion
        partner_motion = None
        if partner_name is not None:
            if partner_name in self.data_dict:
                partner_motion = self.data_dict[partner_name]['motion']
        
        
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption = text_data["caption"]

        # Text
        # all_captions = []
        # for text_dic in text_list:
        #     if text_dic['tokens'] is None:
        #         continue
        #     else:
        #         all_captions.append(' '.join([token.split('/')[0] for token in text_dic['tokens']]))

        all_captions = [i['caption'] for i in text_list]
        if len(all_captions) >= 3:
            all_captions = all_captions[:3]
        else:
            for _ in range(3-len(all_captions)):
                all_captions.append(all_captions[0])

        # Crop the motions in to times of 4, and introduce small variations
        coin2 = np.random.choice(["single", "double"])
        # if coin2 == "double":
        #     m_length = (m_length // self.unit_length - 1) * self.unit_length
        # elif coin2 == "single":
        m_length = (m_length // self.unit_length) * self.unit_length
        # idx = random.randint(0, len(motion) - m_length)
        motion = motion[:m_length]

        motion = (motion - self.mean) / self.std
    
        return self.name_list[idx], motion, m_length, all_captions, transform, partner_motion
