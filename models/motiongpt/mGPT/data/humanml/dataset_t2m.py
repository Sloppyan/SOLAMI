import os
import rich
import random
import pickle
import codecs as cs
import numpy as np
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin
import json
import copy

class Text2MotionDataset(data.Dataset):

    def __init__(
        self,
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
        tmpFile=False,
        tiny=False,
        debug=False,
        **kwargs,
    ):

        # restrian the length of motion and text
        self.max_length = 20
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length

        # Data mean and std
        # TODO change
        self.mean = mean
        self.std = std
        self.transform_mean = transform_mean
        self.transform_std = transform_std
        self.cfg = copy.deepcopy(kwargs['cfg'])
        self.split = split
        # self.start_id = 0
        
        data_root = "SOLAMI_data/tmp_data"
        
        configs = {
            'dlp': {
                'text_embeddings': "SOLAMI_data/DLP-MoCap/embeddings/embeddings.npz",
                'dataset_items': "SOLAMI_data/DLP-MoCap/dataset_items.json",
                'dataset_root_dir': "SOLAMI_data/DLP-MoCap",
            },
            'humanml3d': {
                'text_embeddings': "SOLAMI_data/HumanML3D/embeddings/embeddings.npz",
                'dataset_items': "SOLAMI_data/HumanML3D/dataset_items.json",
                'dataset_root_dir': "SOLAMI_data/HumanML3D",
            },
            'interx': {
                'text_embeddings': "SOLAMI_data/Inter-X/embeddings/embeddings.npz",
                'dataset_items': "SOLAMI_data/Inter-X/dataset_items.json",
                'dataset_root_dir': "SOLAMI_data/Inter-X",
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
                if dataset_items[key]['split_in_original'] == 'train':
                    self.id_list.append(key)
        elif split == 'val':
            for key in dataset_items.keys():
                if dataset_items[key]['split_in_original'] == 'val':
                    self.id_list.append(key)
        elif split == 'test':
            for key in dataset_items.keys():
                if dataset_items[key]['split_in_original'] == 'test':
                    self.id_list.append(key)
        
        # Debug mode
        # start_id = 0
        # maxdata = 100
        if tiny or debug:
            # TODO Test inter mode
            if split == 'train':
                start_id = 25000
            else:
                start_id = 5000
            # self.id_list = self.id_list[start_id:]
            enumerator = enumerate(self.id_list[start_id:])
            maxdata = 100
            subset = '_tiny'
        else:
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

        # Fast loading
        if os.path.exists(pjoin(data_root, f'tmp_tokens/{split}{subset}_{motion_type}_{motion_part}_data.pkl')):
            if tiny or debug:
                with open(pjoin(data_root, f'tmp_tokens/{split}{subset}_{motion_type}_{motion_part}_data.pkl'),
                          'rb') as file:
                    data_dict = pickle.load(file)
            else:
                with rich.progress.open(
                        pjoin(data_root, f'tmp_tokens/{split}{subset}_{motion_type}_{motion_part}_data.pkl'),
                        'rb',
                        description=f"Loading HumanML3D {split}") as file:
                    data_dict = pickle.load(file)
            with open(pjoin(data_root, f'tmp_tokens/{split}{subset}_{motion_type}_{motion_part}_index.pkl'),
                      'rb') as file:
                name_list = pickle.load(file)
            for name in new_name_list:
                length_list.append(data_dict[name]['length'])
        else:
            for idx, name in enumerator:
                if len(new_name_list) > maxdata:
                    break
                
                data_item = dataset_items[name]
                
                dataset = data_item['dataset']
                dataset_root = configs[dataset]['dataset_root_dir']
                motion_path = os.path.join(dataset_root, data_item['motion_data_path'])
                motion_data = np.load(motion_path, allow_pickle=True)
                motion_data = dict(motion_data)
                
                # print('motion len: ' + str(motion_data['smplx_feature'].item()['cont6d_local'].shape[0]))
                
                # motion = motion_data['motion']
                start_frame = data_item['start_frame']
                end_frame = data_item['end_frame']
                
                if self.cfg['EXPER']['transform'] == True and (start_frame != 0 or end_frame != -1):
                    continue
                
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
                
                if (len(motion)) < self.min_motion_length or (len(motion)>= 300):
                    continue

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
                # except:
                #     pass

            name_list, length_list = zip(
                *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

            if tmpFile:
                os.makedirs(pjoin(data_root, 'tmp_tokens'), exist_ok=True)
                with open(pjoin(data_root, f'tmp_tokens/{split}{subset}_{motion_type}_{motion_part}_data.pkl'),
                          'wb') as file:
                    pickle.dump(data_dict, file)
                with open(pjoin(data_root, f'tmp_tokens/{split}{subset}_{motion_type}_{motion_part}_index.pkl'),
                          'wb') as file:
                    pickle.dump(name_list, file)

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        # self.name_list = name_list[start_id:start_id + maxdata]
        self.nfeats = data_dict[name_list[0]]['motion'].shape[1] #263 # data_dict[name_list[0]]['motion'].shape[1]
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    
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
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data_item = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list, transform, partner_name = \
            data_item['motion'], data_item['length'], data_item['text'], data_item['transform'], data_item['partner_motion']
        
        
        ### transform
        if partner_name is None:
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
        all_captions = []
        for text_dic in text_list:
            if text_dic['tokens'] is None:
                continue
            else:
                all_captions.append(' '.join([token.split('/')[0] for token in text_dic['tokens']]))

        # Crop the motions in to times of 4, and introduce small variations
        # coin2 = np.random.choice(["single", "double"])
        # if coin2 == "double":
        #     m_length = (m_length // self.unit_length - 1) * self.unit_length
        # elif coin2 == "single":
        m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        motion = (motion - self.mean) / self.std
    
        return caption, motion, m_length, None, None, None, None, all_captions, transform, partner_motion
