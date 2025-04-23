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
from motion.smplx_process import recover_from_smplx_feature

class MotionTextDataset(data.Dataset):

    def __init__(
        self,
        data_root = "SOLAMI_data/tmp_data",
        dataset_config = {},
        mean=None,
        std=None,
        transform_mean=None,
        transform_std=None,
        max_motion_length=196,
        min_motion_length=40,
        unit_length=4,
        tmpFile=True,
        tiny=False,
    ):

        # restrian the length of motion and text
        self.max_length = 20
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length

        self.mean = mean
        self.std = std
        self.transform_mean = transform_mean
        self.transform_std = transform_std

        if dataset_config == {}:
            configs = {
                'dlp': {
                    'text_embeddings': "SOLAMI_data/DLP-MoCap/embeddings/embeddings.npz",
                    'dataset_items': "SOLAMI_data/DLP-MoCap/dataset_items_post.json",
                    'dataset_root_dir': "SOLAMI_data/DLP-MoCap/unified_data",
                },
                'humanml3d': {
                    'text_embeddings': "SOLAMI_data/HumanML3D/embeddings/embeddings.npz",
                    'dataset_items': "SOLAMI_data/HumanML3D/dataset_items_post.json",
                    'dataset_root_dir': "SOLAMI_data/HumanML3D/unified_data",
                },
                'interx': {
                    'text_embeddings': "SOLAMI_data/Inter-X/embeddings/embeddings.npz",
                    'dataset_items': "SOLAMI_data/Inter-X/dataset_items_post.json",
                    'dataset_root_dir': "SOLAMI_data/Inter-X/unified_data",
                },
            }
        else:
            configs = dataset_config
        dataset_items = {}     
        for key in configs:
            with open(configs[key]['dataset_items'], 'r') as f:
                dataset_items.update(json.load(f))
        print(f'Original Loaded total {len(dataset_items)} dataset items from all datasets')
        
        self.id_list =  list(dataset_items.keys())
        
        if tiny:
            start_id = 25000
            maxdata = 400
            # self.id_list = self.id_list[start_id:]
            enumerator = enumerate(self.id_list[start_id:])
            
            subset = '_tiny'
        else:
            enumerator = enumerate(
                track(
                    self.id_list,
                    f"Loading Data",
                ))
            maxdata = 1e9
            subset = ''

        new_name_list = []
        length_list = []
        data_dict = {}


        # Fast loading
        if os.path.exists(pjoin(data_root, f'tmp_7B_pretrain/{subset}_data.pkl')):
            if tiny:
                with open(pjoin(data_root, f'tmp_7B_pretrain/{subset}_data.pkl'),
                          'rb') as file:
                    data_dict = pickle.load(file)
            else:
                with rich.progress.open(
                        pjoin(data_root, f'tmp_7B_pretrain/{subset}_data.pkl'),
                        'rb',
                        description=f"Loading Data") as file:
                    data_dict = pickle.load(file)
            with open(pjoin(data_root, f'tmp_7B_pretrain/{subset}_index.pkl'),
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
                motion_path = os.path.join(dataset_root, name + '.npz')
                motion_data = np.load(motion_path, allow_pickle=True)
                motion_data = dict(motion_data)
                             
                motion = self.get_motion_feature(motion_data)
                
                
                transforms = motion_data['transforms'].item()
                if 'smplx_relative_cont6d' in transforms:
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
                os.makedirs(pjoin(data_root, 'tmp_7B_pretrain'), exist_ok=True)
                with open(pjoin(data_root, f'tmp_7B_pretrain/{subset}_data.pkl'),
                          'wb') as file:
                    pickle.dump(data_dict, file)
                with open(pjoin(data_root, f'tmp_7B_pretrain/{subset}_index.pkl'),
                          'wb') as file:
                    pickle.dump(name_list, file)

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.nfeats = data_dict[name_list[0]]['motion'].shape[1] #263 # data_dict[name_list[0]]['motion'].shape[1]
    
    def get_motion_feature(self, motion_data):
        motion_len = len(motion_data['ske_feature'])
        motion_smplx = motion_data['smplx_feature'].item()
        motion = np.concatenate([motion_smplx['root_velocity'], 
                                    motion_smplx['root_height'],
                                    motion_smplx['global_root_cont6d'],
                                    motion_smplx['cont6d_local'].reshape(motion_len, -1)], axis=1)
        return motion
    
    
    def __len__(self):
        return len(self.name_list)
    
    def get_data_from_name(self, name):
        data_item_name, motion, m_length, all_captions, transform, partner_motion =  self.__getitem__(name)
        if motion is not None:
            motion = motion * self.std + self.mean
            smplx_params = recover_from_smplx_feature(motion, 'local cont6d')
            smplx_params = smplx_params.numpy()
            transform = transform * self.transform_std + self.transform_mean
        else:
            smplx_params = None
        return data_item_name, smplx_params, m_length, all_captions, transform, partner_motion
    
    def __getitem__(self, item):
        if type(item) is not int:
            if item not in self.name_list:
                return None, None, None, None, None, None
            else:
                data_item_name = item
        else:
            data_item_name = self.name_list[item]
            
        data_item = self.data_dict[data_item_name]
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
        all_captions = [i['caption'] for i in text_list]
        if len(all_captions) >= 3:
            all_captions = all_captions[:3]
        else:
            for _ in range(3-len(all_captions)):
                all_captions.append(all_captions[0])

        # Crop the motions in to times of 4, and introduce small variations
        # coin2 = np.random.choice(["single", "double"])
        # if coin2 == "double":
        #     m_length = (m_length // self.unit_length - 1) * self.unit_length
        # elif coin2 == "single":
        m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        motion = (motion - self.mean) / self.std
    
        return data_item_name, motion, m_length, all_captions, transform, partner_motion
