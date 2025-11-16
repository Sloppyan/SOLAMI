"""
Obtain original dataset items from the original dataset files.

"""

import os
from tqdm import tqdm
import json
import logging
# from DLPMoCap import DLPMoCap_ActionDatabase
import numpy as np

LENS = 1000000


## humanml3d
def get_humanml3d_items(save_path, logger):
    split_ids = {
        'train': [],
        'val': [],
        'test': [],
    }
    
    dataset_items = {}
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            dataset_items = json.load(f)
    
    annot_dir = "SOLAMI_data/HumanML3D/HumanML3D/HumanML3D"
    joints_dir = "SOLAMI_data/HumanML3D/HumanML3D_no_mirror"
    
    
    train_path = os.path.join(annot_dir, 'train.txt')
    with open(train_path, 'r') as f:
        for line in f:
            split_ids['train'].append(line.strip())
    
    val_path = os.path.join(annot_dir, 'val.txt')
    with open(val_path, 'r') as f:
        for line in f:
            split_ids['val'].append(line.strip())
    
    test_path = os.path.join(annot_dir, 'test.txt')
    with open(test_path, 'r') as f:
        for line in f:
            split_ids['test'].append(line.strip())
    
    texts_dir = os.path.join(annot_dir, 'texts')
    file_names = os.listdir(texts_dir)
    file_names = [f for f in file_names if f.endswith('.txt')]
    texts_path = [os.path.join(texts_dir, f) for f in file_names]
    

    texts_path = texts_path[:LENS]
    
    for idx in tqdm(range(len(texts_path))):
        try:
            motion_file_name = file_names[idx].replace('.txt', '')
            joint_dir_tmp = os.path.join(joints_dir, motion_file_name + '.npz')
            if not os.path.exists(joint_dir_tmp):
                continue
            # feat = np.load(joint_dir_tmp, allow_pickle=True)
            # feat = dict(feat)
            # if np.isnan(feat['ske_feature']).any() or np.isnan(feat['transforms'].item()['ske_forward']).any():
            #     logger.info(f"Error in {joint_dir_tmp}")
            #     continue
            with open(texts_path[idx], 'r') as f:
                lines =  f.readlines()
                motion_idx = 0
                
                if motion_file_name in split_ids['train']:
                    split = 'train'
                elif motion_file_name in split_ids['val']:
                    split = 'val'
                elif motion_file_name in split_ids['test']:
                    split = 'test'
                else:
                    split = None
                
                frame_to_motion_name = {}
                
                for line in lines:
                    data = line.strip().replace('\n', '').split('#')
                    text = data[0].strip()
                    if text.endswith('.'):
                        text = text[:-1]
                    tokens = data[1]
                    if data[2] == 'nan':
                        data[2] = 0
                    if data[3] == 'nan':
                        data[3] = 0
                    start_frame = int(float(data[2]) * 30)
                    end_frame = int(float(data[3]) * 30)
                    if start_frame == 0 and end_frame == 0:
                        start_frame = 0
                        end_frame = -1
                    if (start_frame, end_frame) not in frame_to_motion_name:
                        curr_item = {
                            'dataset': 'humanml3d',
                            'motion_name': 'humanml3d--' + motion_file_name + '--' + str(motion_idx),
                            'motion_data_path': file_names[idx].replace('.txt', '.npz'),
                            # 'motion_feature_path': os.path.join('new_joint_vecs', file_names[idx].replace('.txt', '.npy')),
                            # 'motion_transform_path': os.path.join('new_joints', file_names[idx].replace('.txt', '_transform.npy')),
                            'split_in_original': split,
                            
                            'text': [text],
                            'tokens': [tokens],
                            'both_text': [],
                            'emotion': '',
                            
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'fps': 30,
                            'interactive_start_frame': 0,
                            'interactive_end_frame': 0,
                            
                            'actor': 'Null',
                            'last_partner_motion_name': None,
                            'next_partner_motion_name': None,
                        }
                        motion_idx += 1
                        frame_to_motion_name[(start_frame, end_frame)] = curr_item['motion_name']
                        dataset_items[curr_item['motion_name']] = curr_item
                    else:
                        curr_item = dataset_items[frame_to_motion_name[(start_frame, end_frame)]]
                        curr_item['text'].append(text)
                        curr_item['tokens'].append(tokens)
        except:
            logger.info(f"Error in {texts_path[idx]}")
            continue
    # save
    with open(save_path, 'w') as f:
        json.dump(dataset_items, f, indent=4)
    logger.info(f"Finish humanml3d dataset")
    logger.info(f"Total items: {len(dataset_items)}")
                    

## inter x dataset
def get_inter_x_items(save_path, logger):
    split_ids = {
        'train': [],
        'val': [],
        'test': [],
    }
    
    split_dir = "SOLAMI_data/Inter-X/datasets"
    annot_dir = "SOLAMI_data/Inter-X/texts_post_processed"
    joints_dir = "SOLAMI_data/Inter-X/joints"
    
    train_path = os.path.join(split_dir, 'train.txt')
    with open(train_path, 'r') as f:
        for line in f:
            split_ids['train'].append(line.strip())
    
    val_path = os.path.join(split_dir, 'val.txt')
    with open(val_path, 'r') as f:
        for line in f:
            split_ids['val'].append(line.strip())
    
    test_path = os.path.join(split_dir, 'test.txt')
    with open(test_path, 'r') as f:
        for line in f:
            split_ids['test'].append(line.strip())
            
    dataset_items = {}
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            dataset_items = json.load(f)
    
    file_names = os.listdir(annot_dir)
    file_names = [f for f in file_names if f.endswith('.json')]
    
    file_paths = [os.path.join(annot_dir, f) for f in file_names]
    
    file_paths = file_paths[:LENS]
    
    for idx in tqdm(range(len(file_paths))):
        try:
            motion_file_name = file_names[idx].split('.')[0]
            
            joints_path_dir = os.path.join(joints_dir, motion_file_name,  'P1.npz')
            if not os.path.exists(joints_path_dir):
                continue
            # feat1 = np.load(joints_path_dir, allow_pickle=True)
            # feat1 = dict(feat1)
            # if np.isnan(feat1['ske_feature']).any() or np.isnan(feat1['transforms'].item()['ske_forward']).any():
            #     logger.info(f"Error in {joints_path_dir}")
            #     continue
            
            # feat2 = np.load(os.path.join(joints_dir, motion_file_name,  'P2.npz'), allow_pickle=True)
            # feat2 = dict(feat2)
            # if np.isnan(feat2['ske_feature']).any() or np.isnan(feat1['transforms'].item()['ske_forward']).any():
            #     logger.info(f"Error in {joints_path_dir} P2")
            #     continue
            
            if motion_file_name in split_ids['train']:
                split = 'train'
            elif motion_file_name in split_ids['val']:
                split = 'val'
            elif motion_file_name in split_ids['test']:
                split = 'test'
            else:
                split = None
            
            with open(file_paths[idx], 'r') as f:
                data = json.load(f)
                
                actor_texts = []
                reactor_texts = []
                both_texts = []
                for text_data in data['texts']:
                    if text_data['option'] in ['Y', 'R']:
                        text_data['actor'] = text_data['actor'].strip() 
                        text_data['reactor'] = text_data['reactor'].strip()
                        if text_data['actor'].endswith('.'):
                            text_data['actor'] = text_data['actor'][:-1]
                        if text_data['reactor'].endswith('.'):
                            text_data['reactor'] = text_data['reactor'][:-1]
                        actor_texts.append(text_data['actor'])
                        reactor_texts.append(text_data['reactor'])
                    both_texts.append(text_data['both'])
                if len(actor_texts) == 0:
                    continue
                
                if data['actor'] == 'P1':
                    texts_1 = actor_texts
                    texts_2 = reactor_texts
                    actor_1 = 'Y'
                    actor_2 = 'N'
                    last_partner_motion_name_1 = None
                    next_partner_motion_name_1 = 'interx--' + motion_file_name + '--1'
                    last_partner_motion_name_2 = 'interx--' + motion_file_name + '--0'
                    next_partner_motion_name_2 = None
                else:
                    texts_1 = reactor_texts
                    texts_2 = actor_texts
                    actor_1 = 'N'
                    actor_2 = 'Y'
                    last_partner_motion_name_1 = 'interx--' + motion_file_name + '--1'
                    next_partner_motion_name_1 = None
                    last_partner_motion_name_2 = None
                    next_partner_motion_name_2 = 'interx--' + motion_file_name + '--0'
                
                curr_item_1 = {
                    'dataset': 'interx',
                    'motion_name': 'interx--' + motion_file_name + '--0',
                    'motion_data_path': os.path.join(motion_file_name, 'P1.npz'),
                    # 'motion_feature_path': os.path.join('new_joint_vecs', motion_file_name, 'P1.npy'),
                    # 'motion_transform_path': os.path.join('new_joints', motion_file_name, 'P1_transform.npy'),
                    'split_in_original': split,
                    
                    'text': texts_1,
                    'tokens': [],
                    'both_text': both_texts,
                    'emotion': '',
                    
                    'start_frame': 0,
                    'end_frame': -1,
                    'fps': 30,
                    'interactive_start_frame': 0,
                    'interactive_end_frame': 0,
                    
                    'actor': actor_1,
                    'last_partner_motion_name': last_partner_motion_name_1,
                    'next_partner_motion_name': next_partner_motion_name_1,
                }
                dataset_items[curr_item_1['motion_name']] = curr_item_1
                curr_item_2 = {
                    'dataset': 'interx',
                    'motion_name': 'interx--' + motion_file_name + '--1',
                    'motion_data_path': os.path.join(motion_file_name, 'P2.npz'),
                    # 'motion_feature_path': os.path.join('new_joint_vecs', motion_file_name, 'P2.npy'),
                    # 'motion_transform_path': os.path.join('new_joints', motion_file_name, 'P2_transform.npy'),
                    'split_in_original': split,
                    
                    'text': texts_2,
                    'tokens': [],
                    'both_text': both_texts,
                    'emotion': '',
                    
                    'start_frame': 0,
                    'end_frame': -1,
                    'fps': 30,
                    'interactive_start_frame': 0,
                    'interactive_end_frame': 0,
                    
                    'actor': actor_2,
                    'last_partner_motion_name': last_partner_motion_name_2,
                    'next_partner_motion_name': next_partner_motion_name_2,
                }
                dataset_items[curr_item_2['motion_name']] = curr_item_2
        except:
            logger.info(f"Error in {file_paths[idx]}")
            continue
    # save
    with open(save_path, 'w') as f:
        json.dump(dataset_items, f, indent=4)  
        
    logger.info(f"Finish interx dataset")
    logger.info(f"Total items: {len(dataset_items)}")
    pass
         

## dlp dataset
'''
def get_dlp_items(save_path, logger):
    config = {
    "database": {
    "annotated_atomic_path": 'SOLAMI_data/DLP-MoCap/annot/Script_Annotation_V0_atmoic.csv',
    "annotated_short_path": 'SOLAMI_data/DLP-MoCap/annot/Script_Annotation_V0_short.csv',
    "motion_data_dir": '',
    "smplx_data_dir": '/mnt/AFS_jiangjianping/datasets/DLP-MoCap', #'\\10.4.11.59\Zoehuman\DL\smplx_h\'
    "text_to_smplx_data_path": None,
    "smplx_data_npz_path": None, #'.\digital_life_project\motion_database\smplx_data.npz'
    "load_npz": False
    },
    }
    actionset = DLPMoCap_ActionDatabase(config=config)
    actions = actionset.actions
    actions = actions[:LENS]
    
    dataset_items = {}
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            dataset_items = json.load(f)
    
    
    for action in tqdm(actions):
        try:
            file_name = action.file_name
            Px = 'P1' if action.file_path_smplx_sub.split('/')[-1].split('.')[0].split('_')[-1] == '00' else 'P2'
            
            joint_dir_tmp = os.path.join('SOLAMI_data/DLP-MoCap/processed_data', file_name, f'{Px}.npz')
            if not os.path.exists(joint_dir_tmp):
                continue
            # feat = np.load(joint_dir_tmp, allow_pickle=True)
            # feat = dict(feat)
            # if np.isnan(feat['ske_feature']).any() or np.isnan(feat['transforms'].item()['ske_forward']).any():
            #     logger.info(f"Error in {joint_dir_tmp}")
            #     continue
            
            motion_name = 'dlp--' + action.index
            action.sub_action_description = action.sub_action_description.strip()
            if action.sub_action_description.endswith('.'):
                action.sub_action_description = action.sub_action_description[:-1]
            curr_item = {
                'dataset': 'dlp',
                'motion_name': 'dlp--' + action.index,
                'motion_data_path': os.path.join(file_name, f'{Px}.npz'),
                # 'motion_feature_path': os.path.join('new_joint_vecs', file_name, f'{Px}.npy'),
                # 'motion_transform_path': os.path.join('new_joints', file_name, f'{Px}_transform.npy'),
                'split_in_original': 'train',
                
                'text': [action.sub_action_description],
                'tokens': [],
                'both_text': [action.action_description_detailed_both],
                'emotion': action.sub_emotion,
                
                'start_frame': int(action.sub_start_frame / action.fbx_ratio),
                'end_frame': int(action.sub_end_frame / action.fbx_ratio),
                'fps': 30,
                'interactive_start_frame': int(action.A_B_interact_start_frame / action.fbx_ratio),
                'interactive_end_frame': int(action.A_B_interact_end_frame / action.fbx_ratio),
                
                'actor': 'Null',
                'last_partner_motion_name': 'dlp--' + action.last_other_action.index if action.last_other_action else None,
                'next_partner_motion_name': 'dlp--' + action.next_other_action.index if action.next_other_action else None,
            }
            dataset_items[motion_name] = curr_item
        except:
            logger.info(f"Error in {file_name}")
            continue
    
    with open(save_path, 'w') as f:
        json.dump(dataset_items, f, indent=4)
    logger.info(f"Finish dlp dataset")
    logger.info(f"Total items: {len(dataset_items)}")
    pass
    '''


def get_logger(save_path):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    log_file_path = os.path.join(save_path)
    print('logger file: ', log_file_path)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger




if __name__ == '__main__':
    
    log_path = 'SOLAMI_data/HumanML3D/dataset_items.log'
    logger = get_logger(log_path)
    
    save_path = 'SOLAMI_data/HumanML3D/dataset_items_pre.json'
    get_humanml3d_items(save_path, logger)
    
    # save_path = 'SOLAMI_data/Inter-X/dataset_items_pre.json'
    # get_inter_x_items(save_path, logger)
    
    # save_path = 'SOLAMI_data/DLP-MoCap/dataset_items_pre.json'
    # get_dlp_items(save_path, logger)
    pass