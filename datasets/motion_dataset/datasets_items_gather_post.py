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
    
    joints_dir = "SOLAMI_data/HumanML3D/unified_data"
    data_items_pre_path = "SOLAMI_data/HumanML3D/dataset_items_pre.json"
    
    with open(data_items_pre_path, 'r') as f:
        dataset_items_info = json.load(f)
    dataset_items_info = dict(sorted(dataset_items_info.items()))
    motion_name_list = list(dataset_items_info.keys())
    motion_name_list = motion_name_list[:LENS]
    # texts_path = texts_path[:LENS]
    
    for idx in tqdm(range(len(motion_name_list))):
        try:
            # motion_file_name = file_names[idx].replace('.txt', '')
            motion_file_name = motion_name_list[idx]
            joint_dir_tmp = os.path.join(joints_dir, motion_file_name + '.npz')
            if not os.path.exists(joint_dir_tmp):
                dataset_items_info.pop(motion_file_name)
                continue
            feat = np.load(joint_dir_tmp, allow_pickle=True)
            feat = dict(feat)
            if np.isnan(feat['ske_feature']).any() or np.isnan(feat['transforms'].item()['ske_forward']).any():
                dataset_items_info.pop(motion_file_name)
                logger.info(f"Error in {joint_dir_tmp} with NaN Value")
                continue
        except:
            logger.info(f"Error in {texts_path[idx]}")
            continue
    # save
    with open(save_path, 'w') as f:
        json.dump(dataset_items_info, f, indent=4)
    logger.info(f"Finish humanml3d dataset")
    logger.info(f"Total items: {len(dataset_items_info)}")
                    

## inter x dataset
def get_inter_x_items(save_path, logger):
    split_ids = {
        'train': [],
        'val': [],
        'test': [],
    }
    
    split_dir = "SOLAMI_data/Inter-X/datasets"
    # annot_dir = "SOLAMI_data/Inter-X/texts_post_processed"
    joints_dir = "SOLAMI_data/Inter-X/unified_data"
    data_items_pre_path = "SOLAMI_data/Inter-X/dataset_items_pre.json"
    
    with open(data_items_pre_path, 'r') as f:
        dataset_items_info = json.load(f)
    dataset_items_info = dict(sorted(dataset_items_info.items()))
    motion_name_list = list(dataset_items_info.keys())
    motion_name_list = motion_name_list[:LENS]
    # file_paths = file_paths[:LENS]
    
    def remove_item(motion_file_name, dataset_items_info):
        if motion_file_name in dataset_items_info:
            current_item = dataset_items_info[motion_file_name]
            if current_item['last_partner_motion_name']:
                last_partner_motion_name = current_item['last_partner_motion_name']
                if last_partner_motion_name in dataset_items_info:
                    dataset_items_info[last_partner_motion_name]['next_partner_motion_name'] = None
            if current_item['next_partner_motion_name']:
                next_partner_motion_name = current_item['next_partner_motion_name']
                if next_partner_motion_name in dataset_items_info:
                    dataset_items_info[next_partner_motion_name]['last_partner_motion_name'] = None
        dataset_items_info.pop(motion_file_name)
        return dataset_items_info
    
    for idx in tqdm(range(len(motion_name_list))):
        try:
            # motion_file_name = file_names[idx].replace('.txt', '')
            motion_file_name = motion_name_list[idx]
            joint_dir_tmp = os.path.join(joints_dir, motion_file_name + '.npz')
            if not os.path.exists(joint_dir_tmp):
                dataset_items_info = remove_item(motion_file_name, dataset_items_info)
                continue
            feat = np.load(joint_dir_tmp, allow_pickle=True)
            feat = dict(feat)
            if np.isnan(feat['ske_feature']).any() or np.isnan(feat['transforms'].item()['ske_forward']).any():
                dataset_items_info = remove_item(motion_file_name, dataset_items_info)
                logger.info(f"Error in {joint_dir_tmp} with NaN Value")
                continue
        except:
            logger.info(f"Error in {joint_dir_tmp}")
            continue
    
    with open(save_path, 'w') as f:
        json.dump(dataset_items_info, f, indent=4)  
        
    logger.info(f"Finish interx dataset")
    logger.info(f"Total items: {len(dataset_items_info)}")
    pass
         

'''
def get_dlp_items(save_path, logger):
    # split_dir = "SOLAMI_data/Inter-X/datasets"
    # annot_dir = "SOLAMI_data/Inter-X/texts_post_processed"
    joints_dir = "SOLAMI_data/DLP-MoCap/unified_data"
    data_items_pre_path = "SOLAMI_data/DLP-MoCap/dataset_items_pre.json"
    
    with open(data_items_pre_path, 'r') as f:
        dataset_items_info = json.load(f)
    dataset_items_info = dict(sorted(dataset_items_info.items()))
    motion_name_list = list(dataset_items_info.keys())
    motion_name_list = motion_name_list[:LENS]
    # file_paths = file_paths[:LENS]
    
    def remove_item(motion_file_name, dataset_items_info):
        if motion_file_name in dataset_items_info:
            current_item = dataset_items_info[motion_file_name]
            if current_item['last_partner_motion_name']:
                last_partner_motion_name = current_item['last_partner_motion_name']
                if last_partner_motion_name in dataset_items_info:
                    dataset_items_info[last_partner_motion_name]['next_partner_motion_name'] = None
            if current_item['next_partner_motion_name']:
                next_partner_motion_name = current_item['next_partner_motion_name']
                if next_partner_motion_name in dataset_items_info:
                    dataset_items_info[next_partner_motion_name]['last_partner_motion_name'] = None
        dataset_items_info.pop(motion_file_name)
        return dataset_items_info
    
    for idx in tqdm(range(len(motion_name_list))):
        try:
            # motion_file_name = file_names[idx].replace('.txt', '')
            motion_file_name = motion_name_list[idx]
            joint_dir_tmp = os.path.join(joints_dir, motion_file_name + '.npz')
            if not os.path.exists(joint_dir_tmp):
                dataset_items_info = remove_item(motion_file_name, dataset_items_info)
                continue
            feat = np.load(joint_dir_tmp, allow_pickle=True)
            feat = dict(feat)
            if np.isnan(feat['ske_feature']).any() or np.isnan(feat['transforms'].item()['ske_forward']).any():
                dataset_items_info = remove_item(motion_file_name, dataset_items_info)
                logger.info(f"Error in {joint_dir_tmp} with NaN Value")
                continue
        except:
            logger.info(f"Error in {joint_dir_tmp}")
            continue
    with open(save_path, 'w') as f:
        json.dump(dataset_items_info, f, indent=4)  
        
    logger.info(f"Finish DLP dataset")
    logger.info(f"Total items: {len(dataset_items_info)}")
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
    
    log_path = 'SOLAMI_data/HumanML3D/dataset_items_post.log'
    logger = get_logger(log_path)
    
    # save_path = 'SOLAMI_data/HumanML3D/dataset_items_post.json'
    # get_humanml3d_items(save_path, logger)
    
    save_path = 'SOLAMI_data/Inter-X/dataset_items_post.json'
    get_inter_x_items(save_path, logger)
    
    save_path = 'SOLAMI_data/DLP-MoCap/dataset_items_post.json'
    get_dlp_items(save_path, logger)
    pass