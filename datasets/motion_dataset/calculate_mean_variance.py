import os
import numpy as np 
import logging
import json
import sys
import torch
sys.path.append('SOLAMI_data/HumanTOMATO/src/tomato_represenation')
from common.quaternion import *
import copy
from tqdm import tqdm

import roma
LENS = 100000000


def calculate_mean_variance_ske_feature(data):
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    body_num = 22
    l_hand_num = 15
    r_hand_num = 15
    Std[0:1] = Std[0:1].mean() / 1.0
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    
    Std[4: 4+(body_num - 1) * 3] = Std[4: 4+(body_num - 1) * 3].mean() / 1.0
    Std[4+(body_num - 1) * 3: 4+(body_num - 1 + l_hand_num) * 3] = Std[4+(body_num - 1) * 3: 4+(body_num - 1 + l_hand_num) * 3].mean() / 1.0
    Std[4+(body_num - 1 + l_hand_num) * 3: 4+(body_num - 1 + l_hand_num + r_hand_num) * 3] = Std[4+(body_num - 1 + l_hand_num) * 3: 4+(body_num - 1 + l_hand_num + r_hand_num) * 3].mean() / 1.0
    
    Std[157: 157+(body_num - 1) * 6] = Std[157: 157+(body_num - 1) * 6].mean() / 1.0
    Std[157+(body_num - 1) * 6: 157 + (body_num - 1+l_hand_num) * 6] = Std[157+(body_num - 1) * 6: 157 + (body_num - 1+l_hand_num) * 6].mean() / 1.0
    Std[157+(body_num - 1+l_hand_num) * 6: 157 + (body_num - 1+l_hand_num + r_hand_num) * 6] = Std[157+(body_num - 1+l_hand_num) * 6: 157 + (body_num - 1+l_hand_num + r_hand_num) * 6].mean() / 1.0
    
    Std[463: 463+body_num*3] = Std[463: 463+body_num*3].mean() / 1.0
    Std[463+body_num*3: 463+body_num*3+l_hand_num*3] = Std[463+body_num*3: 463+body_num*3+l_hand_num*3].mean() / 1.0
    Std[463+body_num*3+l_hand_num*3: 463+body_num*3+l_hand_num*3+r_hand_num*3] = Std[463+body_num*3+l_hand_num*3: 463+body_num*3+l_hand_num*3+r_hand_num*3].mean() / 1.0
    Std[619:] = Std[619:].mean() / 1.0

    return Mean, Std, data.shape[0]

def calculate_mean_variance_smplx_feature(data):
    mean_variance_dict = {}
    for key in data:
        mean = data[key].mean(axis=0)
        std = data[key].std(axis=0)
        # if key in ['global_root_cont6d', 'root_velocity', 'root_height' ]:
        #     std[:] = std.mean()
        # elif key == 'cont6d_local':
        #     std[:21] = std[:21].mean()
        #     std[21:] = std[21:].mean()
        # elif key == 'cont6d_global':
        #     std[:21] = std[:21].mean()
        #     std[21:24] = std[21:24].mean()
        #     std[24:] = std[24:].mean()
        # else:
        #     raise ValueError(f"Unknown key: {key}")
        mean_variance_dict[key] = {
            'mean': mean,
            'std': std,
        }
    return mean_variance_dict


def calculate_mean_variance(data):
    mean_variance_dict = {}
    for key in data.keys():
        if key == 'ske_feature':
            mean, std, num = calculate_mean_variance_ske_feature(data[key])
            mean_variance_dict[key] = {
                'mean': mean,
                'std': std,
            }
            mean_variance_dict['num'] = num
        elif key == 'smplx_feature':
            mean_variance_smplx = calculate_mean_variance_smplx_feature(data[key])
            mean_variance_dict[key] = mean_variance_smplx
        elif key == 'transforms':
            mean_variance_transforms = {}
            for sub_key in data[key].keys():
                if data[key][sub_key] is None:
                    continue
                mean = data[key][sub_key].mean(axis=0)
                std = data[key][sub_key].std(axis=0)
                # if type(std) == np.ndarray:
                #     std[:] = std.mean()
                mean_variance_transforms[sub_key] = {
                    'mean': mean,
                    'std': std,
                }
            mean_variance_dict[key] = mean_variance_transforms
    return mean_variance_dict


def get_data_dicts():
    data_dicts = {
            'ske_feature': [],
            'smplx_feature': {
                'root_velocity': [],
                'root_height': [],
                'global_root_cont6d': [],
                'cont6d_local': [],
                'cont6d_global': [],
            },
            'transforms': {
                'ske_forward': [],
                'ske_root': [],
                'scale': [],
                'smplx_forward': [],
                'smplx_root': [],
                'ske_relative_pos': [],
                'ske_relative_cont6d': [],
                'smplx_relative_pos': [],
                'smplx_relative_cont6d': [],
            }
        }
    return data_dicts


def update_data_dicts(data_dicts, data):
    data_dicts['ske_feature'].append(data['ske_feature'])
    smplx_feature = data['smplx_feature'].item()
    for key in data_dicts['smplx_feature'].keys():
        data_dicts['smplx_feature'][key].append(smplx_feature[key])
    transforms = data['transforms'].item()
    for key in data_dicts['transforms'].keys():
        if key in ['ske_forward', 'smplx_forward']:
            ### HumanML3D w, xyz; roma xyz, w
            rot_mat = quaternion_to_matrix_np(transforms[key])
            cont6d = np.concatenate([rot_mat[:, :, 0], rot_mat[:, :, 1]], axis=1)
            # a1 =  quaternion_to_matrix(data_ts)
            # b1 = roma.unitquat_to_rotmat(data_ts)
            data_dicts['transforms'][key].append(cont6d)
            pass
        elif key not in transforms.keys():
            pass
        else:
            data_dicts['transforms'][key].append(transforms[key])
    return data_dicts


def concatenate_data_dicts(data_dicts):
    for key in data_dicts.keys():
        if type(data_dicts[key]) == dict:
            for sub_key in data_dicts[key].keys():
                if sub_key != 'scale':
                    if len(data_dicts[key][sub_key]) == 0:
                        data_dicts[key][sub_key] = None
                    else:
                        if sub_key in ['ske_relative_pos', 'ske_relative_cont6d', 'smplx_relative_pos', 'smplx_relative_cont6d']:
                            data_dicts[key][sub_key] = np.stack(data_dicts[key][sub_key])
                        else:
                            data_dicts[key][sub_key] = np.concatenate(data_dicts[key][sub_key], axis=0)
                else:
                    data_dicts[key][sub_key] = np.array(data_dicts[key][sub_key])
        else:
            data_dicts[key] = np.concatenate(data_dicts[key], axis=0)
    return data_dicts


def get_humanml3d_data(data_dir, logger):
    file_names = os.listdir(data_dir)
    file_paths = [os.path.join(data_dir, file_name) for file_name in file_names if file_name.endswith('.npz') ]
    file_paths = file_paths[:LENS]
    data_dicts = get_data_dicts()
    for file_path in tqdm(file_paths):
        try:
            data = np.load(file_path, allow_pickle=True)
            data = dict(data)
            if np.isnan(data['ske_feature']).any() or np.isnan(data['transforms'].item()['ske_forward']).any():
                logger.info(f"Error in {file_path}")
                continue
            
        except:
            logger.info(f"Error in {file_path}")
            continue
        
        data_dicts = update_data_dicts(data_dicts, data)
        # data_dicts['ske_feature'].append(data['ske_feature'])
        # smplx_feature = data['smplx_feature'].item()
        # for key in data_dicts['smplx_feature'].keys():
        #     data_dicts['smplx_feature'][key].append(smplx_feature[key])
        # transforms = data['transforms'].item()
        # for key in data_dicts['transforms'].keys():
        #     if key in ['ske_forward', 'smplx_forward']:
        #         ### HumanML3D w, xyz; roma xyz, w
        #         rot_mat = quaternion_to_matrix_np(transforms[key])
        #         cont6d = np.concatenate([rot_mat[:, :, 0], rot_mat[:, :, 1]], axis=1)
        #         # a1 =  quaternion_to_matrix(data_ts)
        #         # b1 = roma.unitquat_to_rotmat(data_ts)
        #         data_dicts['transforms'][key].append(cont6d)
        #         pass
        #     else:
        #         data_dicts['transforms'][key].append(transforms[key])
    data_dicts = concatenate_data_dicts(data_dicts)
    # for key in data_dicts.keys():
    #     if type(data_dicts[key]) == dict:
    #         for sub_key in data_dicts[key].keys():
    #             if sub_key != 'scale':
    #                 data_dicts[key][sub_key] = np.concatenate(data_dicts[key][sub_key], axis=0)
    #             else:
    #                 data_dicts[key][sub_key] = np.array(data_dicts[key][sub_key])
    #     else:
    #         data_dicts[key] = np.concatenate(data_dicts[key], axis=0)
    return data_dicts


def get_inter_x_data(data_dir, logger):
    file_names = os.listdir(data_dir)
    file_dirs = [os.path.join(data_dir, file_name) for file_name in file_names]
    file_dirs = file_dirs[:LENS]
    data_dicts = get_data_dicts()
    
    for file_dir in tqdm(file_dirs):
        for file_name in ['P1.npz', 'P2.npz']:
            file_path = os.path.join(file_dir, file_name)
            try:
                data = np.load(file_path, allow_pickle=True)
                data = dict(data)
                if np.isnan(data['ske_feature']).any() or np.isnan(data['transforms'].item()['ske_forward']).any():
                    logger.info(f"Error in {file_path}")
                    continue
            except:
                logger.info(f"Error in {file_path}")
                continue
            data_dicts = update_data_dicts(data_dicts, data)
    data_dicts = concatenate_data_dicts(data_dicts)
    return data_dicts


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


if __name__ == "__main__":
    save_dir = "SOLAMI_data/mean_variance"
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger(os.path.join(save_dir, "calculate_mean_variance.log"))
    
    humanml3d_data_dir = "SOLAMI_data/HumanML3D/unified_data"
    humanml3d_data = get_humanml3d_data(humanml3d_data_dir, logger)
    mean_variance_dict_hml3d = calculate_mean_variance(humanml3d_data)
    np.savez(os.path.join(save_dir, "humanml3d_mean_variance.npz"), **mean_variance_dict_hml3d)
    logger.info(f"HumanML3D data num: {mean_variance_dict_hml3d['num']}")
    # logger.info('STD')
    # logger.info(json.dumps(humanml3d_std.tolist()))
    # logger.info('MEAN')
    # logger.info(json.dumps(humanml3d_mean.tolist()))
    
    del humanml3d_data
    
    # inter_x_data_dir = "SOLAMI_data/Inter-X/processed_data"
    # inter_x_data = get_inter_x_data(inter_x_data_dir, logger)
    # mean_variance_dict_inter = calculate_mean_variance(inter_x_data)
    # np.savez(os.path.join(save_dir, "inter_x_mean_variance.npz"), **mean_variance_dict_inter)
    # logger.info(f"Inter-X data num: {mean_variance_dict_inter['num']}")
    # # logger.info('STD')
    # # logger.info(json.dumps(inter_x_std.tolist()))
    # # logger.info('MEAN')
    # # logger.info(json.dumps(inter_x_mean.tolist()))
    # 
    # del inter_x_data
    # 
    # dlp_data_dir = "SOLAMI_data/DLP-MoCap/processed_data"
    # dlp_data = get_inter_x_data(dlp_data_dir, logger)
    # mean_variance_dict_dlp = calculate_mean_variance(dlp_data)
    # np.savez(os.path.join(save_dir, "dlp_mean_variance.npz"), **mean_variance_dict_dlp)
    # logger.info(f"DLP data num: {mean_variance_dict_dlp['num']}")
    # # logger.info('STD')
    # # logger.info(json.dumps(dlp_std.tolist()))
    # # logger.info('MEAN')
    # # logger.info(json.dumps(dlp_mean.tolist()))
    # 
    # # mean_variance_dict_hml3d = np.load(os.path.join(save_dir, "humanml3d_mean_variance.npz"), allow_pickle=True)
    # # mean_variance_dict_inter = np.load(os.path.join(save_dir, "inter_x_mean_variance.npz"), allow_pickle=True)
    # # mean_variance_dict_dlp = np.load(os.path.join(save_dir, "dlp_mean_variance.npz"), allow_pickle=True)
    # 
    # # del dlp_data
    # 
    # # compute overall mean and std
    # 
    # all_num = mean_variance_dict_dlp['num'] + mean_variance_dict_inter['num'] + mean_variance_dict_hml3d['num']
    # final_mean_variance_dicts = copy.deepcopy(mean_variance_dict_inter)
    # final_mean_variance_dicts['num'] = all_num
    # for key in mean_variance_dict_inter.keys():
    #     if key == 'num':
    #         continue
    #     if 'mean' not in mean_variance_dict_inter[key]:
    #         for sub_key in mean_variance_dict_inter[key].keys():
    #             if sub_key in ['ske_relative_pos', 'ske_relative_cont6d', 'smplx_relative_pos', 'smplx_relative_cont6d']:
    #                 final_mean_variance_dicts[key][sub_key] = mean_variance_dict_inter[key][sub_key]
    #             else:
    #                 final_mean_variance_dicts[key][sub_key]['mean'] = (mean_variance_dict_hml3d[key][sub_key]['mean'] * mean_variance_dict_hml3d['num'] + \
    #                                                                 mean_variance_dict_inter[key][sub_key]['mean'] * mean_variance_dict_inter['num'] + \
    #                                                                 mean_variance_dict_dlp[key][sub_key]['mean'] * mean_variance_dict_dlp['num']) / all_num
    #                 final_mean_variance_dicts[key][sub_key]['std'] = (mean_variance_dict_hml3d[key][sub_key]['std'] * mean_variance_dict_hml3d['num'] + \
    #                                                                 mean_variance_dict_inter[key][sub_key]['std'] * mean_variance_dict_inter['num'] + 
    #                                                                 mean_variance_dict_dlp[key][sub_key]['std'] * mean_variance_dict_dlp['num']) / all_num
    #     else:
    #         final_mean_variance_dicts[key]['mean'] = (mean_variance_dict_hml3d[key]['mean'] * mean_variance_dict_hml3d['num'] + \
    #                                                   mean_variance_dict_inter[key]['mean'] * mean_variance_dict_inter['num'] + \
    #                                                   mean_variance_dict_dlp[key]['mean'] * mean_variance_dict_dlp['num']) / all_num
    #         final_mean_variance_dicts[key]['std'] = (mean_variance_dict_hml3d[key]['std'] * mean_variance_dict_hml3d['num'] + \
    #                                                  mean_variance_dict_inter[key]['std'] * mean_variance_dict_inter['num'] + 
    #                                                  mean_variance_dict_dlp[key]['std'] * mean_variance_dict_dlp['num']) / all_num
        
    # # all_data = np.concatenate([humanml3d_data, inter_x_data, dlp_data], axis=0)
    # # all_mean, all_std, all_num = calculate_mean_variance(all_data)
    # np.savez(os.path.join(save_dir, "all_mean_variance.npz"), **final_mean_variance_dicts)
    # 
    # logger.info(f"All data num: {all_num}")
    # print(final_mean_variance_dicts)

    # compute overall mean and std based only on HumanML3D data
    final_mean_variance_dicts = copy.deepcopy(mean_variance_dict_hml3d)
    all_num = mean_variance_dict_hml3d['num']
    np.savez(os.path.join(save_dir, "all_mean_variance.npz"), **final_mean_variance_dicts)
    
    logger.info(f"All data num: {all_num}")
    print(final_mean_variance_dicts)
