"""
Datasets
left smplh  right smplx
./amass_data/  
./amass_data/ACCAD/  --> ACCAD
./amass_data/BioMotionLab_NTroje/  --> BMLrub
./amass_data/BMLhandball/  XX
./amass_data/BMLmovi/   --> BMLmovi
./amass_data/CMU/  --> CMU
./amass_data/DFaust_67/  --> DFaust
./amass_data/EKUT/  --> EKUT
./amass_data/Eyes_Japan_Dataset/  --> Eyes_Japan_Dataset
./amass_data/HumanEva/  --> HumanEva
./amass_data/KIT/  --> KIT
./amass_data/MPI_HDM05/  --> HDM05
./amass_data/MPI_Limits/  XX
./amass_data/MPI_mosh/  --> Mosh
./amass_data/SFU/  --> SFU
./amass_data/SSM_synced/  --> SSM
./amass_data/TCD_handMocap/  --> TCDHands
./amass_data/TotalCapture/  --> TotalCapture
./amass_data/Transitions_mocap/  --> Transitions

"""

import os
import numpy as np 
import sys
sys.path.append('SOLAMI_data/HumanML3D/HumanML3D')
import torch
sys.path.append('tools/smplx')
import smplx
from tqdm import tqdm
import argparse
import pandas as pd
import time
import logging
import roma
import copy

male_model_path = 'SOLAMI_data/SMPL_MODELS/smplx/SMPLX_MALE.npz'
female_model_path = 'SOLAMI_data/SMPL_MODELS/smplx/SMPLX_FEMALE.npz'
neutral_model_path = 'SOLAMI_data/SMPL_MODELS/smplx/SMPLX_NEUTRAL.npz'

model_paths = {
    'male': male_model_path,
    'female': female_model_path,
    'neutral': neutral_model_path,
}

# TODO Warning dataset_name = root.split('/')[6]

full_smplh_mappings_to_smplx= {
    'ACCAD': 'ACCAD',
    'BioMotionLab_NTroje': 'BMLrub',
    'BMLmovi': 'BMLmovi',
    'CMU': 'CMU',
    'DFaust_67': 'DFaust',
    'EKUT': 'EKUT',
    'Eyes_Japan_Dataset': 'Eyes_Japan_Dataset',
    'HumanEva': 'HumanEva',
    'KIT': 'KIT',
    'MPI_HDM05': 'HDM05',
    'MPI_mosh': 'Mosh',
    'SFU': 'SFU',
    'SSM_synced': 'SSM',
    'TCD_handMocap': 'TCDHands',
    'TotalCapture': 'TotalCapture',
    'Transitions_mocap': 'Transitions',
}


smplx_data_dir = 'kit_mocap/amass/smplx/'
no_mirror_save_root = 'SOLAMI_data/HumanML3D/HumanML3D_no_mirror/'
index_path = 'SOLAMI_data/HumanML3D/HumanML3D/index.csv'
mirror_save_root = 'SOLAMI_data/HumanML3D/HumanML3D_mirror'


# for smplh_dataset_name, smplx_dataset_name in smplh_mappings_to_smplx.items():
#     smplx_dataset_dir = os.path.join(smplx_data_dir, smplx_dataset_name)
def main(gpu_id, fps=30, period=4, part=0, debug=False):
    os.makedirs(no_mirror_save_root, exist_ok=True)
    os.makedirs(mirror_save_root, exist_ok=True)
    
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    log_file_path = os.path.join(os.path.dirname(os.path.dirname(no_mirror_save_root)), 'process1_period{}_part{}.log'.format(period, part))
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
    
    
    if gpu_id not in [0, 1, 2, 3, 4, 5, 6, 7]:
        DEVICE = 'cpu'
    else:
        DEVICE = 'cuda:%d'%gpu_id 
    ex_fps = fps
    
    smplh_mappings_to_smplx = copy.deepcopy(full_smplh_mappings_to_smplx)
    
    keys_smplh = list(smplh_mappings_to_smplx.keys())
    for i, key in enumerate(keys_smplh):
        if i % period != part:
            smplh_mappings_to_smplx.pop(key)
    
    def dirname_include_dataset_name(dir, dataset_names):
        flag = False
        dataset_name_ = None
        for dataset_name in dataset_names:
            if dataset_name in dir:
                flag = True
                dataset_name_ = dataset_name
                break
        return flag, dataset_name_

    paths = []
    folders = []
    dataset_names = []
    for root, dirs, files in os.walk(smplx_data_dir):
        if dirname_include_dataset_name(root, smplh_mappings_to_smplx.values())[0]:
            folders.append(root)
            for name in files:
                # TODO depends on your root
                dataset_name = root.split('/')[6]
                if dataset_name in smplh_mappings_to_smplx.values():
                    if dataset_name not in dataset_names:
                        dataset_names.append(dataset_name)
                    if name.endswith('.npz'):
                        paths.append(os.path.join(root, name))
        pass
    pass

    logger.info('Found %d files'%len(paths))
    logger.info('Found %d folders'%len(folders))

    save_folders = [folder.replace(smplx_data_dir, no_mirror_save_root) for folder in folders]
    for folder in save_folders:
        os.makedirs(folder, exist_ok=True)
    group_path = [[path for path in paths if name in path] for name in dataset_names]


    ### original humanml3d code
    trans_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0],
                                [0.0, 1.0, 0.0]])

    def amass_to_pose(src_path, save_path):
        bdata = np.load(src_path, allow_pickle=True)
        fps = 0
        try:
            # print(bdata.files)
            if 'mocap_frame_rate' in bdata.files:
                fps = bdata['mocap_frame_rate']
            else:
                fps = bdata['mocap_framerate']
            frame_number = bdata['trans'].shape[0]
            num_betas = bdata['betas'].shape[0]
        except:
            logger.info('AMASS to Pose Error in : %s'%src_path)
            logger.info(bdata.files)
            return fps
        
        if bdata['gender'] == 'male':
            bm = smplx.create(male_model_path, model_type='smplx', gender='male', ext='npz', num_betas=num_betas, use_pca=False, flat_hand_mean=True)
        elif bdata['gender'] == 'neutral':
            bm = smplx.create(neutral_model_path, model_type='smplx', gender='neutral', ext='npz', num_betas=num_betas, use_pca=False, flat_hand_mean=True)
        else:
            bm = smplx.create(female_model_path, model_type='smplx', gender='female', ext='npz', num_betas=num_betas, use_pca=False, flat_hand_mean=True)
        down_sample = int(fps / ex_fps)


        comp_device = DEVICE
        bdata_trans = bdata['trans'][::down_sample]
        batch_size = len(bdata_trans)
        body_parms = {
                'global_orient': torch.Tensor(bdata['root_orient'][::down_sample]).to(comp_device),
                'body_pose': torch.Tensor(bdata['pose_body'][::down_sample]).to(comp_device),
                'jaw_pose': torch.Tensor(bdata['pose_jaw'][::down_sample]).to(comp_device),
                'leye_pose': torch.Tensor(bdata['pose_eye'][::down_sample, :3]).to(comp_device),
                'reye_pose': torch.Tensor(bdata['pose_eye'][::down_sample, 3:]).to(comp_device),
                'left_hand_pose': torch.Tensor(bdata['pose_hand'][::down_sample, :45]).to(comp_device),
                'right_hand_pose': torch.Tensor(bdata['pose_hand'][::down_sample, 45:]).to(comp_device),
                'transl': torch.Tensor(bdata_trans).to(comp_device),
                'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=batch_size, axis=0)).to(comp_device),
                'expression': torch.zeros([batch_size, 10], dtype=torch.float32).to(comp_device),
            }
        
        bm = bm.to(comp_device)
        with torch.no_grad():
            output = bm(**body_parms)

        joints = output.joints.cpu().numpy()
        t_root_J = output.t_root_J.cpu().numpy()
        pose_seq_np_n = np.dot(joints, trans_matrix)
        
        data_save = {
            'pose_np': pose_seq_np_n,
        }

        data_save.update({
            'global_orient': bdata['root_orient'][::down_sample],
            'body_pose': bdata['pose_body'][::down_sample],
            'jaw_pose': bdata['pose_jaw'][::down_sample],
            'leye_pose': bdata['pose_eye'][::down_sample, :3],
            'reye_pose': bdata['pose_eye'][::down_sample, 3:],
            'left_hand_pose': bdata['pose_hand'][::down_sample, :45],
            'right_hand_pose': bdata['pose_hand'][::down_sample, 45:],
            'transl': bdata_trans,
            'betas': np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=batch_size, axis=0),
            'expression': np.zeros([batch_size, 10]),
            't_root_J': t_root_J,
            'gender': bdata['gender'],
        })
        
        np.savez(save_path, **data_save)
        return fps
    if debug:
        paths_len = 3
    else:
        paths_len = 1000000
    save_paths = []

    # group_path = group_path[:group_len]
    all_count = sum([len(paths[:paths_len]) for paths in group_path])
    cur_count = 0

    # group_path.append(['/mnt/AFS_datasets/kit_mocap/amass/smplx/KIT/4/WalkInCounterClockwiseCircle02_stageii.npz'])

    for paths in group_path:
        dataset_name = paths[0].split('/')[-2]
        pbar = tqdm(paths[:paths_len])
        pbar.set_description('Processing: %s'%dataset_name)
        fps = 0
        for path in pbar:
            save_path = path.replace(smplx_data_dir, no_mirror_save_root)
            save_path = save_path[:-3] + 'npz'
            # TODO continue going
            fps = amass_to_pose(path, save_path)
            save_paths.append(save_path)
            torch.cuda.empty_cache()
            
        cur_count += len(paths)
        logger.info('Processed / All (fps %d): %d/%d'% (fps, cur_count, all_count) )
        time.sleep(0.1)
        

    def swap_left_right(data_npz_, npy_path=None):
        data_npz = copy.deepcopy(data_npz_)
        data = data_npz['pose_np']
        assert len(data.shape) == 3 and data.shape[-1] == 3
        data[..., 0] *= -1
        right_chain = [2, 5, 8, 11, 14, 17, 19, 21, 24]
        left_chain = [1, 4, 7, 10, 13, 16, 18, 20, 23]
        left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
        right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
        
        ## this is smplx model
        left_hand_chain = [item + 3 for item in left_hand_chain]
        right_hand_chain = [item + 3 for item in right_hand_chain]
        
        tmp = data[:, right_chain]
        data[:, right_chain] = data[:, left_chain]
        data[:, left_chain] = tmp
        if data.shape[1] > 24:
            tmp = data[:, right_hand_chain]
            data[:, right_hand_chain] = data[:, left_hand_chain]
            data[:, left_hand_chain] = tmp
        
        data_m = data_npz
        data_m['pose_np'] = data
        
        if npy_path is not None:
            npy_path = npy_path.replace('.npy', 'M.npy')
            np.save(npy_path.replace('.npy', '_ori.npy'), data)
    
        for key in data_m.keys():
            if key in ['global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose']:
                data_m[key][..., 1::3] *= -1
                data_m[key][..., 2::3] *= -1

        
        right_chain_pose = [id-1 for id in right_chain[:-1]]
        left_chain_pose = [id-1 for id in left_chain[:-1]]
        
        body_pose = data_m['body_pose'].copy()
        lens_body = body_pose.shape[0]
        body_pose = body_pose.reshape(lens_body, -1, 3)
        tmp = body_pose[:, right_chain_pose].copy()
        body_pose[:, right_chain_pose] = body_pose[:, left_chain_pose]
        body_pose[:, left_chain_pose] = tmp
        body_pose = body_pose.reshape(lens_body, -1)
        data_m['body_pose'] = body_pose
        
        tmp = data_m['leye_pose'].copy()
        data_m['leye_pose'] = data_m['reye_pose'].copy()
        data_m['reye_pose'] = tmp
        
        tmp = data_m['left_hand_pose'].copy()
        data_m['left_hand_pose'] = data_m['right_hand_pose'].copy()
        data_m['right_hand_pose'] = tmp
        
        data_m['transl'][:, 0] = - (data_m['transl'][:, 0] + data_m['t_root_J'][:, 0]) - data_m['t_root_J'][:, 0]
        
        
        # data_save = copy.deepcopy(data_m)       
        # body_parms_new = {}
        # for key in ['global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose', 'transl', 'betas', 'expression']:
        #     body_parms_new[key] = torch.Tensor(data_save[key]).to(DEVICE)
        
        # num_betas = data_save['betas'].shape[1]
        # gender_ = str(data_save['gender'])
        # model_path = model_paths[gender_]
        # bm = smplx.create(model_path, model_type='smplx', gender=gender_, ext='npz', num_betas=num_betas, use_pca=False, flat_hand_mean=True)
        # bm.to(DEVICE)
        # with torch.no_grad():
        #     output_new = bm(**body_parms_new)

        # joints_new = output_new.joints.cpu().numpy()
        # np.save(npy_path, joints_new)
        
        return data_m


    os.makedirs(mirror_save_root, exist_ok=True)
    index_file = pd.read_csv(index_path)
    # total_amount = index_file.shape[0]
    fps = ex_fps

    index_file_src_paths_tmp = [path.replace('./pose_data/', '') for path in index_file['source_path']]
    index_file_src_paths_list = []
    for path in index_file_src_paths_tmp:
        path = path.replace(' ', '_')
        if 'ACCAD' in path:
            path = path.replace('stageii', 'poses')
        flag_tmp, dataset_name_smplh = dirname_include_dataset_name(path, smplh_mappings_to_smplx.keys())
        if flag_tmp:
            dataset_name_smplx = smplh_mappings_to_smplx[dataset_name_smplh]
            path = path.replace(dataset_name_smplh, dataset_name_smplx)
        index_file_src_paths_list.append(path)

    index_file_src_paths = np.array(index_file_src_paths_list)
    save_paths_index = []
    for i, save_path in enumerate(save_paths):
        save_name = save_path.replace(no_mirror_save_root, '')
        save_name = save_name.replace('stageii', 'poses')
        save_name = save_name.replace('.npz', '.npy')
        index = np.where(index_file_src_paths == save_name)[0]
        if len(index) == 0:
            logger.info('Index File to SMPLX Error: %s'%save_name)
            save_paths_index.append(-1)
            continue
        index = index[0]
        save_paths_index.append(index)
        

    logger.info('Swapping left and right...')
    count_swap = 0
    for i in tqdm(range(len(save_paths_index))):
        if save_paths_index[i] == -1:
            continue
        try:
            idx = save_paths_index[i]
            source_path = save_paths[i]
            new_name = index_file.loc[idx]['new_name']
            new_name = new_name.replace('.npy', '.npz')
            data_npz = np.load(source_path, allow_pickle=True)
            data_npz = dict(data_npz)
            data = data_npz['pose_np']
            start_frame = int(index_file.loc[idx]['start_frame'] / 20. * fps)
            end_frame = int(index_file.loc[idx]['end_frame'] / 20. * fps)
            if 'humanact12' not in source_path:
                start_index = None
                if 'Eyes_Japan_Dataset' in source_path:
                    start_index = 3*fps
                if 'HDM05' in source_path:
                    start_index = 3*fps
                if 'TotalCapture' in source_path:
                    start_index = 1*fps
                if 'MPI_Limits' in source_path:
                    start_index = 1*fps
                if 'Transitions' in source_path:
                    start_index = int(0.5*fps)
                if start_index is not None:
                    data = data[start_index:]
                    
                data = data[start_frame:end_frame]
                data[..., 0] *= -1
                data_npz['pose_np'] = data
                
                for key in data_npz.keys():
                    if key not in ['pose_np', 'gender'] and type(data_npz[key]) is np.ndarray:
                        if start_index is not None:
                            data_tmp = data_npz[key][start_index:]
                        else:
                            data_tmp = data_npz[key]
                        data_npz[key] = data_tmp[start_frame:end_frame]
                
                global_orient = data_npz['global_orient']
                global_orient = roma.rotvec_to_rotmat(torch.tensor(global_orient, dtype=torch.float32), epsilon=1e-8)
                C1 = torch.tensor([[1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0],
                                [0.0, 1.0, 0.0]], dtype=torch.float32)
                C2 = torch.tensor([[-1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]], dtype=torch.float32)
                C = C2 @ C1
                global_orient_new = C @ global_orient
                global_orient_new_rot = roma.rotmat_to_rotvec(global_orient_new)
                data_npz['global_orient'] = global_orient_new_rot.numpy()
                
                transl = data_npz['transl']
                t_root_J = data_npz['t_root_J']
                transl_new = np.dot(C.numpy(), (t_root_J + transl).T).T - t_root_J
                data_npz['transl'] = transl_new
                save_path_modified_one = None
                
                # save_path_modified_one = os.path.join(mirror_save_root.replace('HumanML3D_mirror', 'HumanML3D_mirror_smplx'), new_name)
                # os.makedirs(os.path.dirname(save_path_modified_one), exist_ok=True)
                # save_path_modified_one = save_path_modified_one.replace('.npz', '.npy')
                
                # body_parms_new = {}
                # for key in ['global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose', 'transl', 'betas', 'expression']:
                #     body_parms_new[key] = torch.Tensor(data_npz[key]).to(DEVICE)
                
                # num_betas = data_npz['betas'].shape[1]
                # gender_ = str(data_npz['gender'])
                # model_path = model_paths[gender_]
                # bm = smplx.create(model_path, model_type='smplx', gender=gender_, ext='npz', num_betas=num_betas, use_pca=False, flat_hand_mean=True)
                # bm.to(DEVICE)
                # with torch.no_grad():
                #     output_new = bm(**body_parms_new)

                # joints_new = output_new.joints.cpu().numpy()
                # np.save(save_path_modified_one, joints_new)
                # np.save(save_path_modified_one.replace('.npy', '_ori.npy'), data)
                
            else:
                continue
            
            data_m = swap_left_right(data_npz, npy_path=save_path_modified_one)
            np.savez(os.path.join(mirror_save_root, new_name), **data_npz)
            np.savez(os.path.join(mirror_save_root, 'M'+new_name), **data_m)
            count_swap += 1
        except Exception as e:
            logger.info('Swap Error: %s'%source_path)
    logger.info('Total saved: %d'%count_swap)
    logger.info('Total amount: %d'%len(save_paths_index))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Process HumanML3D dataset')
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--period', type=int, default=4)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    main(args.gpu_id, args.fps, args.period, args.part, args.debug)