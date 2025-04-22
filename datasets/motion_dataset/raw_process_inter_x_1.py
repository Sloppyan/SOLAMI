import os
import numpy as np 
import sys
import torch
sys.path.append('tools/smplx')
import smplx
from tqdm import tqdm
import logging
import argparse

male_model_path = 'SOLAMI_data/SMPL_MODELS/smplx/SMPLX_MALE.npz'
female_model_path = 'SOLAMI_data/SMPL_MODELS/smplx/SMPLX_FEMALE.npz'
neutral_model_path = 'SOLAMI_data/SMPL_MODELS/smplx/SMPLX_NEUTRAL.npz'

model_paths = {
    'male': male_model_path,
    'female': female_model_path,
    'neutral': neutral_model_path,
}

betas = np.array([-0.06134899, -0.4861751 ,  0.8630473 , -3.07320443,  1.10772016,
       -1.44656493,  2.97690664, -1.12731489,  1.24817344, -1.4111463 ,
       -0.04035034, -0.29547926,  0.38509519,  0.13750311,  0.94445029,
       -0.47172116])

# trans_matrix = np.array([[1.0, 0.0, 0.0],
#                             [0.0, 0.0, 1.0],
#                             [0.0, 1.0, 0.0]])
trans_matrix = np.array([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])

SMPLX_MODELS = {}


def smplx_to_pose(gpu_id, data_root, data_name, save_root_dir, logger, fps=30, ex_fps=120):
    if gpu_id not in [0, 1, 2, 3, 4, 5, 6, 7]:
        DEVICE = 'cpu'
    else:
        DEVICE = 'cuda:%d'%gpu_id 
        
    for file_name in ['P1.npz', 'P2.npz']:
        try:
            file_path = os.path.join(data_root, data_name, file_name)
            bdata = np.load(file_path, allow_pickle=True)
            save_path = os.path.join(save_root_dir, data_name, file_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            num_betas = bdata['betas'].shape[1]
            
            gender = str(bdata['gender'])
            if gender not in SMPLX_MODELS:
                bm = smplx.create(model_paths[gender], model_type='smplx', gender=gender, 
                                ext='npz', num_betas=num_betas, use_pca=False, flat_hand_mean=True)
                bm = bm.to(DEVICE)
            else:
                bm = SMPLX_MODELS[gender]
            
            # if bdata['gender'] == 'male':
            #     bm = smplx.create(male_model_path, model_type='smplx', gender='male', ext='npz', num_betas=num_betas, use_pca=False, flat_hand_mean=True)
            # elif bdata['gender'] == 'neutral':
            #     bm = smplx.create(neutral_model_path, model_type='smplx', gender='neutral', ext='npz', num_betas=num_betas, use_pca=False, flat_hand_mean=True)
            # else:
            #     bm = smplx.create(female_model_path, model_type='smplx', gender='female', ext='npz', num_betas=num_betas, use_pca=False, flat_hand_mean=True)

            comp_device = DEVICE
            down_sample = int(ex_fps / fps)
            bdata_trans = torch.Tensor(bdata['trans'][::down_sample]).to(comp_device)
            batch_size = len(bdata_trans)
            jaw_pose = torch.zeros([batch_size, 3], dtype=torch.float32).to(comp_device)
            leye_pose = torch.zeros([batch_size, 3], dtype=torch.float32).to(comp_device)
            reye_pose = torch.zeros([batch_size, 3], dtype=torch.float32).to(comp_device)
            expression = torch.zeros([batch_size, 10], dtype=torch.float32).to(comp_device)
            body_parms = {
                    'global_orient': torch.Tensor(bdata['root_orient'][::down_sample]).to(comp_device),
                    'body_pose': torch.Tensor(bdata['pose_body'][::down_sample]).reshape(batch_size, -1).to(comp_device),
                    'jaw_pose': jaw_pose,
                    'leye_pose': leye_pose,
                    'reye_pose': reye_pose,
                    'left_hand_pose': torch.Tensor(bdata['pose_lhand'][::down_sample]).reshape(batch_size, -1).to(comp_device),
                    'right_hand_pose': torch.Tensor(bdata['pose_rhand'][::down_sample]).reshape(batch_size, -1).to(comp_device),
                    'transl': bdata_trans,
                    'betas': torch.Tensor(np.repeat(bdata['betas'], repeats=batch_size, axis=0)).to(comp_device),
                    'expression': expression,
                }
            
            with torch.no_grad():
                output = bm(**body_parms)

            joints = output.joints.cpu().numpy()
            t_root_J = output.t_root_J.cpu().numpy()
            pose_seq_np_n = np.dot(joints, trans_matrix)
            
            data_npz = {
                'pose_np': pose_seq_np_n,
                'global_orient': bdata['root_orient'][::down_sample],
                'body_pose': bdata['pose_body'][::down_sample].reshape(batch_size, -1),
                'jaw_pose': jaw_pose.cpu().numpy(),
                'leye_pose': leye_pose.cpu().numpy(),
                'reye_pose': reye_pose.cpu().numpy(),
                'left_hand_pose': bdata['pose_lhand'][::down_sample].reshape(batch_size, -1),
                'right_hand_pose': bdata['pose_rhand'][::down_sample].reshape(batch_size, -1),
                'transl': bdata_trans.cpu().numpy(),
                'betas': np.repeat(bdata['betas'], repeats=batch_size, axis=0),
                'expression': expression.cpu().numpy(),
                't_root_J': t_root_J,
                'gender': bdata['gender'],
            }
            
            np.savez(save_path, **data_npz)
        except:
            logger.info('Error in processing %s'%file_path)
            continue
        torch.cuda.empty_cache()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Process Inter-X dataset')
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--period', type=int, default=4)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    
    data_root = 'SOLAMI_data/Inter-X/motions'
    save_root_dir = 'SOLAMI_data/Inter-X/joints'
    data_names_all = os.listdir(data_root)
    
    os.makedirs(save_root_dir, exist_ok=True)
    
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    log_file_path = os.path.join(os.path.dirname(save_root_dir),
                                 'process1_period{}_part{}.log'.format(args.period, args.part))
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
    
    
    data_names = data_names_all[args.part::args.period]
    
    if args.debug:
        data_names = data_names[:5]
    
    for data_name in tqdm(data_names):
        smplx_to_pose(args.gpu_id, data_root, data_name, save_root_dir, logger)
    