# coding=utf-8
# Copyright 2022 The IDEA Authors (Shunlin Lu and Ling-Hao Chen). All rights reserved.
#
# For all the datasets, be sure to read and follow their license agreements,
# and cite them accordingly.
# If the unifier is used in your research, please consider to cite as:
#
# @article{humantomato,
#   title={HumanTOMATO: Text-aligned Whole-body Motion Generation},
#   author={Lu, Shunlin and Chen, Ling-Hao and Zeng, Ailing and Lin, Jing and Zhang, Ruimao and Zhang, Lei and Shum, Heung-Yeung},
#   journal={arxiv:2310.12978},
#   year={2023}
# }
#
# @InProceedings{Guo_2022_CVPR,
#     author    = {Guo, Chuan and Zou, Shihao and Zuo, Xinxin and Wang, Sen and Ji, Wei and Li, Xingyu and Cheng, Li},
#     title     = {Generating Diverse and Natural 3D Human Motions From Text},
#     booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#     month     = {June},
#     year      = {2022},
#     pages     = {5152-5161}
# }
#
# Licensed under the IDEA License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/IDEA-Research/HumanTOMATO/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. We provide a license to use the code, 
# please read the specific details carefully.
#
# ------------------------------------------------------------------------------------------------
# Copyright (c) Chuan Guo.
# ------------------------------------------------------------------------------------------------
# Portions of this code were adapted from the following open-source project:
# https://github.com/EricGuo5513/HumanML3D
# ------------------------------------------------------------------------------------------------

import os
import numpy as np 
import sys
sys.path.append('SOLAMI_data/HumanTOMATO/src/tomato_represenation')
sys.path.append('tools/smplx')
import smplx
import roma
import argparse
import copy
from os.path import join as pjoin

from common.skeleton import Skeleton
import numpy as np
import os
from common.quaternion import *
from paramUtil import *
import torch
from tqdm import tqdm
import os
import json


male_model_path = 'SOLAMI_data/SMPL_MODELS/smplx/SMPLX_MALE.npz'
female_model_path = 'SOLAMI_data/SMPL_MODELS/smplx/SMPLX_FEMALE.npz'
neutral_model_path = 'SOLAMI_data/SMPL_MODELS/smplx/SMPLX_NEUTRAL.npz'

model_paths = {
    'male': male_model_path,
    'female': female_model_path,
    'neutral': neutral_model_path,
}


parents = [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
        16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34,
        35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52,
        53]


def findAllFile(base):
    """
    Recursively find all files in the specified directory.

    Args:
        base (str): The base directory to start the search.

    Returns:
        list: A list of file paths found in the directory and its subdirectories.
    """
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path


def uniform_skeleton(positions, target_offset):
    """
    Uniformly scales a skeleton to match a target offset.

    Args:
        positions (numpy.ndarray): Input skeleton joint positions.
        target_offset (torch.Tensor): Target offset for the skeleton.

    Returns:
        numpy.ndarray: New joint positions after scaling and inverse/forward kinematics.
    """
    # Creating a skeleton with a predefined kinematic chain
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')

    # Calculate the global offset of the source skeleton
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()

    # Calculate Scale Ratio as the ratio of legs
    src_leg_len = np.abs(src_offset[l_idx1]).max(
    ) + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max(
    ) + np.abs(tgt_offset[l_idx2]).max()

    # Scale ratio for uniform scaling
    scale_rt = tgt_leg_len / src_leg_len
    
    # Extract the root position of the source skeleton
    src_root_pos = positions[:, 0]
    # Scale the root position based on the calculated ratio
    tgt_root_pos = src_root_pos * scale_rt

    # Inverse Kinematics to get quaternion parameters
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)

    # Forward Kinematics with the new root position and target offset
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints, scale_rt


def get_original_root_rot(positions):
    origin_root = positions[0, 0]
    # Center the skeleton at the origin in the XZ plane
    root_pos_init = positions[0]

    # Ensure the initial facing direction is along Z+
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # Ensure that all poses initially face Z+
    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / \
        np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    # Calculate quaternion for root orientation
    return root_quat_init, origin_root


def get_all_original_root_rot(positions):
    origin_root = positions[:, 0].copy()
    root_pos_init = positions.copy()
    # Center the skeleton at the origin in the XZ plane
    # Ensure the initial facing direction is along Z+
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[:, r_hip] - root_pos_init[:, l_hip]
    across2 = root_pos_init[:, sdr_r] - root_pos_init[:, sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # Ensure that all poses initially face Z+
    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]).repeat(len(across), axis=0), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / \
        np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
    target = np.array([[0, 0, 1]]).repeat(len(across), axis=0)
    root_quat_init = qbetween_np(forward_init, target)
    # Calculate quaternion for root orientation
    return root_quat_init, origin_root


def process_file(input_, feet_thre):
    """
    Processes motion capture data, including downsampling, skeleton normalization,
    floor alignment, and feature extraction.

    Args:
        positions (numpy.ndarray): Motion capture data (seq_len, joints_num, 3).
        feet_thre (float): Threshold for foot detection.

    Returns:
        tuple: A tuple containing processed data, global positions, aligned positions, and linear velocity.
    """
    # Uniformly scale the skeleton to match a target offset
    positions = input_['ske_joints'].copy()
    positions = positions[:, body_joints_id+hand_joints_id, :]
    origin_quat, origin_root = get_all_original_root_rot(positions)
    positions, scale = uniform_skeleton(positions, tgt_offsets)
    
    # Put the skeleton on the floor by subtracting the minimum height
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    # Center the skeleton at the origin in the XZ plane
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # Ensure the initial facing direction is along Z+
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # Ensure that all poses initially face Z+
    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / \
        np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    # Calculate quaternion for root orientation
    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    # Rotate the motion capture data using the calculated quaternion
    # positions_b = positions.copy()
    positions = qrot_np(root_quat_init, positions)
    
    # Store the global positions for further analysis
    global_positions = positions.copy()

    # You can try to visualize it!
    # plot_3d_motion("./positions_2.mp4", kinematic_chain, positions, 'title', fps=20)
    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(positions[:, 0, 0], positions[:, 0, 2], marker='o', color='r')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array(
            [thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z)
                  < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z)
                  < velfactor)).astype(np.float32)
        return feet_l, feet_r
    
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        """
        Adjusts the motion capture data to a local pose representation and ensures
        that all poses face in the Z+ direction.

        Args:
            positions (numpy.ndarray): Input motion capture data with shape (seq_len, joints_num, 3).

        Returns:
            numpy.ndarray: Adjusted motion capture data in a local pose representation.
        """
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(
            np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        """
        Computes quaternion parameters, root linear velocity, and root angular velocity
        based on the input motion capture data.

        Args:
            positions (numpy.ndarray): Input motion capture data with shape (seq_len, joints_num, 3).

        Returns:
            tuple: A tuple containing quaternion parameters, root angular velocity, root linear velocity, and root rotation.
        """
        # Initialize a skeleton object with a specified kinematic chain
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(
            positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        """
        Computes continuous 6D parameters, root linear velocity, and root angular velocity
        based on the input motion capture data.

        Args:
            positions (numpy.ndarray): Input motion capture data with shape (seq_len, joints_num, 3).

        Returns:
            tuple: A tuple containing continuous 6D parameters, root angular velocity, root linear velocity, and root rotation.
        """
        # Initialize a skeleton object with a specified kinematic chain
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(
            positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot
    
    # Extract additional features including root height and root data
    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    # Root height
    root_y = positions[:, 0, 1:2]

    # Root rotation and linear velocity
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    # Get Joint Rotation Representation
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    # Get Joint Rotation Invariant Position Represention
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    # Get Joint Velocity Representation
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    # Concatenate all features into a single array
    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity, (origin_quat, origin_root, scale, root_quat_init)


# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)

def recover_root_rot_pos(data):
    """
    Recover root rotation and position from the given motion capture data.

    Args:
        data (torch.Tensor): Input motion capture data with shape (..., features).

    Returns:
        tuple: A tuple containing the recovered root rotation quaternion and root position.
    """
    # Extract root rotation velocity from the input data
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    """
    Recover joint positions from the given motion capture data using root rotation information.

    Args:
        data (torch.Tensor): Input motion capture data with shape (..., features).
        joints_num (int): Number of joints in the skeleton.
        skeleton (Skeleton): Skeleton object used for forward kinematics.

    Returns:
        torch.Tensor: Recovered joint positions.
    """
    # Recover root rotation quaternion and position from the input data
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    # Convert root rotation quaternion to continuous 6D representation
    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    # Define indices for relevant features in the input data
    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6

    # Extract continuous 6D parameters from the input data
    cont6d_params = data[..., start_indx:end_indx]
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    # Perform forward kinematics to obtain joint positions
    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)
    
    return positions


def recover_from_ric(data, joints_num):
    """
    Recover joint positions from the given motion capture data using root rotation information.

    Args:
        data (torch.Tensor): Input motion capture data with shape (..., features).
        joints_num (int): Number of joints in the skeleton.

    Returns:
        torch.Tensor: Recovered joint positions.
    """
    # Recover root rotation quaternion and position from the input data
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(
        positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions



# def process_smplx(data, source_data, quat_transfer, debug):
#     """
#     Recover joint positions from the given motion capture data using SMPL-X model.

#     Args:
#         source_data (torch.Tensor): Input motion capture data with shape (..., features).
#         joints_num (int): Number of joints in the skeleton.

#     Returns:
#         torch.Tensor: Recovered joint positions.
#     """
#     # Recover root rotation quaternion and position from the input data
#     r_rot_quat, r_pos = recover_root_rot_pos(data)

#     # Extract root linear velocity and root angular velocity from the input data
#     global_orient = roma.unitquat_to_rotvec(r_rot_quat).squeeze(0)
#     r_pos = r_pos.squeeze(0)
#     data = data.squeeze(0)
    
    
#     gender_ = str(source_data['gender'])
#     model_path = model_paths[gender_]
#     num_betas = source_data['betas'].shape[1]
#     bm = smplx.create(model_path, model_type='smplx', gender=gender_, ext='npz', num_betas=num_betas, use_pca=False, flat_hand_mean=True)

#     body_parms = {}
#     for key in ['global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose', 'transl', 'betas', 'expression']:
#         body_parms[key] = torch.tensor(source_data[key][:-1], dtype=torch.float32).to(DEVICE)

#     # body_parms = {
#     #     'global_orient': torch.tensor(source_data['global_orient'][:-1], dtype=torch.float32), # global_orient, #torch.tensor(source_data['global_orient'][:-1], dtype=torch.float32),
#     #     'transl': torch.tensor(source_data['transl'][:-1], dtype=torch.float32),
#     #     'betas': torch.tensor(source_data['betas'][:-1], dtype=torch.float32),
#     #     'expression': torch.tensor(source_data['expression'][:-1], dtype=torch.float32),
#     #     'body_pose': torch.tensor(source_data['body_pose'][:-1], dtype=torch.float32),
#     #     'left_hand_pose': torch.tensor(source_data['left_hand_pose'][:-1], dtype=torch.float32),
#     #     'right_hand_pose': torch.tensor(source_data['right_hand_pose'][:-1], dtype=torch.float32),
#     #     'jaw_pose': torch.tensor(source_data['jaw_pose'][:-1], dtype=torch.float32),
#     #     'leye_pose': torch.tensor(source_data['leye_pose'][:-1], dtype=torch.float32),
#     #     'reye_pose': torch.tensor(source_data['reye_pose'][:-1], dtype=torch.float32),
#     # }
#     bm = bm.to(DEVICE)
#     with torch.no_grad():
#         output = bm(**body_parms)
    
#     joints_ori = output.joints[:, body_joints_id+hand_joints_id, :].cpu().numpy()
    
#     trans_new = torch.tensor(source_data['transl'], dtype=torch.float32)
#     floor_height = joints_ori.min(axis=0).min(axis=0)[1]
#     trans_new[:, 1] -= floor_height
#     xz = joints_ori[0, 0]
#     xz[1] *= 0
#     trans_new = trans_new - xz

#     source_data.update({
#         'transl': trans_new.numpy(),
#     })
    
#     smplx_orient = source_data['global_orient'][:-1].copy()
#     smplx_orient_mat = roma.rotvec_to_rotmat(torch.Tensor(smplx_orient))
    
#     # apply transform on smplx parameters
#     euler_transfer = qeuler_np(quat_transfer, 'yzx') / 180 * np.pi
#     # since forward init vector is set in xz plane, so only rotate y axis!
#     transfer_mat = roma.euler_to_rotmat('yzx', torch.Tensor(euler_transfer[:-1, 0]))
#     smplx_orient_modified_mat = torch.matmul(transfer_mat, smplx_orient_mat)
#     smplx_orient_modified = roma.rotmat_to_rotvec(smplx_orient_modified_mat)
    
#     source_data.update({
#         'global_orient': smplx_orient_modified.numpy(),
#     })
    
#     transl = source_data['transl'][:-1]
#     t_root_J = source_data['t_root_J'][:-1]
#     transl_new_ = np.dot(transfer_mat[0].numpy(), (t_root_J + transl).T).T - t_root_J
    
#     source_data.update({
#         'transl': transl_new_
#     })
    
#     for key in ['body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose', 'betas', 'expression']:
#         source_data[key] = source_data[key][:-1]
    
#     joints_ori = None
    
#     ## test
#     if debug:
#         body_parms_test = {}
#         for key in ['global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose', 'transl', 'betas', 'expression']:
#             body_parms_test[key] = torch.tensor(source_data[key], dtype=torch.float32).to(DEVICE)
#         with torch.no_grad():
#             output = bm(**body_parms_test)
        
#         joints_ori = output.joints[:, body_joints_id+hand_joints_id, :].cpu().numpy()
#     ## test
    
    
#     return source_data, joints_ori


def preprocess_smplx(source_data, debug, DEVICE='cpu'):
    # Uniformly scale the skeleton to match a target offset
    
    gender_ = str(source_data['gender'])
    model_path = model_paths[gender_]
    num_betas = source_data['betas'].shape[1]
    bm = smplx.create(model_path, model_type='smplx', gender=gender_, ext='npz', num_betas=num_betas, use_pca=False, flat_hand_mean=True)

    body_parms = {}
    for key in ['global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose', 'transl', 'betas', 'expression']:
        body_parms[key] = torch.tensor(source_data[key], dtype=torch.float32).to(DEVICE)
    
    bm = bm.to(DEVICE)
    with torch.no_grad():
        output = bm(**body_parms)
    
    positions = output.joints[:, body_joints_id+hand_joints_id, :].cpu().numpy()
    
    origin_root = positions[:, 0]
    
    
    # Put the skeleton on the floor by subtracting the minimum height
    floor_height = positions.min(axis=0).min(axis=0)[1]
    # positions[:, :, 1] -= floor_height

    # Center the skeleton at the origin in the XZ plane
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    root_pose_init_xz[1] = floor_height
    # positions = positions - root_pose_init_xz
    

    # Ensure the initial facing direction is along Z+
    root_pos_init = positions[0]
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # Ensure that all poses initially face Z+
    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / \
        np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    # Calculate quaternion for root orientation
    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    
    origin_quat = root_quat_init.copy()
    
    root_quat_init = np.ones(positions.shape[:1] + (4,)) * root_quat_init

    # # Rotate the motion capture data using the calculated quaternion
    # positions = qrot_np(root_quat_init, positions)
    smplx_orient = source_data['global_orient'].copy()
    smplx_orient_mat = roma.rotvec_to_rotmat(torch.Tensor(smplx_orient))
    
    # apply transform on smplx parameters
    euler_transfer = qeuler_np(root_quat_init, 'yzx') / 180 * np.pi
    # since forward init vector is set in xz plane, so only rotate y axis!
    transfer_mat = roma.euler_to_rotmat('yzx', torch.Tensor(euler_transfer))
    smplx_orient_modified_mat = torch.matmul(transfer_mat, smplx_orient_mat)
    smplx_orient_modified = roma.rotmat_to_rotvec(smplx_orient_modified_mat)
    source_data.update({
        'global_orient': smplx_orient_modified.numpy(),
    })
    
    
    # root_pose_init_xz_new = (transfer_mat[0] @ torch.Tensor(root_pose_init_xz)).numpy()
    transl = source_data['transl']
    transl_new =  (transfer_mat[0] @ torch.Tensor(transl - root_pose_init_xz + source_data['t_root_J']).unsqueeze(-1)).squeeze(-1).numpy() - source_data['t_root_J']
    source_data.update({
        'transl': transl_new
    })
    
    # test
    positions_smplx = None
    if debug:
        body_parms_new = {}
        for key in ['global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose', 'transl', 'betas', 'expression']:
            body_parms_new[key] = torch.tensor(source_data[key], dtype=torch.float32).to(DEVICE)
        
        bm = bm.to(DEVICE)
        with torch.no_grad():
            output = bm(**body_parms_new)
        
        positions_smplx = output.joints[:, body_joints_id+hand_joints_id, :].cpu().numpy()
    
    return source_data, positions_smplx, (origin_quat, origin_root)

# global

# velocity

def get_smplx_feature(source_data, debug=True):
    ## step 1 : r velocity linear velocity root y
    # step 2 : local cont6d
    # step 3 : global rotation towards root cont6d
    
    ### root velocity  alongside xz
    root_velocity = (source_data['transl'][1:] - source_data['transl'][:-1]).copy()
    root_rotation_euler = roma.rotvec_to_euler('yzx', torch.Tensor(source_data['global_orient']))
    root_rotation_euler[:, 1:] = 0
    root_rotmat_y = roma.euler_to_rotmat('yzx', root_rotation_euler)
    root_rotmat_inv = roma.rotmat_inverse(root_rotmat_y)
    root_velocity = root_rotmat_inv[:-1] @ torch.Tensor(root_velocity[..., np.newaxis])
    root_velocity_xz = root_velocity[:, [0, 2]]
    
    ### root height y
    root_height = source_data['transl'][:, 1:2] + source_data['t_root_J'][:, 1:2]
    
    
    ### global root cont6d
    global_orient = roma.rotvec_to_rotmat(torch.Tensor(source_data['global_orient']))
    global_root_cont6d = torch.cat([global_orient[..., 0], global_orient[..., 1]], dim=-1)
    
    
    ### local cont6d rotation towards parents
    ### T * (21 + 15 + 15) * 6
    seq_len = len(source_data['body_pose'])
    cont6d_local = []
    for key in ['body_pose', 'left_hand_pose', 'right_hand_pose',]:
        rotmat = roma.rotvec_to_rotmat(torch.Tensor(source_data[key]).reshape(seq_len, -1, 3))
        cont6d = torch.cat([rotmat[..., 0], rotmat[..., 1]], dim=-1)
        cont6d_local.append(cont6d)
    cont6d_local = torch.cat(cont6d_local, dim=1)
    
    ### global cont6d rotation towards root
    ### T * (1 + 21 + 3 + 15 + 15) * 6
    cont6d_global = {}
    rot_vec = []
    for key in ['global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose']:
        rot_vec.append(torch.Tensor(source_data[key]).reshape(seq_len, -1, 3))
    rot_vec = torch.cat(rot_vec, dim=1)
    rot_mat = roma.rotvec_to_rotmat(rot_vec)
    global_rotmat = []
    # global_rotmat.append(rot_mat[:, 0])
    global_rotmat.append(torch.eye(3).repeat(len(rot_mat), 1, 1))
    for i in range(1, len(parents)):
        current_res = torch.matmul(global_rotmat[parents[i]], rot_mat[:, i])
        global_rotmat.append(current_res)
    global_rotmat = torch.stack(global_rotmat, dim=1)
    cont6d_global = torch.cat([global_rotmat[..., 0], global_rotmat[..., 1]], dim=-1)
    cont6d_global = cont6d_global[:, 1:]
    
    smplx_features = {
        'root_velocity': root_velocity_xz.squeeze(-1).numpy(),
        'root_height': root_height[:-1],
        'global_root_cont6d': global_root_cont6d[:-1].numpy(),
        'cont6d_local': cont6d_local[:-1].numpy(),
        'cont6d_global': cont6d_global[:-1].numpy()
    }
    
    if debug:
        ### recover from feature
        #### root pos
        motion_len = len(smplx_features['root_velocity'])
        root_rotmat = cont6d_to_matrix(torch.Tensor(smplx_features['global_root_cont6d']))
        root_rotvec = roma.rotmat_to_rotvec(root_rotmat)
        
        root_rotmat_y = roma.rotmat_to_euler('yzx', root_rotmat)
        root_rotmat_y[:, 1:] = 0
        root_rotmat_y = roma.euler_to_rotmat('yzx', root_rotmat_y)
        root_vel_xz_np = np.zeros((motion_len, 3))
        root_vel_xz_np[:, [0, 2]] = smplx_features['root_velocity']
        root_xz_vel = torch.matmul(root_rotmat_y, torch.Tensor(root_vel_xz_np).unsqueeze(-1)).squeeze(-1)

        root_pos_xz = torch.zeros((motion_len, 3))
        root_pos_xz[1:] = root_xz_vel[:-1]
        root_pos_xz = torch.cumsum(root_pos_xz, dim=0)
        root_pos_xz[:, 1:2] = torch.Tensor(smplx_features['root_height'])
        
        
        # check root pos & rot
        b = source_data['transl'] + source_data['t_root_J']
        print((root_pos_xz.numpy() - b[:-1]).max())
        print((root_pos_xz.numpy() - b[:-1]).min())
        
        # check global rot & local rot
        rot_global = cont6d_to_matrix(torch.Tensor(smplx_features['cont6d_global']))
        rot_global = torch.cat([torch.eye(3).repeat(motion_len, 1, 1, 1), rot_global], dim=1)
        
        rot_local = []
        for i in range(1, len(parents)):
            current_res = torch.matmul(roma.rotmat_inverse(rot_global[:, parents[i]]), rot_global[:, i])
            rot_local.append(current_res)
        
        rot_local = torch.stack(rot_local, dim=1)
        rot_local_vec = roma.rotmat_to_rotvec(rot_local)
        print((rot_local_vec - rot_vec[:-1, 1:]).max())
        print((rot_local_vec - rot_vec[:-1, 1:]).min())
    return smplx_features
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser('Process Inter-X dataset')
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--period', type=int, default=8)
    parser.add_argument('--part', type=int, default=0)
    args = parser.parse_args()
    
    if args.gpu_id not in [0, 1, 2, 3, 4, 5, 6, 7]:
        DEVICE = 'cpu'
    else:
        DEVICE = 'cuda:%d'%args.gpu_id 
    
    # Lower legs
    l_idx1, l_idx2 = 5, 8
    # Right/Left foot
    fid_r, fid_l = [8, 11], [7, 10]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [2, 1, 17, 16]
    # body,hand joint idx
    # 2*3*5=30, left first, then right
    hand_joints_id = [i for i in range(25, 55)]
    body_joints_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                      12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # 22 joints
    # l_hip, r_hip
    r_hip, l_hip = 2, 1
    joints_num = 52
    # ds_num = 8

    # change your motion_data joint
    data_dir = 'SOLAMI_data/Inter-X/joints'
    # change your save folder
    save_dir = 'SOLAMI_data/Inter-X/unified_data/'
    os.makedirs(save_dir, exist_ok=True)

    dataset_items_info_path = "SOLAMI_data/Inter-X/dataset_items_pre.json"
    
    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_body_hand_kinematic_chain

    # Get offsets of target skeleton
    # we random choose one
    example_data = np.load('SOLAMI_data/HumanML3D/HumanML3D_mirror/000021.npz')
    example_data = example_data['pose_np']
    example_data = example_data[:, body_joints_id + hand_joints_id, :]
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)

    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')

    # (joints_num, 3)
    # tgt_offsets is the 000021 skeleton bone lengths with the predefined offset directions. global postion offsets
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])

    #  source_list = findAllFile(data_dir)
    frame_num = 0
    with open(dataset_items_info_path, 'r') as f:
        dataset_items_info = json.load(f)
    dataset_items_info = dict(sorted(dataset_items_info.items()))
    source_list = list(dataset_items_info.keys())
    # data_names = os.listdir(data_dir)
    # source_list = [pjoin(data_dir, data_name) for data_name in data_names]
    
    if args.debug:
        source_list = source_list[:5]
    else:
        source_list = source_list[args.part::args.period]
    
    for source_file in tqdm(source_list):
        try:
            data_item_1 = dataset_items_info[source_file]
            motion_data_path_1 = pjoin(data_dir, data_item_1['motion_data_path'])
            source_data_P1_ = np.load(motion_data_path_1)
            start_frame_P1 = data_item_1['start_frame']
            end_frame_P1 = data_item_1['end_frame']
            # source_data_P1_ = np.load(pjoin(source_file, 'P1.npz'))
            source_data_P1_ = dict(source_data_P1_)
            source_data_P1 = {}
            source_data_P1['smplx_params'] = {}
            for key in source_data_P1_.keys():
                if key != 'pose_np':
                    if key != 'gender':
                        source_data_P1['smplx_params'][key] = source_data_P1_[key][start_frame_P1:end_frame_P1]
                    else:
                        source_data_P1['smplx_params'][key] = source_data_P1_[key]
                else:
                    source_data_P1['ske_joints'] = source_data_P1_[key][start_frame_P1:end_frame_P1]
            
            # source_data = np.load(source_file)[:, body_joints_id+hand_joints_id, :]
            data_1, ground_positions_1, positions_1, l_velocity_1, (ske_forward_1, ske_root_1, scale_1, root_quat_1) = process_file(
                source_data_P1, 0.002)
            rec_ric_data_1 = recover_from_ric(torch.from_numpy(
                data_1).unsqueeze(0).float(), joints_num)
            source_data_P1['ske_joints'] = rec_ric_data_1.squeeze().numpy()
            source_data_P1['ske_feature'] = data_1
            
            source_data_smplx_params_1, joints_smplx_1, (smplx_forward_1, smplx_root_1) = preprocess_smplx(copy.deepcopy(source_data_P1['smplx_params']), args.debug, DEVICE)
            source_data_P1['smplx_params'] = source_data_smplx_params_1
            smplx_feature_1 = get_smplx_feature(copy.deepcopy(source_data_P1['smplx_params']), args.debug)
            source_data_P1['smplx_feature'] = smplx_feature_1
            
            data_transforms_1 = {
                "ske_forward": ske_forward_1,
                "ske_root": ske_root_1,
                "scale": scale_1,
                "smplx_forward": smplx_forward_1,
                "smplx_root": smplx_root_1,
            }
            source_data_P1.update({
                'transforms': data_transforms_1
            })
            
            
            if data_item_1['next_partner_motion_name'] != None:
                data_item_2 = dataset_items_info[data_item_1['next_partner_motion_name']]
                motion_data_path_2 = pjoin(data_dir, data_item_2['motion_data_path'])
                source_data_P2_ = np.load(motion_data_path_2)
                # source_data_P2_ = np.load(pjoin(source_file, 'P2.npz'))
                source_data_P2_ = dict(source_data_P2_)
                source_data_P2 = {}
                source_data_P2['smplx_params'] = {}
                for key in source_data_P2_.keys():
                    if key != 'pose_np':
                        if key != 'gender':
                            source_data_P2['smplx_params'][key] = source_data_P2_[key][start_frame_P1:end_frame_P1]
                        else:
                            source_data_P2['smplx_params'][key] = source_data_P2_[key]
                    else:
                        source_data_P2['ske_joints'] = source_data_P2_[key][start_frame_P1:end_frame_P1]
                    
                data_2, ground_positions_2, positions_2, l_velocity_2, (ske_forward_2, ske_root_2, scale_2, root_quat_2) = process_file(
                    source_data_P2, 0.002)

                
                source_data_smplx_params_2, joints_smplx_2, (smplx_forward_2, smplx_root_2) = preprocess_smplx(copy.deepcopy(source_data_P2['smplx_params']), args.debug, DEVICE)
               
                
                def get_relative_pose( ske_forward_1,  ske_forward_2, scale_1, scale_2, ske_root_1, ske_root_2):
                
                    quat_P1_in_P2 = qmul_np(qinv_np(ske_forward_1), ske_forward_2)
                    pos_P1_in_P2 = qrot_np(ske_forward_2[:1], (scale_1 + scale_2) / 2. * (ske_root_1[:1] - ske_root_2[:1]))
                    
                    quat_P1_in_P2_np = np.ones(rec_ric_data_1.shape[:-1] + (4,)) * quat_P1_in_P2[0]
                    pos_P1_in_P2_np = np.ones(rec_ric_data_1.shape[:-1] + (3,)) * pos_P1_in_P2[0]
                    rec_ric_data_1_in_P2 = qrot_np(quat_P1_in_P2_np, rec_ric_data_1.numpy()) + pos_P1_in_P2_np
                    
                    cont6d_P1_in_P2_save = quaternion_to_cont6d_np(quat_P1_in_P2[0])
                    pos_P1_in_P2_save = pos_P1_in_P2[0]
                    
                    cont6d_P2_in_P1_save = quaternion_to_cont6d_np(qinv_np(quat_P1_in_P2[0]))
                    pos_P2_in_P1_save = -qrot_np(qinv_np(quat_P1_in_P2[0]), pos_P1_in_P2[0])
                    
                    return cont6d_P1_in_P2_save, pos_P1_in_P2_save, cont6d_P2_in_P1_save, pos_P2_in_P1_save, rec_ric_data_1_in_P2
                
                
                cont6d_P1_in_P2_ske, pos_P1_in_P2_ske, cont6d_P2_in_P1_ske, pos_P2_in_P1_ske, rec_ric_data_1_in_P2_ske = \
                    get_relative_pose(ske_forward_1,  ske_forward_2, scale_1, scale_2, ske_root_1, ske_root_2)
                source_data_P1['transforms'].update({
                    'ske_relative_cont6d': cont6d_P1_in_P2_ske,
                    'ske_relative_pos': pos_P1_in_P2_ske,
                })

                cont6d_P1_in_P2_smplx, pos_P1_in_P2_smplx, cont6d_P2_in_P1_smplx, pos_P2_in_P1_smplx, rec_ric_data_1_in_P2_smplx = \
                    get_relative_pose(smplx_forward_1,  smplx_forward_2, 1, 1, smplx_root_1, smplx_root_2)
                    
                source_data_P1['transforms'].update({
                    'smplx_relative_cont6d': cont6d_P1_in_P2_smplx,
                    'smplx_relative_pos': pos_P1_in_P2_smplx,
                })

            np.savez(pjoin(save_dir, data_item_1['motion_name'] + '.npz'), **source_data_P1)
            frame_num += data_1.shape[0]
        except Exception as e:
            print(source_file)
            print(e)

    print('Total clips: %d, Frames: %d, Duration: %fm' %
          (len(source_list), frame_num, frame_num / 30 / 60))