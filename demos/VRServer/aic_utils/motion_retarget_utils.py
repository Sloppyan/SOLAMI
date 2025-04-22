import json
import numpy as np
import transformations as tf
from aic_utils.file_utils import read_json_file, read_bmap_file

def calculate_scale_factor(source_bone_data, target_bone_data, source_root_matrix, target_root_matrix):

    source_armature_matrix = np.array(source_bone_data['armature']['armature_matrix'])
    target_armature_matrix = np.array(target_bone_data['armature']['armature_matrix'])

    source_global_matrix = source_armature_matrix @ source_root_matrix
    target_global_matrix = target_armature_matrix @ target_root_matrix

    source_height = source_global_matrix[:3, 3][2]
    target_height = target_global_matrix[:3, 3][2]
    # print(f"soutce_height: {source_height}")
    # print(f"target_height: {target_height}")

    scale_factor = target_height / source_height
    return scale_factor

def calculate_rotation_difference(matrix1, matrix2):
    rotation1 = tf.quaternion_from_matrix(matrix1)
    rotation2 = tf.quaternion_from_matrix(matrix2)
    rotation_diff = tf.quaternion_multiply( rotation2, tf.quaternion_inverse(rotation1))
    return tf.quaternion_matrix(rotation_diff)

def apply_rotation_to_children(remapped_rotmat, parent_rotation, source_bones, joint_names, parent_name, t):
    children = [bone for bone, data in source_bones.items() if data['parent'] == parent_name]
    for child in children:
        child_index = np.where(joint_names == child)[0][0]
        remapped_rotmat[t, child_index] = parent_rotation @ remapped_rotmat[t, child_index]
    return remapped_rotmat

def retarget_motion(motion_rotmat, motion_loc, joint_names, retargetmap_path, source_bone_data, target_bone_data):
    bone_mapping = read_bmap_file(retargetmap_path)
    
    source_bones = source_bone_data['bones']
    target_bones = target_bone_data['bones']
    
    source_root = next(bone for bone, data in source_bones.items() if data['parent'] is None)
    target_root = next(bone for bone, data in target_bones.items() if data['parent'] is None)
    
    source_root_matrix = np.array(source_bones[source_root]['matrix_local'])
    target_root_matrix = np.array(target_bones[target_root]['matrix_local'])
    
    # root_rotation_difference = calculate_rotation_difference(source_root_matrix, target_root_matrix)
    # root_rotation_difference_rev = calculate_rotation_difference(target_root_matrix, source_root_matrix)
    scale_factor = calculate_scale_factor(source_bone_data, target_bone_data, source_root_matrix, target_root_matrix)

    T, J, _, _ = motion_rotmat.shape
    remapped_rotmat = np.zeros_like(motion_rotmat)
    remapped_loc = np.zeros_like(motion_loc)
    remapped_joint_names = np.empty(J, dtype=object)
    for j in range(J):
        joint_name = joint_names[j]
        if(joint_name in bone_mapping):
            target_bone_name = bone_mapping[joint_name]
        else:
            target_bone_name = "NP"
        remapped_joint_names[j] = target_bone_name
    for t in range(T):
        for j in range(J):
            source_joint_name = joint_names[j]
            target_joint_name = remapped_joint_names[j]
            if target_joint_name == "NP":
                continue
            else:
                source_matrix_local = source_bones[source_joint_name]['matrix_local']
                target_matrix_local = target_bones[target_joint_name]['matrix_local']
                rest_diff = np.linalg.inv(source_matrix_local) @ target_matrix_local
                if target_joint_name == target_root:
                    remapped_rotmat[t, j] = motion_rotmat[t, j] @ rest_diff
                else:
                    source_parent_name = source_bones[source_joint_name]['parent']
                    target_parent_name = target_bones[target_joint_name]['parent']
                    # print(f"source joint name: {source_joint_name}")
                    # print(f"source parent name: {source_parent_name}")
                    source_parent_matrix_local = source_bones[source_parent_name]['matrix_local']
                    target_parent_matrix_local = target_bones[target_parent_name]['matrix_local']
                    parent_diff = np.linalg.inv(target_parent_matrix_local) @ source_parent_matrix_local
                    remapped_rotmat[t, j] = parent_diff @ motion_rotmat[t, j] @ rest_diff
                    # source_parent_matrix = motion_rotmat[t, parent_index]
                    # target_parent_matrix = remapped_rotmat[t, parent_index]
                    # print(f"source_parent_matrix: {source_parent_matrix}")
                    # print(f"target_parent_matrix: {target_parent_matrix}")
                    # target_parent_matrix_inv = np.linalg.inv(target_parent_matrix)
                    # source_parent_matrix = motion_rotmat[t, parent_index]
                    # remapped_rotmat[t, j] = target_parent_matrix_inv @ source_parent_matrix @ motion_rotmat[t, j] @ rest_diff
    if source_root not in joint_names:
        raise ValueError(f"source_root '{source_root}' not found in joint_names")
    root_index = np.where(joint_names == source_root)[0][0]
    # print("root_index:", root_index)
    for t in range(T):
        remapped_loc[t] = motion_loc[t] *scale_factor *1.03
        # print(f"root rotation difference: {root_rotation_difference}")
        # remapped_rotmat[t, root_index] =  remapped_rotmat[t, root_index] @ root_rotation_difference
        # remapped_rotmat = apply_rotation_to_children(remapped_rotmat, root_rotation_difference_rev, source_bones, joint_names, source_root, t)
        
    return remapped_rotmat, remapped_loc, remapped_joint_names
