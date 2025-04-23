import json
import numpy as np
import transformations as tf
from aic_utils.file_utils import read_json_file, read_bmap_file

def calculate_scale_factor(source_bone_data, target_bone_data, source_root_matrix, target_root_matrix):
    """
    Calculate the scaling factor between source and target character skeletons.
    
    This function computes the ratio between the heights of the target and source
    character skeletons to determine the proper scaling for retargeting.
    
    Args:
        source_bone_data (dict): Bone data for the source character
        target_bone_data (dict): Bone data for the target character
        source_root_matrix (numpy.ndarray): Transformation matrix of the source root bone
        target_root_matrix (numpy.ndarray): Transformation matrix of the target root bone
        
    Returns:
        float: The scaling factor to apply to the source animation
    """
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
    """
    Calculate the rotation difference between two transformation matrices.
    
    This function computes the quaternion that represents the difference in
    orientation between two matrices, which can be used to align
    orientations during animation retargeting.
    
    Args:
        matrix1 (numpy.ndarray): First 4x4 transformation matrix
        matrix2 (numpy.ndarray): Second 4x4 transformation matrix
        
    Returns:
        numpy.ndarray: 4x4 matrix representing the rotation difference
    """
    rotation1 = tf.quaternion_from_matrix(matrix1)
    rotation2 = tf.quaternion_from_matrix(matrix2)
    rotation_diff = tf.quaternion_multiply(rotation2, tf.quaternion_inverse(rotation1))
    return tf.quaternion_matrix(rotation_diff)

def apply_rotation_to_children(remapped_rotmat, parent_rotation, source_bones, joint_names, parent_name, t):
    """
    Apply a parent rotation to all child bones in a hierarchy.
    
    This function propagates rotation changes from a parent bone to all its 
    children in the skeletal hierarchy for a specific frame of animation.
    
    Args:
        remapped_rotmat (numpy.ndarray): The current rotation matrices being modified
        parent_rotation (numpy.ndarray): The parent rotation to apply
        source_bones (dict): Bone data containing hierarchy information
        joint_names (numpy.ndarray): Array of joint names
        parent_name (str): Name of the parent bone
        t (int): Current frame index
        
    Returns:
        numpy.ndarray: Updated rotation matrices
    """
    children = [bone for bone, data in source_bones.items() if data['parent'] == parent_name]
    for child in children:
        child_index = np.where(joint_names == child)[0][0]
        remapped_rotmat[t, child_index] = parent_rotation @ remapped_rotmat[t, child_index]
    return remapped_rotmat

def retarget_motion(motion_rotmat, motion_loc, joint_names, retargetmap_path, source_bone_data, target_bone_data):
    """
    Retarget a motion sequence from a source character to a target character.
    
    This function adapts animation data from one character's skeleton to another by
    computing appropriate transformations for each bone based on the differences
    in skeletal structure.
    
    Args:
        motion_rotmat (numpy.ndarray): Rotation matrices for all joints and frames (T, J, 3, 3)
        motion_loc (numpy.ndarray): Root locations for all frames (T, 3)
        joint_names (numpy.ndarray): Array of joint names
        retargetmap_path (str): Path to the bone mapping file
        source_bone_data (dict): Bone data for the source character
        target_bone_data (dict): Bone data for the target character
        
    Returns:
        tuple: (remapped_rotmat, remapped_loc, remapped_joint_names)
               - remapped_rotmat: Retargeted rotation matrices
               - remapped_loc: Scaled and adjusted locations
               - remapped_joint_names: Mapped joint names
    """
    bone_mapping = read_bmap_file(retargetmap_path)
    
    source_bones = source_bone_data['bones']
    target_bones = target_bone_data['bones']
    
    # Find root bones for both skeletons
    source_root = next(bone for bone, data in source_bones.items() if data['parent'] is None)
    target_root = next(bone for bone, data in target_bones.items() if data['parent'] is None)
    
    source_root_matrix = np.array(source_bones[source_root]['matrix_local'])
    target_root_matrix = np.array(target_bones[target_root]['matrix_local'])
    
    scale_factor = calculate_scale_factor(source_bone_data, target_bone_data, source_root_matrix, target_root_matrix)

    T, J, _, _ = motion_rotmat.shape
    remapped_rotmat = np.zeros_like(motion_rotmat)
    remapped_loc = np.zeros_like(motion_loc)
    remapped_joint_names = np.empty(J, dtype=object)
    
    # Map joint names from source to target
    for j in range(J):
        joint_name = joint_names[j]
        if(joint_name in bone_mapping):
            target_bone_name = bone_mapping[joint_name]
        else:
            target_bone_name = "NP"  # "Not Present" marker
        remapped_joint_names[j] = target_bone_name
    
    # Process each frame and joint
    for t in range(T):
        for j in range(J):
            source_joint_name = joint_names[j]
            target_joint_name = remapped_joint_names[j]
            if target_joint_name == "NP":
                continue
            else:
                # Get rest poses and compute adjustment matrices
                source_matrix_local = source_bones[source_joint_name]['matrix_local']
                target_matrix_local = target_bones[target_joint_name]['matrix_local']
                rest_diff = np.linalg.inv(source_matrix_local) @ target_matrix_local
                
                if target_joint_name == target_root:
                    # Root bone is handled specially
                    remapped_rotmat[t, j] = motion_rotmat[t, j] @ rest_diff
                else:
                    # Non-root bones need to account for parent differences
                    source_parent_name = source_bones[source_joint_name]['parent']
                    target_parent_name = target_bones[target_joint_name]['parent']
                    # print(f"source joint name: {source_joint_name}")
                    # print(f"source parent name: {source_parent_name}")
                    source_parent_matrix_local = source_bones[source_parent_name]['matrix_local']
                    target_parent_matrix_local = target_bones[target_parent_name]['matrix_local']
                    parent_diff = np.linalg.inv(target_parent_matrix_local) @ source_parent_matrix_local
                    remapped_rotmat[t, j] = parent_diff @ motion_rotmat[t, j] @ rest_diff
    
    if source_root not in joint_names:
        raise ValueError(f"source_root '{source_root}' not found in joint_names")
    
    root_index = np.where(joint_names == source_root)[0][0]
    # print("root_index:", root_index)
    
    # Scale and adjust root positions
    for t in range(T):
        remapped_loc[t] = motion_loc[t] * scale_factor * 1.03  # Small additional adjustment factor
        
    return remapped_rotmat, remapped_loc, remapped_joint_names
