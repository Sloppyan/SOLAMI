import os
import math
import numpy as np
import transformations as tf
from aic_utils.file_utils import read_json_file
from aic_utils.motion_retarget_utils import retarget_motion

joint_names = [
    "pelvis", "right_hip", "left_hip", "spine1", "right_knee", "left_knee", "spine2", "right_ankle", "left_ankle", 
    "spine3", "right_foot", "left_foot", "neck", "right_collar", "left_collar", "head", "right_shoulder", 
    "left_shoulder", "right_elbow", "left_elbow", "right_wrist", "left_wrist", "right_eye_smplhf", "left_eye_smplhf", 
    "jaw", "right_index1", "right_index2", "right_index3", "right_middle1", "right_middle2", "right_middle3", 
    "right_pinky1", "right_pinky2", "right_pinky3", "right_ring1", "right_ring2", "right_ring3", "right_thumb1", 
    "right_thumb2", "right_thumb3", "left_index1", "left_index2", "left_index3", "left_middle1", "left_middle2", 
    "left_middle3", "left_pinky1", "left_pinky2", "left_pinky3", "left_ring1", "left_ring2", "left_ring3", 
    "left_thumb1", "left_thumb2", "left_thumb3"
]

ignore_list = [0, 1, 2, 13, 14, 22, 23, 24, 25, 31, 34, 37, 40, 46, 49, 52]

# Convert the coordinate system from standard (Y-up) to Blender (Z-up) and rotate 180 degrees around Z-axis
def convert_coordinate_system(coord):
    blender_coord = np.array([coord[0], -coord[2], coord[1], 1])  # Convert to homogeneous coordinate (4x1)
    rotation_matrix = tf.rotation_matrix(math.pi, (0, 0, 1))  # Rotate 180 degrees around Z-axis
    rotated_coord = np.dot(rotation_matrix, blender_coord)
    return rotated_coord[:3]  # Convert back to 3x1 coordinate

def calculate_quaternion(start, end):
    direction = end - start
    direction = direction / np.linalg.norm(direction)
    up = np.array([0, 0, 1])
    quaternion = tf.quaternion_about_axis(np.arccos(np.dot(up, direction)), np.cross(up, direction))
    return quaternion

def apply_additional_rotation(matrix):
    additional_rot = tf.euler_matrix(-np.pi / 2, 0, 0, 'sxyz')
    return np.dot(additional_rot, matrix)

def convert_blender_to_unity(matrix, root_loc):
    quat = tf.quaternion_from_matrix(matrix)
    unity_pos = (root_loc[0], root_loc[2], -root_loc[1])
    unity_rot = (quat[1], -quat[2], -quat[3], quat[0])
    return unity_pos, unity_rot

def convert_to_parent_space(global_quat, parent_matrix_local):
    parent_quat_local = tf.quaternion_from_matrix(parent_matrix_local)
    parent_quat_local_inv = tf.quaternion_inverse(parent_quat_local)
    parent_space_quat = tf.quaternion_multiply(parent_quat_local_inv, global_quat)
    return parent_space_quat

def load_npy_anim(file_path, char_id):
    animation_file_name = os.path.basename(file_path)
    source_char_id = animation_file_name.split('_')[0]
    source_bone_path = f"datasets/bone_data/{source_char_id}.json"
    source_bone_data = read_json_file(source_bone_path)
    
    
    motion_data = np.load(file_path)
    if motion_data.shape[1] == 52:
        eyes = np.repeat(motion_data[:, 15:16], 3, axis=1)
    motion_data = np.concatenate([motion_data[:, :22], eyes, motion_data[:, 22:]], axis=1)
    motion_loc = motion_data[:,0, :]
    T,J,_ = motion_data.shape
    parent_rotmats = np.zeros((T, J, 4, 4))
    
    
    for t in range(T):          
        # Initialize bone positions list
        bone_position_list = []
        for j in range(J):
            bone_name = joint_names[j]
            try:
                bone_position = convert_coordinate_system(motion_data[t][j])
                bone_position_list.append(bone_position)
            except IndexError:
                # print(f"Index error with bone {bone_name} at index {i}")
                continue
            
        root_bone_name = "pelvis"
        for j in range(J):
            if j in ignore_list:
                continue
            bone_name = joint_names[j]
            try:
                bone_position = bone_position_list[j]
                target_bone_name = source_bone_data["bones"][bone_name]["parent"]
                if target_bone_name:
                    target_index = joint_names.index(target_bone_name)
                    target_position = bone_position_list[target_index]
                
                    quaternion = calculate_quaternion(target_position, bone_position)
                    # print(f"{target_bone_name} rotation: {quaternion}")
                    
                    # if target_bone_name == root_bone_name:
                    #     quaternion = apply_additional_rotation(quaternion)
                    #     parent_rotmats[t, target_index] = tf.quaternion_matrix(quaternion)
                    # else:
                    #     parent_bone_name = source_bone_data["bones"][target_bone_name]["parent"]
                    #     parent_matrix_local = source_bone_data["bones"][parent_bone_name]['matrix_local']
                    #     parent_space_quat = convert_to_parent_space(quaternion, parent_matrix_local)
                    #     parent_rotmats[t, target_index] = tf.quaternion_matrix(parent_space_quat)
                    parent_rotmats[t, target_index] = tf.quaternion_matrix(quaternion)
            except IndexError:
                print(f"Index error with bone {bone_name} at index {i}")
                continue
    
    # parent_rotmats = convert_to_parent_space(motion_rotmats, source_bone_data['bones'], joint_names)
    joint_names_np = np.array(joint_names)
    
    if char_id != source_char_id:
        retargetmap_path = f"datasets/retargetmap/{source_char_id}2{char_id}.bmap"
        target_bone_path = f"datasets/bone_data/{char_id}.json"
        target_bone_data = read_json_file(target_bone_path)
        re_rotmats, re_motion_loc, re_joint_names = retarget_motion(
            parent_rotmats, 
            motion_loc, 
            joint_names_np,
            retargetmap_path, 
            source_bone_data, 
            target_bone_data
            )
        root_name = next(bone for bone, data in target_bone_data["bones"].items() if data['parent'] is None)
    else:
        re_rotmats, re_motion_loc, re_joint_names = parent_rotmats, motion_loc, joint_names_np
        
    frames = []
    T, J, _, _ = re_rotmats.shape

    for i in range(T):
        frame_data = {"frame": i + 1}
        for j in range(J):
            joint_name = re_joint_names[j]
            if joint_name != "NP":
                mat = re_rotmats[i, j]
                
                if joint_name == root_name:
                    mat = apply_additional_rotation(mat)
                    unity_pos, unity_rot = convert_blender_to_unity(mat, re_motion_loc[i])
                    frame_data[joint_name + "_position"] = [round(num, 7) for num in unity_pos]
                    frame_data[joint_name + "_rotation"] = [round(num, 7) for num in unity_rot]
                else:
                    unity_pos, unity_rot = convert_blender_to_unity(mat, re_motion_loc[i])
                    frame_data[joint_name + "_rotation"] = [round(num, 7) for num in unity_rot]

        frames.append(frame_data)

    return frames