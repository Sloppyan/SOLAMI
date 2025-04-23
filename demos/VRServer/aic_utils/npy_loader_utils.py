import os
import math
import numpy as np
import transformations as tf
from aic_utils.file_utils import read_json_file
from aic_utils.motion_retarget_utils import retarget_motion

# List of all joint names in the SMPL skeleton hierarchy
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

# List of joint indices to ignore during processing
ignore_list = [0, 1, 2, 13, 14, 22, 23, 24, 25, 31, 34, 37, 40, 46, 49, 52]

def convert_coordinate_system(coord):
    """
    Convert a coordinate from standard (Y-up) coordinate system to Blender (Z-up) 
    and rotate 180 degrees around Z-axis.
    
    Args:
        coord (numpy.ndarray): 3D coordinate in standard coordinate system
        
    Returns:
        numpy.ndarray: Transformed 3D coordinate in Blender coordinate system
    """
    blender_coord = np.array([coord[0], -coord[2], coord[1], 1])  # Convert to homogeneous coordinate (4x1)
    rotation_matrix = tf.rotation_matrix(math.pi, (0, 0, 1))  # Rotate 180 degrees around Z-axis
    rotated_coord = np.dot(rotation_matrix, blender_coord)
    return rotated_coord[:3]  # Convert back to 3x1 coordinate

def calculate_quaternion(start, end):
    """
    Calculate a quaternion that represents the rotation from the up vector to 
    the direction vector between start and end points.
    
    This is used to determine bone orientations from joint positions.
    
    Args:
        start (numpy.ndarray): 3D coordinate of the start point (parent joint)
        end (numpy.ndarray): 3D coordinate of the end point (child joint)
        
    Returns:
        numpy.ndarray: Quaternion [x, y, z, w] representing the rotation
    """
    direction = end - start
    direction = direction / np.linalg.norm(direction)
    up = np.array([0, 0, 1])
    quaternion = tf.quaternion_about_axis(np.arccos(np.dot(up, direction)), np.cross(up, direction))
    return quaternion

def apply_additional_rotation(matrix):
    """
    Apply an additional rotation around X-axis by -90 degrees to a transformation matrix.
    
    This is often needed to convert between different coordinate systems where
    the up-axis differs (e.g., Y-up to Z-up).
    
    Args:
        matrix (numpy.ndarray): 4x4 transformation matrix to modify
        
    Returns:
        numpy.ndarray: The modified transformation matrix
    """
    additional_rot = tf.euler_matrix(-np.pi / 2, 0, 0, 'sxyz')
    return np.dot(additional_rot, matrix)

def convert_blender_to_unity(matrix, root_loc):
    """
    Convert a transformation matrix and position from Blender coordinate system to Unity.
    
    Blender uses right-handed Z-up, Unity uses left-handed Y-up. This function
    handles the conversion between these coordinate systems.
    
    Args:
        matrix (numpy.ndarray): 4x4 transformation matrix in Blender space
        root_loc (numpy.ndarray): 3D position in Blender space
        
    Returns:
        tuple: (unity_pos, unity_rot) - Position and rotation (quaternion) in Unity space
    """
    quat = tf.quaternion_from_matrix(matrix)
    unity_pos = (root_loc[0], root_loc[2], -root_loc[1])
    unity_rot = (quat[1], -quat[2], -quat[3], quat[0])
    return unity_pos, unity_rot

def convert_to_parent_space(global_quat, parent_matrix_local):
    """
    Convert a global quaternion to a local quaternion in parent space.
    
    Args:
        global_quat (numpy.ndarray): Quaternion in global space
        parent_matrix_local (numpy.ndarray): Local transformation matrix of the parent
        
    Returns:
        numpy.ndarray: Quaternion in parent space
    """
    parent_quat_local = tf.quaternion_from_matrix(parent_matrix_local)
    parent_quat_local_inv = tf.quaternion_inverse(parent_quat_local)
    parent_space_quat = tf.quaternion_multiply(parent_quat_local_inv, global_quat)
    return parent_space_quat

def load_npy_anim(file_path, char_id):
    """
    Load animation data from an NPY file and optionally retarget it to a different character.
    
    This function reads joint positions from an NPY file, calculates joint rotations,
    and if needed, retargets the animation to a different character skeleton.
    
    Args:
        file_path (str): Path to the NPY animation file
        char_id (str): ID of the target character
        
    Returns:
        list: A list of animation frames, where each frame is a dictionary of bone
              transformations (positions and rotations)
    """
    animation_file_name = os.path.basename(file_path)
    source_char_id = animation_file_name.split('_')[0]
    source_bone_path = f"datasets/bone_data/{source_char_id}.json"
    source_bone_data = read_json_file(source_bone_path)
    
    # Load the motion data - expected to be joint positions
    motion_data = np.load(file_path)
    if motion_data.shape[1] == 52:
        # Handle case with fewer joints - duplicate head position for eyes
        eyes = np.repeat(motion_data[:, 15:16], 3, axis=1)
    motion_data = np.concatenate([motion_data[:, :22], eyes, motion_data[:, 22:]], axis=1)
    motion_loc = motion_data[:, 0, :]  # Root joint position
    T, J, _ = motion_data.shape
    parent_rotmats = np.zeros((T, J, 4, 4))
    
    # Process each frame of animation
    for t in range(T):          
        # Convert joint positions to Blender coordinate system
        bone_position_list = []
        for j in range(J):
            bone_name = joint_names[j]
            try:
                bone_position = convert_coordinate_system(motion_data[t][j])
                bone_position_list.append(bone_position)
            except IndexError:
                # print(f"Index error with bone {bone_name} at index {i}")
                continue
            
        # Calculate rotations from joint positions
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
                
                    # Calculate quaternion rotation from parent to child joint
                    quaternion = calculate_quaternion(target_position, bone_position)
                    parent_rotmats[t, target_index] = tf.quaternion_matrix(quaternion)
            except IndexError:
                continue
    
    # Convert joint names to numpy array for retargeting
    joint_names_np = np.array(joint_names)
    
    # Retarget animation if needed
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
        
    # Convert to frame dictionaries in Unity format
    frames = []
    T, J, _, _ = re_rotmats.shape

    for i in range(T):
        frame_data = {"frame": i + 1}
        for j in range(J):
            joint_name = re_joint_names[j]
            if joint_name != "NP":
                mat = re_rotmats[i, j]
                
                if joint_name == root_name:
                    # Special handling for the root joint
                    mat = apply_additional_rotation(mat)
                    unity_pos, unity_rot = convert_blender_to_unity(mat, re_motion_loc[i])
                    frame_data[joint_name + "_position"] = [round(num, 7) for num in unity_pos]
                    frame_data[joint_name + "_rotation"] = [round(num, 7) for num in unity_rot]
                else:
                    # Non-root joints only need rotation
                    unity_pos, unity_rot = convert_blender_to_unity(mat, re_motion_loc[i])
                    frame_data[joint_name + "_rotation"] = [round(num, 7) for num in unity_rot]

        frames.append(frame_data)

    return frames