import os
import numpy as np
import transformations as tf
from aic_utils.file_utils import read_json_file
from aic_utils.motion_retarget_utils import retarget_motion

# Standard joint names in the SMPLX skeleton model
SMPLX_JOINT_NAMES = [
    'pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle','spine3',
    'left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder','left_elbow',
    'right_elbow','left_wrist','right_wrist','jaw','left_eye_smplhf','right_eye_smplhf','left_index1','left_index2',
    'left_index3','left_middle1','left_middle2','left_middle3','left_pinky1','left_pinky2','left_pinky3','left_ring1',
    'left_ring2','left_ring3','left_thumb1','left_thumb2','left_thumb3','right_index1','right_index2','right_index3',
    'right_middle1','right_middle2','right_middle3','right_pinky1','right_pinky2','right_pinky3','right_ring1',
    'right_ring2','right_ring3','right_thumb1','right_thumb2','right_thumb3'
]

def apply_additional_rotation(matrix):
    """
    Apply an additional rotation to a transformation matrix.
    
    This function is used to adjust the orientation of bones when needed,
    though in this implementation it applies an identity rotation.
    
    Args:
        matrix (numpy.ndarray): 4x4 transformation matrix to modify
        
    Returns:
        numpy.ndarray: The modified transformation matrix
    """
    additional_rot = tf.euler_matrix(0, 0, 0, 'sxyz')  # Currently identity rotation
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
    unity_pos = (-root_loc[0], root_loc[1], root_loc[2])
    # todo amass animation convention
    # unity_pos = (unity_pos_original[])
    unity_rot = (quat[1], -quat[2], -quat[3], quat[0])
    return unity_pos, unity_rot

def convert_to_parent_space(motion_rotmat, bone_data, joint_names):
    """
    Convert rotation matrices to full 4x4 transformation matrices.
    
    This function takes 3x3 rotation matrices and converts them to 4x4 transformation
    matrices that can be used for further processing.
    
    Args:
        motion_rotmat (numpy.ndarray): Rotation matrices with shape (T, J, 3, 3)
        bone_data (dict): Dictionary containing bone information
        joint_names (list): List of joint names
        
    Returns:
        numpy.ndarray: 4x4 transformation matrices with shape (T, J, 4, 4)
    """
    T, J, _, _ = motion_rotmat.shape
    parent_rotmats = np.zeros((T, J, 4, 4))
    
    for t in range(T):
        for j, joint_name in enumerate(joint_names):
            current_mat_3x3 = motion_rotmat[t, j]
            current_mat = np.eye(4)
            current_mat[:3, :3] = current_mat_3x3
            parent_rotmats[t, j] = current_mat
            
    return parent_rotmats

def rodrigues_to_rotmat(poses, rodrigues_reference = None):
    """
    Convert axis-angle rotations (Rodrigues vectors) to rotation matrices.
    
    This function converts poses represented as axis-angle vectors (also known as
    Rodrigues vectors) to 3x3 rotation matrices for all joints across all frames.
    
    Args:
        poses (numpy.ndarray): Axis-angle vectors with shape (T, N*3)
        rodrigues_reference (numpy.ndarray, optional): Reference poses to add to the input poses
        
    Returns:
        numpy.ndarray: Rotation matrices with shape (T, N, 3, 3)
    """
    T = poses.shape[0]
    N = poses.shape[1] // 3
    poses = poses.reshape(T, N, 3)
    rotmats = np.zeros((T, N, 3, 3))
    
    for t in range(T):
        for j in range(N):
            rod = poses[t, j] 
            if rodrigues_reference is not None:
                rod += rodrigues_reference[j]
            theta = np.linalg.norm(rod)
            if theta < 1e-8:
                R = np.identity(3)  # For very small rotations, use identity matrix
            else:
                axis = rod / theta
                R_full = tf.rotation_matrix(theta, axis)
                R = R_full[:3, :3]  # Extract the 3x3 rotation part
            rotmats[t, j] = R
    return rotmats

def load_npz_anim(file_path, char_id):
    """
    Load animation data from an NPZ file and optionally retarget it to a different character.
    
    This function reads pose data (as Rodrigues vectors) and translations from an NPZ file,
    converts them to rotation matrices, and if needed, retargets the animation to a 
    different character skeleton.
    
    Args:
        file_path (str): Path to the NPZ animation file
        char_id (str): ID of the target character
        
    Returns:
        list: A list of animation frames, where each frame is a dictionary of bone
              transformations (positions and rotations)
    """
    animation_file_name = os.path.basename(file_path)
    source_char_id = animation_file_name.split('_')[0]
    source_bone_path = f"datasets/bone_data/{source_char_id}.json"
    source_bone_data = read_json_file(source_bone_path)
    
    # Load animation data from NPZ file
    data = np.load(file_path, allow_pickle=True)
    motion_loc = data['trans']  # Root translations
    motion_rotmat = rodrigues_to_rotmat(data['poses'])  # Convert poses to rotation matrices
    joint_names = np.array(SMPLX_JOINT_NAMES)
    root_name = 'pelvis'
        
    # Convert rotation matrices to transformation matrices
    parent_rotmats = convert_to_parent_space(motion_rotmat, source_bone_data['bones'], joint_names)
    # print(f"char id: {char_id}, source char id: {source_char_id}")
    
    # Retarget animation if needed
    if char_id != source_char_id:
        retargetmap_path = f"datasets/retargetmap/{source_char_id}2{char_id}.bmap"
        target_bone_path = f"datasets/bone_data/{char_id}.json"
        target_bone_data = read_json_file(target_bone_path)
        re_rotmats, re_motion_loc, re_joint_names = retarget_motion(
            parent_rotmats, 
            motion_loc, 
            joint_names,
            retargetmap_path, 
            source_bone_data, 
            target_bone_data
            )
        root_name = next(bone for bone, data in target_bone_data["bones"].items() if data['parent'] is None)
    else:
        re_rotmats, re_motion_loc, re_joint_names = parent_rotmats, motion_loc, joint_names
    
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