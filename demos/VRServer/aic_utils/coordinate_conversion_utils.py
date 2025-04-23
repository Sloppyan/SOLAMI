import numpy as np
import transformations as tf

def apply_additional_rotation_inverse(matrix):
    """
    Apply an inverse additional rotation around X-axis by -90 degrees to a transformation matrix.
    
    This is often needed to convert between different coordinate systems
    where the up-axis differs (e.g., Y-up to Z-up or vice versa).
    
    Args:
        matrix: 4x4 transformation matrix to modify
        
    Returns:
        numpy.ndarray: The matrix with the additional rotation applied
    """
    additional_rot = tf.euler_matrix(-np.pi / 2, 0, 0, 'sxyz')
    return np.dot(additional_rot, matrix)

def convert_unity_to_blender(quaternion):
    """
    Convert a quaternion from Unity coordinate system to Blender coordinate system.
    
    Unity uses a left-handed coordinate system, while Blender uses right-handed.
    This function handles the conversion between these different conventions.
    
    Args:
        quaternion: Quaternion in Unity format [x, y, z, w]
        
    Returns:
        numpy.ndarray: 4x4 transformation matrix in Blender coordinate system
    """
    unity_quat = np.array([quaternion[3], quaternion[0], -quaternion[1], -quaternion[2]])
    blender_matrix = tf.quaternion_matrix(unity_quat)
    return blender_matrix

def convert_to_rest_pose(rotmats, bone_data, joint_names):
    """
    Convert animation rotation matrices to rest pose space.
    
    This function transforms rotation matrices from the animation space to the rest pose space,
    accounting for the skeletal hierarchy and local transformations of each bone.
    
    Args:
        rotmats (numpy.ndarray): Rotation matrices with shape (T, J, 3, 3) where T is the number
                                 of frames, J is the number of joints
        bone_data (dict): Dictionary containing bone information including local matrices and parent relationships
        joint_names (list): List of joint names corresponding to the joints in rotmats
        
    Returns:
        numpy.ndarray: Rotation matrices in rest pose space with shape (T, J, 3, 3)
    """
    T, J, _, _ = rotmats.shape
    rest_rotmats = np.zeros((T, J, 3, 3))
    
    for t in range(T):
        for j, joint_name in enumerate(joint_names):
            current_mat_3x3 = rotmats[t, j]
            try:
                # re_joint_name = bone_mapping[joint_name]
                self_local_mat = np.array(bone_data[joint_name]['matrix_local'])[:3, :3]
                parent_name = bone_data[joint_name]['parent']
                if parent_name:
                    parent_local_mat = np.array(bone_data[parent_name]['matrix_local'])[:3, :3]
                    parent_rest_mat = np.linalg.inv(parent_local_mat) @ self_local_mat @ current_mat_3x3
                    rest_rotmats[t, j] = parent_rest_mat
                else:
                    rest_rotmats[t, j] = self_local_mat @ current_mat_3x3
            except:
                print(f"not found joint")

    return rest_rotmats