import numpy as np
import transformations as tf

def apply_additional_rotation_inverse(matrix):
    additional_rot = tf.euler_matrix(-np.pi / 2,0,0, 'sxyz')
    return np.dot(additional_rot, matrix)

def convert_unity_to_blender( quaternion):
    unity_quat = np.array([quaternion[3], quaternion[0], -quaternion[1], -quaternion[2]])
    blender_matrix = tf.quaternion_matrix(unity_quat)
    return blender_matrix

def convert_to_rest_pose(rotmats, bone_data, joint_names):
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