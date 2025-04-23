import numpy as np
import transformations as tf
from aic_utils.file_utils import read_json_file, read_bmap_file


def calculate_scale_factor(source_bone_data, target_bone_data, source_root, target_root):
    """
    Calculate the scaling factor between source and target character skeletons.
    
    This function computes the ratio between the heights of the target and source
    character skeletons to determine the proper scaling for retargeting.
    
    Args:
        source_bone_data (dict): Bone data for the source character
        target_bone_data (dict): Bone data for the target character
        source_root (str): Name of the root bone in the source skeleton
        target_root (str): Name of the root bone in the target skeleton
        
    Returns:
        float: The scaling factor to apply to the source animation
    """
    source_armature_matrix = np.array(source_bone_data['armature']['armature_matrix'])
    target_armature_matrix = np.array(target_bone_data['armature']['armature_matrix'])
    source_root_matrix = np.array(source_bone_data['bones'][source_root]['matrix_local'])
    target_root_matrix = np.array(target_bone_data['bones'][target_root]['matrix_local'])

    source_global_matrix = source_armature_matrix @ source_root_matrix
    target_global_matrix = target_armature_matrix @ target_root_matrix

    source_height = source_global_matrix[:3, 3][2]
    target_height = target_global_matrix[:3, 3][2]

    scale_factor = target_height / source_height
    return scale_factor


def calculate_rotation_difference(source_root_matrix, target_root_matrix):
    """
    Calculate the rotation difference between source and target root bones.
    
    This function computes the quaternion that represents the difference in
    orientation between the two root bones, which can be used to align
    the source animation to the target skeleton.
    
    Args:
        source_root_matrix (numpy.ndarray): 4x4 transformation matrix of the source root bone
        target_root_matrix (numpy.ndarray): 4x4 transformation matrix of the target root bone
        
    Returns:
        tuple: The quaternion representing the rotation difference, adapted for Unity
    """
    source_quat = tf.quaternion_from_matrix(source_root_matrix)
    target_quat = tf.quaternion_from_matrix(target_root_matrix)
    rotation_diff = tf.quaternion_multiply(target_quat, tf.quaternion_inverse(source_quat))
    
    # Convert the rotation difference matrix from Blender to Unity coordinate system
    corrected_rotation_diff = (rotation_diff[1], -rotation_diff[2], -rotation_diff[3], rotation_diff[0])
    
    return rotation_diff

def load_json_anim(file_path, char_id=None):
    """
    Load animation data from a JSON file and optionally retarget it to a different character.
    
    This function reads animation data from a JSON file and, if specified, retargets the
    animation to a different character by adjusting bone rotations and positions based on
    the differences between the skeletons.
    
    Args:
        file_path (str): Path to the JSON animation file
        char_id (str, optional): ID of the target character. If None or same as source,
                                 no retargeting is performed.
        
    Returns:
        list: A list of animation frames, where each frame is a dictionary of bone
              transformations (positions and rotations)
    """
    source_char_id = file_path.split('/')[-1].split('_')[0]
    source_bone_path = f"demos/VRServer/datasets/bone_data/{source_char_id}.json"
    source_bone_data = read_json_file(source_bone_path)

    animation_data = read_json_file(file_path)
    frames = []

    if char_id and char_id != source_char_id:
        # Retargeting is needed
        retargetmap_path = f"demos/VRServer/datasets/retargetmap/{source_char_id}2{char_id}.bmap"
        target_bone_path = f"demos/VRServer/datasets/bone_data/{char_id}.json"
        target_bone_data = read_json_file(target_bone_path)
        bone_mapping = read_bmap_file(retargetmap_path)
        
        source_bones = source_bone_data['bones']
        target_bones = target_bone_data['bones']

        # Find root bones for both skeletons
        source_root = next(bone for bone, data in source_bones.items() if data['parent'] is None)
        target_root = next(bone for bone, data in target_bones.items() if data['parent'] is None)
        
        # Calculate scaling factor between skeletons
        scale_factor = calculate_scale_factor(source_bone_data, target_bone_data, source_root, target_root)
        
        for frame in animation_data:
            frame_dict = {'frame': frame['frame']}
            remapped_loc = None

            for joint_name, transform in frame.items():
                if joint_name == 'frame':
                    continue

                if 'rotation' in joint_name:
                    # Handle rotation retargeting
                    source_bone_name = joint_name.replace('_rotation', '')
                    target_bone_name = bone_mapping.get(source_bone_name, "NP")
                    if target_bone_name == "NP":
                        continue
                    else:
                        # Get local rest poses of the source and target bones
                        source_matrix_local = source_bones[source_bone_name]['matrix_local']
                        target_matrix_local = target_bones[target_bone_name]['matrix_local']
                        rest_diff = np.linalg.inv(source_matrix_local) @ target_matrix_local
                        rest_diff_quat = tf.quaternion_from_matrix(rest_diff)
                        
                        if target_bone_name == target_root:
                            # Root bone special handling
                            blender_rotation = (transform[3], transform[0], -transform[1], -transform[2])
                            adjusted_rotation = tf.quaternion_multiply(blender_rotation, rest_diff_quat)
                            unity_rotation = (adjusted_rotation[1], -adjusted_rotation[2], -adjusted_rotation[3], adjusted_rotation[0])
                            frame_dict[target_bone_name + '_rotation'] = [round(num, 7) for num in unity_rotation]
                        else:
                            # Non-root bones need parent transformations as well
                            source_parent_name = source_bones[source_bone_name]['parent']
                            target_parent_name = target_bones[target_bone_name]['parent']
                            source_parent_matrix_local = source_bones[source_parent_name]['matrix_local']
                            target_parent_matrix_local = target_bones[target_parent_name]['matrix_local']
                            parent_diff = np.linalg.inv(target_parent_matrix_local) @ source_parent_matrix_local
                            parent_diff_quat = tf.quaternion_from_matrix(parent_diff)
                            blender_rotation = (transform[3], transform[0], -transform[1], -transform[2])
                            adjusted_rotation = tf.quaternion_multiply(parent_diff_quat, blender_rotation)
                            adjusted_rotation2 = tf.quaternion_multiply(adjusted_rotation, rest_diff_quat)
                            unity_rotation = (adjusted_rotation2[1], -adjusted_rotation2[2], -adjusted_rotation2[3], adjusted_rotation2[0])
                            frame_dict[target_bone_name + '_rotation'] = [round(num, 7) for num in unity_rotation]

                elif 'position' in joint_name:
                    # Scale the root position
                    remapped_loc = np.array(transform) * scale_factor
                    frame_dict[target_root + '_position'] = [round(num, 7) for num in remapped_loc]

            frames.append(frame_dict)
    else:
        # No retargeting needed, just copy the animation data
        for frame in animation_data:
            frame_dict = {'frame': frame['frame']}
            for joint_name, transform in frame.items():
                if joint_name == 'frame':
                    continue
                frame_dict[joint_name] = transform

            frames.append(frame_dict)
    return frames
