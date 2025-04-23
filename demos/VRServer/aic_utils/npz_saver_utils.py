import numpy as np
import asyncio
import transformations as tf
from aic_utils.redis_utils import RedisConnection
from aic_utils.file_utils import read_json_file, read_bmap_file, get_key_from_value
from aic_utils.coordinate_conversion_utils import convert_unity_to_blender, convert_to_rest_pose, apply_additional_rotation_inverse
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor()
file_lock = asyncio.Lock()

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

# config = read_json_file("datasets/standard_bone_config.json")
redis_comm = RedisConnection('localhost', 6379, 0, 'userStream')
last_id_user = '0-0'
last_id_ai = '0-0'

def savez_wrapper(file_path, data):
    """
    Save data to an NPZ file.
    
    Args:
        file_path (str): Path where the NPZ file will be saved
        data (dict): Dictionary of arrays to save
    """
    np.savez(file_path, **data)
    
def quaternion_to_axis_angle(quaternion):
    """
    Convert a quaternion to an axis-angle representation (Rodrigues vector).
    
    This function takes a quaternion [w, x, y, z] and converts it to an axis-angle
    representation where the direction of the vector represents the rotation axis
    and the magnitude represents the rotation angle in radians.
    
    Args:
        quaternion (numpy.ndarray): Quaternion [w, x, y, z]
        
    Returns:
        numpy.ndarray: Axis-angle representation (Rodrigues vector)
    """
    q = np.array(quaternion)
    theta = 2* np.arccos(q[0])
    sin_theta_over_2 = np.sqrt(1 - q[0] ** 2)
    
    if sin_theta_over_2 < 1e-8:
        axis = np.array([1.0, 0.0, 0.0])  # Default axis for very small rotations
    else:
        axis = q[1:] / sin_theta_over_2
    return axis * theta

def retarget_back(char_id):
    """
    Prepare for reverse retargeting from a character back to SMPLX skeleton.
    
    This function loads the necessary mapping data and computes scaling factors
    needed to convert animations from a specific character to the standard SMPLX
    skeleton.
    
    Args:
        char_id (str): ID of the character to retarget from
        
    Returns:
        tuple: (inverse_bone_mapping, scale_factor, source_bones, target_bones)
               - inverse_bone_mapping: Mapping from character bones to SMPLX bones
               - scale_factor: Scaling factor to apply to translations
               - source_bones: Bone data for the source character
               - target_bones: Bone data for the SMPLX skeleton
    """
    retargetmap_path = f"datasets/retargetmap/Smplx2{char_id}.bmap"
    bone_mapping = read_bmap_file(retargetmap_path)
    source_bone_path = f"datasets/bone_data/{char_id}.json"
    target_bone_path = f"datasets/bone_data/smplx.json"
    source_armature_matrix = np.array(read_json_file(source_bone_path)['armature']['armature_matrix'])
    target_armature_matrix = np.array(read_json_file(source_bone_path)['armature']['armature_matrix'])
    source_bones = read_json_file(source_bone_path)['bones']
    target_bones = read_json_file(target_bone_path)['bones']
    
    # Calculate scale factor based on character heights
    source_root = next(bone for bone, data in source_bones.items() if data['parent'] is None)
    source_root_matrix = np.array(source_bones[source_root]['matrix_local'])
    target_root_matrix = np.array(target_bones['pelvis']['matrix_local'])
    source_global_matrix = source_armature_matrix @ source_root_matrix
    target_global_matrix = target_armature_matrix @ target_root_matrix
    source_height = source_global_matrix[:3, 3][2]
    target_height = target_global_matrix[:3, 3][2]
    scale_factor = target_height / source_height
    
    return bone_mapping.inv, scale_factor, source_bones, target_bones
    
async def save_npz(file_path, rotmat_user, transl_user, rotmat_ai, transl_ai):
    """
    Asynchronously save animation data to an NPZ file.
    
    This function saves the user and AI motion data to a single NPZ file, using
    a lock to prevent concurrent file access.
    
    Args:
        file_path (str): Path where the NPZ file will be saved
        rotmat_user (numpy.ndarray): User pose data in axis-angle format
        transl_user (numpy.ndarray): User root translations
        rotmat_ai (numpy.ndarray): AI pose data in axis-angle format
        transl_ai (numpy.ndarray): AI root translations
    """
    data = {
        'poses': rotmat_user,
        'trans': transl_user,
        "poses_ai": rotmat_ai,
        "trans_ai": transl_ai
    }
    async with file_lock:
        np.savez(file_path, **data)
        print(f"File saved at {file_path}")

async def process_redis_data(file_path):
    """
    Process motion data from Redis streams and save it to an NPZ file.
    
    This function retrieves animation data from Redis streams for both user and AI,
    processes and converts the data from Unity format to SMPLX format, and saves
    the combined data to an NPZ file.
    
    Args:
        file_path (str): Path where the NPZ file will be saved
        
    Returns:
        str: Path to the saved file
    """
    global last_id_user
    global last_id_ai
    
    # Retrieve messages from Redis streams
    messages_user, last_id_user = await redis_comm.receive_frame_data('userStream', last_id_user, count=300)
    messages_ai, last_id_ai = await redis_comm.receive_frame_data('aifbStream', last_id_ai, count = 300)
    
    # Initialize arrays for pose and translation data
    frames_user = len(messages_user)
    joints = len(SMPLX_JOINT_NAMES)
    rotmat_user = np.zeros((frames_user, joints*3))
    transl_user = np.zeros((frames_user, 3))
    rotmat_ai = np.zeros((1, joints*3))
    transl_ai = np.zeros((1, 3))
    
    # Process user motion data
    for frame_index, (message_id, message) in enumerate(messages_user):
        for key, value in message.items():
            if key.endswith('_rotation'):
                joint_name = key.replace('_rotation', '')
                if joint_name in SMPLX_JOINT_NAMES:
                    joint_index = SMPLX_JOINT_NAMES.index(joint_name)
                    unity_rot = np.array([float(v) for v in value.split(',')])
                    # Convert from Unity to Blender rotation format
                    blender_rot = (unity_rot[3], unity_rot[0], -unity_rot[1], -unity_rot[2])
                    if joint_name == "pelvis":
                        # Apply additional rotation for the root
                        rotation_quat = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0, 0])
                        blender_rot = tf.quaternion_multiply(rotation_quat, blender_rot)
                    # Convert to axis-angle representation
                    axis_angle = quaternion_to_axis_angle(blender_rot)
                    rotmat_user[frame_index, joint_index *3: joint_index*3 +3] = axis_angle
            elif key.endswith('_position'):
                unity_pos = np.array([float(v) for v in value.split(',')])
                # Convert position from Unity to Blender coordinate system
                transl_user[frame_index] = (-unity_pos[0], -unity_pos[2], unity_pos[1])
        # Delete processed entries from Redis
        await redis_comm.delete_processed_entries('userStream', [message_id])
                
    # Get current character ID and prepare for retargeting
    char_id = redis_comm.r.get("char_id")
    char_id = char_id.decode('utf-8')
    bone_mapping, scale_factor, source_bones, target_bones = retarget_back(char_id)
    
    # Process AI motion data
    for frame_index, (message_id_ai, message) in enumerate(messages_ai):
        for key, value in message.items():
            if key.endswith('_rotation'):
                unity_rot = np.array([float(v) for v in value.split(',')])
                # Convert from Unity to Blender rotation format
                blender_rot = (unity_rot[3], unity_rot[0], -unity_rot[1], -unity_rot[2])
                joint_name = key.replace('_rotation', '')
                if joint_name in bone_mapping:
                    mapping_name = bone_mapping[joint_name]
                    joint_index = SMPLX_JOINT_NAMES.index(mapping_name)
                    
                    # Apply appropriate transformations based on skeleton differences
                    source_matrix_local = source_bones[joint_name]['matrix_local']
                    target_matrix_local = target_bones[mapping_name]['matrix_local']
                    rest_diff = np.linalg.inv(source_matrix_local) @ target_matrix_local
                    rest_diff_quat = tf.quaternion_from_matrix(rest_diff)
                    
                    if mapping_name == "pelvis":
                        # Special handling for the root joint
                        rotation_x_quat = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0, 0])
                        rotation_z_quat = np.array([0, 0, 0, 1])
                        rotation_quat = tf.quaternion_multiply(rotation_z_quat, rotation_x_quat)
                        blender_rot = tf.quaternion_multiply(rotation_quat, blender_rot)
                        blender_rot = tf.quaternion_multiply(blender_rot , rest_diff_quat)
                    else:
                        # Account for parent bone differences for non-root joints
                        source_parent_name = source_bones[joint_name]['parent']
                        target_parent_name = target_bones[mapping_name]['parent']
                        source_parent_matrix_local = source_bones[source_parent_name]['matrix_local']
                        target_parent_matrix_local = target_bones[target_parent_name]['matrix_local']
                        parent_diff = np.linalg.inv(target_parent_matrix_local) @ source_parent_matrix_local
                        parent_diff_quat = tf.quaternion_from_matrix(parent_diff)
                        blender_rot = tf.quaternion_multiply(parent_diff_quat, blender_rot)
                        blender_rot = tf.quaternion_multiply(blender_rot , rest_diff_quat)
                    
                    # Convert to axis-angle representation
                    axis_angle = quaternion_to_axis_angle(blender_rot)
                    rotmat_ai[0, joint_index *3:joint_index*3+3] = axis_angle
                else:
                    continue
            elif key.endswith('_position'):
                unity_pos = np.array([float(v) for v in value.split(',')])
                # Convert and scale the position
                transl_ai[0] = (-unity_pos[0]/scale_factor/1.03, -unity_pos[2]/scale_factor/1.03, unity_pos[1]/scale_factor/1.03)
        
        # Delete processed entries from Redis
        await redis_comm.delete_processed_entries('aifbStream', [message_id_ai])    
    
    # Trim frames to keep a steady window of the animation
    if frames_user > 45:
        rotmat_user = rotmat_user[30:-15]
        transl_user = transl_user[30:-15]
             
    # Save the processed data
    await save_npz(file_path, rotmat_user, transl_user, rotmat_ai, transl_ai)
    print(f"process motion file finished")
    
    return file_path