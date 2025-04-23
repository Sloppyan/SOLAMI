from aic_utils.json_loader_utils import load_json_anim
from aic_utils.npz_loader_utils import load_npz_anim
from aic_utils.npy_loader_utils import load_npy_anim
from aic_utils.face_csv_loader_utils import load_csv_face_anim
from aic_utils.face_npz_loader_utils import load_npz_face_anim
from aic_utils.file_utils import read_json_file

def load_config(config_path):
    """
    Load configuration data from a JSON file.
    
    Args:
        config_path (str): Path to the configuration JSON file
        
    Returns:
        dict: The loaded configuration data
    """
    return read_json_file(config_path)
    
def load_motion_data(file_path, char_id):
    """
    Load motion animation data from various file formats.
    
    This function serves as a unified interface for loading motion data
    from different file formats (JSON, NPZ, NPY), selecting the appropriate
    loader based on the file extension.
    
    Args:
        file_path (str): Path to the animation data file
        char_id (str): Character ID for which the animation is being loaded
        
    Returns:
        dict: Loaded animation data appropriate for the specified character
        
    Raises:
        ValueError: If the file format is not supported
    """
    if file_path.endswith('.json'):
        return load_json_anim(file_path, char_id)
    elif file_path.endswith('.npz'):
        return load_npz_anim(file_path, char_id)
    elif file_path.endswith('.npy'):
        return load_npy_anim(file_path, char_id)
    else:
        raise ValueError(F"Unsupported file format: {file_path}")

def load_face_data(file_path):
    """
    Load facial animation data from various file formats.
    
    This function serves as a unified interface for loading facial animation
    data from different file formats (CSV, NPZ), selecting the appropriate
    loader based on the file extension.
    
    Args:
        file_path (str): Path to the facial animation data file
        
    Returns:
        dict: Loaded facial animation data in a standardized format
    """
    if file_path.endswith('.csv'):
        return load_csv_face_anim(file_path)
    elif file_path.endswith('.npz'):
        return load_npz_face_anim(file_path)