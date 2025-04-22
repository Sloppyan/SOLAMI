from aic_utils.json_loader_utils import load_json_anim
from aic_utils.npz_loader_utils import load_npz_anim
from aic_utils.npy_loader_utils import load_npy_anim
from aic_utils.face_csv_loader_utils import load_csv_face_anim
from aic_utils.face_npz_loader_utils import load_npz_face_anim
from aic_utils.file_utils import read_json_file

def load_config(config_path):
    return read_json_file(config_path)
    
def load_motion_data(file_path, char_id):
    if file_path.endswith('.json'):
        return load_json_anim(file_path, char_id)
    elif file_path.endswith('.npz'):
        return load_npz_anim(file_path, char_id)
    elif file_path.endswith('.npy'):
        return load_npy_anim(file_path, char_id)
    else:
        raise ValueError(F"Unsupported file format: {file_path}")

def load_face_data(file_path):
    if file_path.endswith('.csv'):
        return load_csv_face_anim(file_path)
    elif file_path.endswith('.npz'):
        return load_npz_face_anim(file_path)