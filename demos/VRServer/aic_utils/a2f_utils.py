import requests
import os
import io

def get_server_info()->tuple:
    base_url = ""
    apikey =  ""
    return base_url, apikey

def generate(audio_path: str, model_name = None, emo_id = 11):
    base_url, apikey = get_server_info()
    try:
        with open(audio_path, 'rb') as f_wav:
            resp = requests.post(
                f"{base_url}/audio2face_upload/",
                files = {'audio': (os.path.basename(audio_path), f_wav, 'application/octet-stream')},
                data = dict(model_name = model_name, emo_id = emo_id),
                verify = False,
                headers = dict(apikey = apikey),
            )
            resp.raise_for_status()
            return resp.content, None
    except Exception as e:
        return None, str(e)
    
def save_data(data: bytes, temp_dir:str) -> str:
    # import numpy as np
    # from datetime import datetime
    dir_path = os.path.dirname(temp_dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with io.BytesIO(data) as file:
        with open(temp_dir, 'wb') as file:
            file.write(data)
        return temp_dir
    
def generate_face_data(wav_file: str, save_path: str, callback=None):
    model_name = "unitalker_emo_base_v0.4.0"
    emo_id = 11
    results = None
    err = None
    face_npz, err = generate(wav_file, model_name, emo_id)
    if face_npz:
        results = save_data(face_npz, save_path)
        err = None
        return results
    if callback:
        callback(results, err)