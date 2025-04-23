import requests
import os
import io

def get_server_info()->tuple:
    """
    Get server URL and API key for the audio to face service.
    Users need to set up their own audio to face service based on UniTalker
    (https://github.com/X-niper/UniTalker)
    
    Returns:
        tuple: (base_url, apikey) for the audio to face service
    """
    base_url = ""  # The URL of your Audio2Face service
    apikey =  ""   # Your API key for authentication
    return base_url, apikey

def generate(audio_path: str, model_name = None, emo_id = 11):
    """
    Send audio file to the Audio2Face service to generate face animation data.
    This requires setting up your own server using UniTalker
    (https://github.com/X-niper/UniTalker)
    
    Args:
        audio_path (str): Path to the audio file
        model_name (str, optional): Model name to use for generation
        emo_id (int, optional): Emotion ID to apply. Defaults to 11
        
    Returns:
        tuple: (binary_content, error_message) - binary content is the response data,
               error_message is None if successful
    """
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
    """
    Save binary face animation data to a file.
    
    Args:
        data (bytes): Binary data to save
        temp_dir (str): Path to save the data
        
    Returns:
        str: Path where the data was saved
    """
    dir_path = os.path.dirname(temp_dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with io.BytesIO(data) as file:
        with open(temp_dir, 'wb') as file:
            file.write(data)
        return temp_dir
    
def generate_face_data(wav_file: str, save_path: str, callback=None):
    """
    Generate facial animation data from audio file and save it.
    
    Args:
        wav_file (str): Path to the input audio file
        save_path (str): Path to save the output face animation data
        callback (callable, optional): Function to call with results when complete
        
    Returns:
        str: Path to the saved animation data if successful, None otherwise
    """
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