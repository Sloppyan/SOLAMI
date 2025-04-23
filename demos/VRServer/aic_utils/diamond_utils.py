import requests
import base64
import warnings
import json

warnings.filterwarnings("ignore", message="Unverified HTTPS request")
character_dict = {
        "Batman": 'Batman',
        "Bnncat": 'Banaya',
        "Chappie": '11-45-G',
        "Kvrc": '11-45-G',
        "Link": 'Link',
        "Smplx": 'User',
        "Trump": 'Trump',
        "Vrmgirl": 'Samantha',
        "Test": 'echo_default_character'
    }

def audio2audio(input_path, output_path):
    """
    Process an audio file through the backend service and save the converted audio.
    
    This function sends an audio file to the backend service for processing
    and saves the returned transformed audio.
    
    Args:
        input_path (str): Path to the input audio file
        output_path (str): Path to save the processed audio file
    """
    url = "" # The URL of your solami service
    with open(input_path, 'rb') as f:
        audio_data = f.read()

    response = requests.post(url, files={'file': audio_data}, verify=False)  # SSL verification disabled

    if response.status_code == 200:
        try:
            audio_bytes = base64.b64decode(response.json()['audio_data'])
            with open(output_path, 'wb') as f:
                f.write(audio_bytes)
        except KeyError:
            print("KeyError: 'audio_data' not found in the response")
            print("Response content:", response.content)
    else:
        print("Request failed with status code:", response.status_code)
        print("Response content:", response.content)
        
def delete_session(url, session_id):
    """
    Delete a session from the backend service.
    
    Users need to deploy their own backend service for this functionality.
    
    Args:
        url (str): Base URL of the backend service
        session_id (str): ID of the session to delete
        
    Returns:
        tuple: (response_content, error_message) - response_content is the JSON response,
               error_message is None if successful
    """
    try:
        response = requests.post(
            f"{url}/delete_session/",
            data = dict(session_id = session_id),
        )
        response.raise_for_status()
        content = json.loads(response.content)
        print(f"delete session {content}")
        return content, None
    except Exception as e:
        print(str(e))
        return None, str(e)
        
        
def create_session(url, session_name, method, character_id):
    """
    Create a new session with the backend service.
    
    Users need to deploy their own backend service for this functionality.
    
    Args:
        url (str): Base URL of the backend service
        session_name (str): Name for the new session
        method (str): Method type for the session (e.g., "echo")
        character_id (str): ID of the character to use in the session
        
    Returns:
        tuple: (session_id, error_message) - session_id if successful,
               error_message is None if successful
    """
    character = character_dict[character_id]
    print(f"start create_session:")
    print(f"{character_id} + {method}")
    try:
        response = requests.post(
            f"{url}/create_session/",
            data = dict(session_name = session_name,method = method, character = character ),
        )
        # print(f"request: {dict(session_name = session_name,method = method, character = character )}")
        response.raise_for_status()
        content = json.loads(response.content)
        session_id = content["session_id"]
        print(f"{session_id}")
        return session_id, None
    except Exception as e:
        print(str(e))
        return None, str(e)
    

def download_file(url, output_path):
    """
    Download a file from a URL and save it to the specified path.
    
    Args:
        url (str): URL of the file to download
        output_path (str): Path where the downloaded file will be saved
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size = 8192):
                f.write(chunk)
        print(f"download file {url} successful: {output_path}")
    except Exception as e:
        print(f"Failed to download file from {url}: {str(e)}")
        
    
def interact(url, session_id, audio_user, motion_user, audio_ai, motion_ai):
    """
    Send audio and motion data to the backend service and receive processed responses.
    
    This function handles the communication with the backend service for sending
    user audio/motion and receiving AI-generated audio/motion.
    Users need to deploy their own backend service for this functionality.
    
    Args:
        url (str): Base URL of the backend service
        session_id (str): ID of the active session
        audio_user (str): Path to the user's audio file to send
        motion_user (str): Path to the user's motion file to send
        audio_ai (str): Path to save the AI's audio response
        motion_ai (str): Path to save the AI's motion response
        
    Returns:
        tuple: (response_content, error_message) - response_content is the JSON response,
               error_message is None if successful
    """
    try:
        with open(audio_user, 'rb') as audio_file, open(motion_user, 'rb') as motion_file:
            files = {
                'audio': ('audio_user.wav', audio_file, 'audio/wav'),
                'motion': ('motion_user.npz', motion_file, 'application/octet-stream')
            }
            data = {'session_id': session_id}
            response = requests.post(f"{url}/interact/", files=files, data=data)
            response.raise_for_status()
            content = json.loads(response.content)
            print(f"interact response: {content}")
            if "motion_url" in content and "audio_url" in content:
                motion_url = url + content["motion_url"]
                audio_url = url + content["audio_url"]
                print("start download file!!!!")
                download_file(motion_url, motion_ai)
                download_file(audio_url, audio_ai)                
            return content, None
    except Exception as e:
        print(str(e))
        return None, str(e)
    
    
