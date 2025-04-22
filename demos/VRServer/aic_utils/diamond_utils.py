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
    url = "https://humman-utils.diamond.zoe.sensetime.com/api/v1/pipeline/"
    with open(input_path, 'rb') as f:
        audio_data = f.read()

    response = requests.post(url, files={'file': audio_data}, verify=False)  # 关闭 SSL 验证

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
    # url = "http://kub-api.zoe.sensetime.com/embodied/forward1/"
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
    # url = "http://kub-api.zoe.sensetime.com/embodied/forward1/"
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
    
# content, exception = create_session("test", "echo", "char_0")
# print(content)

def download_file(url, output_path):
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
    # url = "http://kub-api.zoe.sensetime.com/embodied/forward1/"
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
                # print(motion_url)
                # print(audio_url)
                download_file(motion_url, motion_ai)
                download_file(audio_url, audio_ai)                
            return content, None
    except Exception as e:
        print(str(e))
        return None, str(e)
    

# session_id, exception = create_session("test", "echo", "char_0")  
# if session_id:
#     print("Session ID:", session_id)
#     audio_user_path = r"E:/repos/redis_anim/datasets/audio/audio_user.wav"
#     motion_user_path = r"E:/repos/redis_anim/datasets/anim_data/motion_user.npz"
#     audio_ai_path = r"E:/repos/redis_anim/datasets/audio/audio_ai.wav"
#     motion_ai_path = r"E:/repos/redis_anim/datasets/anim_data/motion_ai.npz"
#     content, exception = interact(session_id, audio_user_path, motion_user_path, audio_ai_path, motion_ai_path)
#     if content:
#         print("Response content:", content)
#     else:
#         print("Error:", exception)
# else:
#     print("Session creation failed:", exception)
    
    
