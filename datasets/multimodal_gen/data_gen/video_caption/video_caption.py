import os
from openai import AzureOpenAI, OpenAI
import cv2 
import base64
import json

AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
openai_api_key = '$YOUR_OPENAI_API_KEY' # todo
openai_base_url = 'https://api.openai.com/v1'


## TODO choose the api you want to use and change the video path
USE_AZURE = True
video_path = 'hug.mp4'

# Attention!!!
# Max frames of videos into AzureOpenAI API in this script is 20
# While for your personal openai api, you can use more frames
# If you want to use more frames in azure openai api, you can split the video into several parts and process them separately
# Or you can refer to this: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/gpt-with-vision?tabs=python%2Csystem-assigned%2Cresource


def process_video(video_path, seconds_per_frame=0.2):
    base64Frames = []
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame=0

    # Loop through the video and extract frames at specified sampling rate
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()
    print(f"Extracted {len(base64Frames)} frames")
    return base64Frames

# Extract 1 frame per second. You can adjust the `seconds_per_frame` parameter to change the sampling rate
base64Frames = process_video(video_path, seconds_per_frame=0.3)


# TODO change you prompt here!
# here we output json format results
# Basic principles for design prompt:
# task description, input format, output format, example

SYS_PROMPT = """You are generating a video summary. 
                Give you some frames of a video about the human interactive motion of two characters. 
                Please generate a brief semantic description of the motion in the video.
                THe description should only include the main actions of the characters, and should be no more than 10 words.
                You must response in json format. 'summary' means the main action of the characters in the video.
                Example of the output format:
                {"summary": "One hugs another, expressing affection and care."}
                """
                
if not USE_AZURE:
    # OAI prompt
    PROMPT_MESSAGES = [
        {
            "role": "system",
            "content": SYS_PROMPT,
        },
        {
            "role": "user", 
            "content": [
            "These are the frames from the video.",
            *map(lambda x: {"type": "image_url", 
                            "image_url": {"url": f'data:image/jpg;base64,{x}', "resize": 768}}, base64Frames)
            ],
        }
    ]
else:
    # Azure prompt
    PROMPT_MESSAGES = [
        {
            "role": "system",
            "content": SYS_PROMPT,
        },
        {
            "role": "user", 
            "content": [
                {
                    "type": "text",
                    "text": "These are the frames from the video."
                },
            *map(lambda x: {"type": "image_url", 
                            "image_url": {"url": f'data:image/jpg;base64,{x}', "resize": 768}}, base64Frames)
            ],
        }
    ]

params = {
    "model": "gpt-4o",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 1200,
    "response_format":  {"type": "json_object"},
}

if USE_AZURE:
    client = AzureOpenAI(api_key=AZURE_OPENAI_API_KEY, azure_endpoint=AZURE_OPENAI_ENDPOINT, api_version="2024-05-01-preview")
else:
    client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

completion = client.chat.completions.create(
        model='gpt-4o', 
        messages=PROMPT_MESSAGES,
        response_format={"type": "json_object"},
        )
results = json.loads(completion.choices[0].message.content)
print(results)
