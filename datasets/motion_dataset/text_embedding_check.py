import os
from openai import AzureOpenAI, OpenAI
import json
import numpy as np
from tqdm import tqdm
import logging
import time


AZURE_OPENAI_API_KEY = '$YOUR_API_KEY'
AZURE_OPENAI_ENDPOINT = '$YOUR_ENDPOINT'

LOGS = {'count': 0, 'times': 0}

def get_embedding(text, 
                  model="text-embedding-ada-002", 
                  api_key='',
                  base_url='https://api.openai.com/v1',
                  client='azure',
                  tokens_blog={}, 
                  **kwargs): 
    time_start = time.time()
    if client == 'openai':
        client = OpenAI(api_key=api_key, base_url=base_url)
    elif client == 'azure':
        if api_key == '':
            api_key = AZURE_OPENAI_API_KEY
            azure_endpoint = AZURE_OPENAI_ENDPOINT
        else:
            azure_endpoint = base_url
        client = AzureOpenAI(api_key=api_key, azure_endpoint=azure_endpoint, api_version="2024-05-01-preview")
    else:
        raise ValueError("Invalid client")
    
    text = text.replace("\n", " ")
    if not text: 
        text = "this is blank"
    res = None
    # try:
        # res = openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
    response = client.embeddings.create(input=[text], model=model)
    if model not in tokens_blog.keys():
        tokens_blog[model] = {'prompt': 0}
    tokens_blog[model]['prompt'] += response.usage.prompt_tokens
    res = response.data[0].embedding
    LOGS['count'] += 1
    LOGS['times'] += time.time() - time_start
    return res
    # except:
    #     print("get_embedding ERROR")
        

### test_examples
"""
G001T000A000R001

"""

raw_examples = [
    "Walks towards the other person and hugs the other person around the shoulders, gently patting the other person's back with his/her right hand",
     "Shakes hands with the other person using the right hand",
     "Places his/her left hand in front of his/her left shoulder in a pose",
]

same_motion_examples = [
    "Walks towards the other person, facing the other person. Reaches out his/her hand and places his/her arm around the other person's shoulders",
    "Using the right hand to shake hands",
    "Raises his/her left hand in an upward scissor motion, bringing it to chest level",
]

other_motion_examples = [
    "Kneels down on his/her right knee, with his/her left hand resting on his/her left knee",
    "Wave hands using the right hand",
    "jumping up and down",
]

simplified_examples = [
    "Walks towards and hugs the other person",
    "Shakes hands",
    "left-hand defense on chest",
]

no_motion_examples = [
    "That's a nice day",
    "Hello, how are you?",
    "What's wrong with you?",
]


model = "text-embedding-3-large" 
# model = "text-embedding-ada-002" 
for i in range(3):
    print(f"Example {i}")
    raw_embedding = get_embedding(raw_examples[i], model=model)
    same_motion_embedding = get_embedding(same_motion_examples[i], model=model)
    other_motion_embedding = get_embedding(other_motion_examples[i], model=model)
    simplified_embedding = get_embedding(simplified_examples[i], model=model)
    no_motion_embedding = get_embedding(no_motion_examples[i], model=model)
    similarity_same = np.dot(raw_embedding, same_motion_embedding) / (np.linalg.norm(raw_embedding) * np.linalg.norm(same_motion_embedding))
    similarity_other = np.dot(raw_embedding, other_motion_embedding) / (np.linalg.norm(raw_embedding) * np.linalg.norm(other_motion_embedding))
    similarity_simplified = np.dot(raw_embedding, simplified_embedding) / (np.linalg.norm(raw_embedding) * np.linalg.norm(simplified_embedding))
    similarity_no = np.dot(raw_embedding, no_motion_embedding) / (np.linalg.norm(raw_embedding) * np.linalg.norm(no_motion_embedding))
    print(f"Similarity between raw and same motion: {similarity_same}")
    print(f"Similarity between raw and other motion: {similarity_other}")
    print(f"Similarity between raw and simplified motion: {similarity_simplified}")
    print(f"Similarity between raw and no motion: {similarity_no}") 

print(f"Average time: {LOGS['times'] / LOGS['count']}")