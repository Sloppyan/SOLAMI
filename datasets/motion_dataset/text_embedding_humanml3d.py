import os
from openai import AzureOpenAI, OpenAI
import json
import numpy as np
from tqdm import tqdm
import logging
import time
import argparse

AZURE_OPENAI_API_KEY = '$YOUR_API_KEY'
AZURE_OPENAI_ENDPOINT = '$YOUR_ENDPOINT'

TOKEN_BLOGS = {}

def get_embedding(text, 
                  model="text-embedding-ada-002", 
                  api_key='',
                  base_url='https://api.openai.com/v1',
                  client='azure',
                  tokens_blog={}, 
                  **kwargs): 
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
    try:
        # res = openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
        response = client.embeddings.create(input=[text], model=model)
        if model not in tokens_blog.keys():
            tokens_blog[model] = {'prompt': 0}
        tokens_blog[model]['prompt'] += response.usage.prompt_tokens
        res = response.data[0].embedding
        return res
    except:
        print("get_embedding ERROR")
        

def calculate_api_cost(tokens_blog={}):
    cost_dict = {
        'gpt-4o': {
            'prompt': 5.,
            'completion': 15.,
        },
        'gpt-4o-2024-05-13': {
            'prompt': 5.,
            'completion': 15.,
        },
        'gpt-3.5-turbo-0125': {
            'prompt': 0.5,
            'completion': 1.5,
        },
        'gpt-3.5-turbo-instruct': {
            'prompt': 1.5,
            'completion': 2.0,
        },
        'text-embedding-ada-002': {
            'prompt': 0.1,
        },
        'text-embedding-3-small': {
            'prompt': 0.02,
        },
        'text-embedding-3-large': {
            'prompt': 0.13,
        }
    }
    
    Cost_all = 0
    Cost = {}
    for model, tokens in tokens_blog.items():
        Cost[model] = {}
        if model in cost_dict.keys():
            for token_type, token_num in tokens_blog[model].items():
                if token_type in cost_dict[model].keys():
                    Cost[model][token_type] = token_num * cost_dict[model][token_type] / 1000000
                    Cost_all += Cost[model][token_type]
    
    return Cost_all, Cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Text embedding humanml3d process')
    parser.add_argument('--period', type=int, default=8)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    
    text_root_dir = "SOLAMI_data/HumanML3D/HumanML3D/texts"
    save_dir = "SOLAMI_data/HumanML3D/embeddings"
    Embedding_Model = 'text-embedding-3-large'
    
    save_path = os.path.join(save_dir, f"embeddings_{args.period}_{args.part}.npz")
    os.makedirs(save_dir, exist_ok=True)
    
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    log_file_path = os.path.join(save_dir,
                                 'text_embeddings_period{}_part{}.log'.format(args.period, args.part))
    print('logger file: ', log_file_path)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    names = os.listdir(text_root_dir)
    names.sort()
    
    file_paths_all = [os.path.join(text_root_dir, name) for name in names]
    file_paths_tmp = file_paths_all[args.part::args.period]
    
    if args.debug:
        file_paths_tmp = file_paths_tmp[:10]
    
    embeddings = {}
    if os.path.exists(save_path):
        data = np.load(save_path, allow_pickle=True)
        embeddings = data['embeddings'].item()
    
    updated = False
    for idx in tqdm(range(len(file_paths_tmp))):
        file_path = file_paths_tmp[idx]
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    line = line.split('#')[0]
                    if line.endswith('.'):
                        line = line[:-1]
                    if line in embeddings.keys():
                        continue
                    embeddings[line] = get_embedding(line, model=Embedding_Model, tokens_blog=TOKEN_BLOGS)
                    updated = True
        except Exception as e:
            logger.info(f"Error in file {file_path}: {e}")
        
        if idx % 100 == 0 and updated:
            np.savez(save_path, embeddings=embeddings)
            logger.info(f"Saved embeddings for {idx} files")
            updated = False
    
    np.savez(save_path, embeddings=embeddings)
    logger.info(f"Saved embeddings for all files")
    
    Cost_all, Cost = calculate_api_cost(tokens_blog=TOKEN_BLOGS)
    logger.info(f"Total cost: {Cost_all}")
    logger.info(f"Cost details:")
    logger.info(json.dumps(Cost, indent=4))            