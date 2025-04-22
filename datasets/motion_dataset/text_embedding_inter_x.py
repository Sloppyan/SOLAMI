import os
from openai import AzureOpenAI, OpenAI
import json
import numpy as np
from tqdm import tqdm
import logging
import time
import argparse
from text_embedding_humanml3d import get_embedding, calculate_api_cost, AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT
TOKEN_BLOGS = {}

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Text embedding Inter-X process')
    parser.add_argument('--period', type=int, default=8)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    
    text_root_dir = "SOLAMI_data/Inter-X/texts_post_processed"
    save_dir = "SOLAMI_data/Inter-X/embeddings"
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
    
    file_paths = [os.path.join(text_root_dir, name) for name in names]

    file_paths = file_paths[args.part::args.period]
    if args.debug:
        file_paths = file_paths[:10]
    
    embeddings = {}
    if os.path.exists(save_path):
        data = np.load(save_path, allow_pickle=True)
        embeddings = data['embeddings'].item()
    
    updated = False
    for idx in tqdm(range(len(file_paths))):
        file_path = file_paths[idx]
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for item in data['texts']:
            if item['option'] != 'N':
                for key in ['actor', 'reactor']:
                    line = item[key]
                    line = line.strip()
                    if line.endswith('.'):
                        line = line[:-1]
                    if line in embeddings.keys():
                        continue
                    embeddings[line] = get_embedding(line, model=Embedding_Model, tokens_blog=TOKEN_BLOGS)
                    updated = True
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