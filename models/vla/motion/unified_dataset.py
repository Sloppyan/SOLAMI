import os
from tqdm import tqdm
import json
import numpy as np
import logging
from FlagEmbedding import FlagModel
import torch
import torch.distributed as dist
import debugpy

def initialize_debugpy():
    # if not dist.is_initialized() or dist.get_rank() == 0:
        # print(f"Rank: {dist.get_rank()} - Debugpy is listening on port 15696")
        print("Debugpy is listening on port 15696")
        debugpy.listen(("0.0.0.0", 15696))
        debugpy.wait_for_client()
        
# def initialize_distributed():
#     if not dist.is_initialized():
#         dist.init_process_group(backend='nccl')

# initialize_distributed()
# initialize_debugpy()


class UnifiedDataset:
    def __init__(self, configs, logger=None, device='cuda:0'):
        self.configs = configs
        self.device = device
        self.text_embeddings = {}
        self.dataset_items = {}
        self.text_to_actions = {}
        self.load()
        
    def load_text_embeddings(self):
        for key in self.configs:
            embeddings = np.load(self.configs[key]['text_embeddings'], allow_pickle=True)
            self.text_embeddings.update(embeddings['embeddings'].item())
        logging.info('Loaded text embeddings from all datasets')
        
    def load_dataset_items(self):
        for key in self.configs:
            with open(self.configs[key]['dataset_items'], 'r') as f:
                self.dataset_items.update(json.load(f))
        logging.info(f'Loaded total {len(self.dataset_items)} dataset items from all datasets')
        
    def check_dataset_connection(self):
        for item_name, item in self.dataset_items.items():
            if item['last_partner_motion_name'] is not None:
                if item['last_partner_motion_name'] not in self.dataset_items:
                    logging.info(f"Item {item_name} has a partner item {item['last_partner_motion_name']} not found in the dataset")
                    item['last_partner_motion_name'] = None
            if item['next_partner_motion_name'] is not None:
                if item['next_partner_motion_name'] not in self.dataset_items:
                    logging.info(f"Item {item_name} has a partner item {item['next_partner_motion_name']} not found in the dataset")
                    item['next_partner_motion_name'] = None
        logging.info('Checked dataset connection')
               
    def get_text_to_actions_dict(self):
        for item_name, item in self.dataset_items.items():
            for text in item['text']:
                if text not in self.text_to_actions:
                    self.text_to_actions[text] = []
                self.text_to_actions[text].append(item)
        logging.info(f'Generated text to actions dict with {len(self.text_to_actions)} entries')
        
    def check_text_embeddings(self):
        texts = list(self.text_embeddings.keys())
        count = 0
        # count_2 = 0
        for text in texts:
            if text not in self.text_to_actions:
                self.text_embeddings.pop(text)
                count += 1
        # for text in self.text_to_actions:
        #     if text not in texts:
        #         count_2 += 1
        logging.info(f'Removed {count} text embeddings not found in text to actions dict')
        # logging.info(f'{count_2} text to actions not found in text embeddings')
    
    
    def retrieval_motion_by_embeddings(self, query_embeddings, top_k=1):
        query_embeddings = query_embeddings.reshape(1, -1)
        query_embeddings = torch.tensor(query_embeddings).to(self.device)
        similarities = (query_embeddings @ self.embeddings_pt).flatten()
        top_k_indices = torch.argsort(similarities, descending=True)[:top_k]
        top_k_texts = [self.embeddings_text[i] for i in top_k_indices]
        top_k_items = []
        for text in top_k_texts:
            top_k_items.extend(self.text_to_actions[text])
        
        results = np.random.choice(top_k_items, top_k, replace=False)
        
        motions = []
        for item in results:
            motion_tmp = item['motion_name']
            motions.append(motion_tmp)
        return results, motions
    
    
    def load(self):
        self.load_text_embeddings()
        self.load_dataset_items()
        self.check_dataset_connection()
        self.get_text_to_actions_dict()
        self.check_text_embeddings()
        self.embeddings_text = list(self.text_embeddings.keys())
        self.embeddings_pt = torch.tensor(list(self.text_embeddings.values())).transpose(1, 0).to(self.device)
        self.embeddings_pt /= torch.linalg.norm(self.embeddings_pt, axis=0, keepdims=True)

if __name__ == "__main__":
    
    configs = {
        'dlp': {
            'text_embeddings': "SOLAMI_data/DLP-MoCap/embeddings/bge_large_embeddings.npz",
            'dataset_items': "SOLAMI_data/DLP-MoCap/dataset_items_post.json",
            'dataset_root_dir': "SOLAMI_data/DLP-MoCap",
        },
        'humanml3d': {
            'text_embeddings': "SOLAMI_data/HumanML3D/embeddings/bge_large_embeddings.npz",
            'dataset_items': "SOLAMI_data/HumanML3D/dataset_items_post.json",
            'dataset_root_dir': "SOLAMI_data/HumanML3D",
        },
        'inter-x': {
            'text_embeddings': "SOLAMI_data/Inter-X/embeddings/bge_large_embeddings.npz",
            'dataset_items': "SOLAMI_data/Inter-X/dataset_items_post.json",
            'dataset_root_dir': "SOLAMI_data/Inter-X",
        },
    }

    key = {0: 'dlp', 1: 'humanml3d', 2: 'inter-x'}[1]
    configs = {key: configs[key]}
    unified_dataset = UnifiedDataset(configs)
    print(len(unified_dataset.text_embeddings))
    print(len(unified_dataset.dataset_items))
    
    test = "Shakes hands and nodes"
    model = FlagModel('BAAI/bge-large-en-v1.5',
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
        query_instruction_format="{}{}",
        use_fp16=True,
        devices="cuda:0",   # if you don't have a GPU, you can use "cpu"
        pooling_method='cls',)
    
    query_embedding = model.encode_queries([test])
    actions, motions = unified_dataset.retrieval_motion_by_embeddings(query_embedding[0], top_k=3)
    
    print(actions)
    print(motions)
    pass