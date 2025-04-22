import os
from tqdm import tqdm
import json
import numpy as np
import logging


configs = {
    'dlp': {
        'text_embeddings': "SOLAMI_data/DLP-MoCap/embeddings/embeddings.npz",
        'dataset_items': "SOLAMI_data/DLP-MoCap/dataset_items_post.json",
        'dataset_root_dir': "SOLAMI_data/DLP-MoCap",
    },
    'humanml3d': {
        'text_embeddings': "SOLAMI_data/HumanML3D/embeddings/embeddings.npz",
        'dataset_items': "SOLAMI_data/HumanML3D/dataset_items_post.json",
        'dataset_root_dir': "SOLAMI_data/HumanML3D",
    },
    'inter-x': {
        'text_embeddings': "SOLAMI_data/Inter-X/embeddings/embeddings.npz",
        'dataset_items': "SOLAMI_data/Inter-X/dataset_items_post.json",
        'dataset_root_dir': "SOLAMI_data/Inter-X",
    },
}


class UnifiedDataset:
    def __init__(self, configs, logger=None):
        self.configs = configs
        if logger is None:
            self.logger = logging.getLogger('my_logger')
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            self.logger.addHandler(ch)
        else:
            self.logger = logger
        self.text_embeddings = {}
        self.dataset_items = {}
        self.text_to_actions = {}
        self.load()
        
    def load_text_embeddings(self):
        for key in self.configs:
            embeddings = np.load(self.configs[key]['text_embeddings'], allow_pickle=True)
            self.text_embeddings.update(embeddings['embeddings'].item())
        self.logger.info('Loaded text embeddings from all datasets')
        
    def load_dataset_items(self):
        for key in self.configs:
            with open(self.configs[key]['dataset_items'], 'r') as f:
                self.dataset_items.update(json.load(f))
        self.logger.info(f'Loaded total {len(self.dataset_items)} dataset items from all datasets')
        
    def check_dataset_connection(self):
        for item_name, item in self.dataset_items.items():
            if item['last_partner_motion_name'] is not None:
                if item['last_partner_motion_name'] not in self.dataset_items:
                    self.logger.info(f"Item {item_name} has a partner item {item['last_partner_motion_name']} not found in the dataset")
                    item['last_partner_motion_name'] = None
            if item['next_partner_motion_name'] is not None:
                if item['next_partner_motion_name'] not in self.dataset_items:
                    self.logger.info(f"Item {item_name} has a partner item {item['next_partner_motion_name']} not found in the dataset")
                    item['next_partner_motion_name'] = None
        self.logger.info('Checked dataset connection')
               
    def get_text_to_actions_dict(self):
        for item_name, item in self.dataset_items.items():
            for text in item['text']:
                if text not in self.text_to_actions:
                    self.text_to_actions[text] = []
                self.text_to_actions[text].append(item)
        self.logger.info(f'Generated text to actions dict with {len(self.text_to_actions)} entries')
        
    def check_text_embeddings(self):
        texts = list(self.text_embeddings.keys())
        count = 0
        count_2 = 0
        for text in texts:
            if text not in self.text_to_actions:
                self.text_embeddings.pop(text)
                count += 1
        for text in self.text_to_actions:
            if text not in texts:
                count_2 += 1
        self.logger.info(f'Removed {count} text embeddings not found in text to actions dict')
        self.logger.info(f'{count_2} text to actions not found in text embeddings')
            
    
    def load(self):
        self.load_text_embeddings()
        self.load_dataset_items()
        self.check_dataset_connection()
        self.get_text_to_actions_dict()
        self.check_text_embeddings()
        self.embeddings_text = list(self.text_embeddings.keys())
        self.embeddings_np = np.array(list(self.text_embeddings.values())).transpose(1, 0)
        self.embeddings_np /= np.linalg.norm(self.embeddings_np, axis=0, keepdims=True)

if __name__ == "__main__":
    unified_dataset = UnifiedDataset(configs)
    print(len(unified_dataset.text_embeddings))
    print(len(unified_dataset.dataset_items))