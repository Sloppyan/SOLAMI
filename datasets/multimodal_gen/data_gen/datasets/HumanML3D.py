import os
import json
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


class Action:
    def __init__(self, action_description, line_idx, file_name):
        self.last_self_action = None
        self.last_other_action = None
        self.next_other_action = None
        self.next_self_action = None
        self.dataset_name = 'humanml3d'
        self.sub_action_description = action_description
        self.index = str(file_name) + '---' + str(line_idx)
        self.action_type = 'action_solo'
        pass


class HumanML3D_ActionDatabase:
    def __init__(self, configs):
        self.configs = configs
        self.actions = []
        self.build_actions()

    def build_actions(self):
        data_all = {}
        if os.path.exists(self.configs['text_to_motion_id_path']):
            with open(self.configs['text_to_motion_id_path'], 'r') as f:
                data_all = json.load(f)
        elif os.path.exists(self.configs['text_dir']):
            text_files = os.listdir(self.configs['text_dir'])
            data_all = {}
            for text_file in tqdm(text_files):
                with open(os.path.join(self.configs['text_dir'], text_file), 'r',  encoding="utf-8") as f:
                    lines = f.readlines()
                    data_all[text_file[:-4]] = []
                    for line_idx, line in enumerate(lines):
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        if caption[-1] != '.':
                            caption = caption + '.'
                        data_all[text_file[:-4]].append({'action_description':caption, 'line_idx':line_idx})
            with open(self.configs['text_to_motion_id_path'], 'w') as f:
                json.dump(data_all, f)
        else:
            logging.info("No text data found")
            raise Exception("No text data found")

        for key, value in data_all.items():
            for item in value:
                self.actions.append(Action(item['action_description'], item['line_idx'], key))
        
        logging.info("[HumanML3D] Total actions: {}".format(len(self.actions)))
    
    def get_attr_to_actions_dict(self, attr):
        attr_to_actions = {}
        for action in self.actions:
            if hasattr(action, attr):
                attr_value = getattr(action, attr)
                if attr_value not in attr_to_actions:
                    attr_to_actions[attr_value] = []
                attr_to_actions[attr_value].append(action)
        return attr_to_actions