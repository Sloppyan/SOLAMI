import rich
import random
import pickle
import os
import numpy as np
import codecs as cs
from torch.utils import data
from os.path import join as pjoin
from rich.progress import track
import json
import spacy
import pandas as pd

class InterSynthDatasetCB(data.Dataset):
    def __init__(
        self,
        data_root,
        scripts_path,
        split,
        mean,
        std,
        max_motion_length=196,
        min_motion_length=20,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        stage='lm_instruct',
        code_path='TOKENS',
        task_path=None,
        std_text=False,
        **kwargs,
    ):
        self.tiny = tiny
        self.unit_length = unit_length

        # Data mean and std
        self.mean = mean
        self.std = std
        self.max_length = 20
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.split = split
        
        # add original motion ids
        self.id_list = []
        for key in ['train', 'val', 'test']:
            split_file = pjoin(data_root, key + '.txt')
            with cs.open(split_file, "r") as f:
                for line in f.readlines():
                    self.id_list.append(line.strip())
  
        if code_path is None:
            # only for test interaction
            code_path = 'TOKENS'
        motion_code_dir = pjoin(data_root, code_path)
        motion_dir = pjoin(data_root, 'new_joint_vecs')
        text_dir = pjoin(data_root, 'texts')


        enumerator = enumerate(
            track(
                self.id_list,
                f"Loading HumanML3D Interaction! {split}",
            ))
        maxdata = 1e10
        subset = ''
        
        
        length_list = []
        new_name_list = []
        data_dict = {}
        
        # check tmp file exists, if not, load data
        tmp_data_dir = 'tmp_inter_hm3d_' + code_path
        if os.path.exists(pjoin(data_root, tmp_data_dir, f'{split}_{subset}_tokens_data.pkl')) \
            and os.path.exists(pjoin(data_root, tmp_data_dir, f'{split}_{subset}_tokens_index.pkl')):
            with open(pjoin(data_root, tmp_data_dir, f'{split}_{subset}_tokens_data.pkl'), 'rb') as file:
                data_dict = pickle.load(file)
            with open(pjoin(data_root, tmp_data_dir, f'{split}_{subset}_tokens_index.pkl'), 'rb') as file:
                new_name_list = pickle.load(file)
        else:
            # Fast loading
            for i, name in enumerator:
                if len(new_name_list) > maxdata:
                    break
                try:
                    # Load motion tokens
                    m_token_list = np.load(pjoin(motion_code_dir, f'{name}.npy'))
                    motion = np.load(pjoin(motion_dir, f'{name}.npy'))
                    # Read text
                    with cs.open(pjoin(text_dir, name + '.txt')) as f:
                        text_data = []
                        flag = False
                        lines = f.readlines()

                        for line_id, line in enumerate(lines):
                            try:
                                text_dict = {}
                                line_split = line.strip().split('#')
                                caption = line_split[0]
                                t_tokens = line_split[1].split(' ')
                                f_tag = float(line_split[2])
                                to_tag = float(line_split[3])
                                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                                text_dict['caption'] = caption
                                text_dict['tokens'] = t_tokens
                                if f_tag == 0.0 and to_tag == 0.0:
                                    flag = True
                                    text_data.append(text_dict)
                                else:
                                    m_token_list_new = [
                                        tokens[int(f_tag * fps / unit_length
                                                    ):int(to_tag * fps /
                                                            unit_length)]
                                        for tokens in m_token_list
                                        if int(f_tag * fps / unit_length) <
                                        int(to_tag * fps / unit_length)
                                    ]
                                    motion_new = motion[int(f_tag *
                                                        fps):int(to_tag * fps)]

                                    if len(m_token_list_new) == 0:
                                        continue
                                    # new_name = '%s_%f_%f' % (name, f_tag,
                                    #                             to_tag)
                                    new_name = name + '_' + str(line_id)

                                    data_dict[new_name] = {
                                        'm_token_list': m_token_list_new,
                                        'motion': motion_new,
                                        'length': len(motion_new),
                                        'text': [text_dict]
                                    }
                                    new_name_list.append(new_name)
                                    length_list.append(len(motion_new))
                            except:
                                pass

                    if flag:
                        if len(m_token_list) == 0:
                            continue
                        data_dict[name] = {
                            'm_token_list': m_token_list,
                            'motion': motion,
                            'length': len(motion),
                            'text': text_data
                        }
                        new_name_list.append(name)
                except:
                    pass

        if tmpFile and not os.path.exists(pjoin(data_root, tmp_data_dir, f'{split}_{subset}_tokens_data.pkl')):
            os.makedirs(pjoin(data_root, tmp_data_dir), exist_ok=True)
            with open(
                    pjoin(data_root, tmp_data_dir, f'{split}_{subset}_tokens_data.pkl'),
                    'wb') as file:
                pickle.dump(data_dict, file)
            with open(
                    pjoin(data_root, tmp_data_dir, f'{split}_{subset}_tokens_index.pkl'),
                    'wb') as file:
                pickle.dump(new_name_list, file)

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list
        self.nlp = spacy.load('en_core_web_sm')
        self.std_text = std_text
        
        
        # Data path
        # load csv file from scripts_path and get id
        df = pd.read_csv(scripts_path)
        # first row is title, then each row is an item
        item_id = df['Number']
        # read each line and build a dictionary, first row is the number, second row is the action descirption
        self.item_finals = []
        self.item_raw_all = []
        for i in range(len(item_id)):
            csv_id = df['Number'][i]
            a_motion_id = df['A Action File'][i]
            b_motion_id = df['B Action File'][i]
            a_text_seq_id = df['A Action ID'][i]
            b_text_seq_id = df['B Action ID'][i]
            
            if len(str(a_motion_id)) >= 6:
                a_query_data_index = str(a_motion_id)
            else:
                a_query_data_index = str(a_motion_id).zfill(6)
            
            if len(str(b_motion_id)) >= 6:
                b_query_data_index = str(b_motion_id)
            else:
                b_query_data_index = str(b_motion_id).zfill(6)
            
            
            item_dict = {}
            item_dict['csv_id'] = csv_id
            item_dict['a_motion_id'] = a_motion_id
            item_dict['a_motion_des'] = df['A Action'][i]
            item_dict['b_motion_id'] = b_motion_id
            item_dict['b_motion_des'] = df['B Action'][i]
            item_dict['a_speech'] = df['A Speech'][i]
            item_dict['b_speech'] = df['B Speech'][i]
            item_dict['a_emotion'] = df['Emotion of A'][i]
            item_dict['b_emotion'] = df['Emotion of B'][i]
            # item_dict['a_query'] = a_query_data_index
            # item_dict['b_query'] = b_query_data_index
            # self.item_raw_all.append(item_dict)
            
            item_valid = False
            
            part_a_index = a_query_data_index + '_' + str(a_text_seq_id)
            part_b_index = b_query_data_index + '_' + str(b_text_seq_id)
            
            if a_query_data_index not in self.data_dict.keys():
                a_query_data_index = part_a_index
            if b_query_data_index not in self.data_dict.keys():
                b_query_data_index = part_b_index
            
            if a_query_data_index in self.data_dict.keys() and b_query_data_index in self.data_dict.keys():
                item_valid = True
            
            item_dict['a_query'] = a_query_data_index
            item_dict['b_query'] = b_query_data_index
            self.item_raw_all.append(item_dict)
            if item_valid:
                self.item_finals.append(item_dict)
        
        if self.split == 'train':
            self.item_finals = self.item_finals[int(len(self.item_finals)*0.0):int(len(self.item_finals)*0.7)]
        elif self.split == 'val':
            self.item_finals = self.item_finals[int(len(self.item_finals)*0.7):int(len(self.item_finals)*0.85)]
        else:
            self.item_finals = self.item_finals[int(len(self.item_finals)*0.85):int(len(self.item_finals)* 1.)]
        
        instructions = pjoin(data_root, 'interaction_instructions.json')
        self.instructions = json.load(open(instructions, 'r'))
        self.tasks = []
        for task in self.instructions.keys():
            for subtask in self.instructions[task].keys():
                self.tasks.append(self.instructions[task][subtask])
        motion_tmp_path = motion_dir = pjoin(data_root, 'new_joint_vecs', '000021.npy')
        motion = np.load(motion_tmp_path)
        self.nfeats = motion.shape[1]



    def __len__(self):
        # return len(self.name_list) * len(self.tasks)
        # return 640
        # if self.split == 'train':
        #     return 32
        return len(self.item_finals) * len(self.tasks)

    def __getitem__(self, item):
        data_idx = item % len(self.item_finals)
        task_idx = item // len(self.item_finals)


        item = self.item_finals[data_idx]
        csv_id = item['csv_id']
        
        a_query_data_index = item['a_query']
        a_raw_data = self.data_dict[a_query_data_index]
        a_m_token_list, a_text_list = a_raw_data['m_token_list'], a_raw_data['text']
        a_motion, a_m_length = a_raw_data['motion'], a_raw_data['length']

        a_m_tokens = random.choice(a_m_token_list)
        a_text_data = random.choice(a_text_list)
        a_caption = a_text_data['caption']
        a_speech = item['a_speech']
        
        a_all_captions = [
            ' '.join([token.split('/')[0] for token in text_dic['tokens']])
            for text_dic in a_text_list]
        
        if self.split == 'val':
            if len(a_all_captions) > 3:
                a_all_captions = a_all_captions[:3]
            elif len(a_all_captions) == 2:
                a_all_captions = a_all_captions + a_all_captions[0:1]
            elif len(a_all_captions) == 1:
                a_all_captions = a_all_captions * 3


        b_query_data_index = item['b_query']
        b_raw_data = self.data_dict[b_query_data_index]
        b_m_token_list, b_text_list = b_raw_data['m_token_list'], b_raw_data['text']
        b_motion, b_m_length = b_raw_data['motion'], b_raw_data['length']
        
        b_m_tokens = random.choice(b_m_token_list)
        b_text_data = random.choice(b_text_list)
        b_caption = b_text_data['caption']
        b_speech = item['b_speech']
        
        b_all_captions = [
            ' '.join([token.split('/')[0] for token in text_dic['tokens']])
            for text_dic in b_text_list
        ]
        

        # Random crop
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            a_m_length = (a_m_length // self.unit_length - 1) * self.unit_length
            b_m_length = (b_m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            a_m_length = (a_m_length // self.unit_length) * self.unit_length
            b_m_length = (b_m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(b_motion) - b_m_length)
        b_motion = b_motion[idx:idx + b_m_length]
        a_motion = (a_motion - self.mean) / self.std
        b_motion = (b_motion - self.mean) / self.std

        coin = np.random.choice([False, False, True])

        if coin:
            # drop one token at the head or tail
            coin2 = np.random.choice([True, False])
            if coin2:
                a_m_tokens = a_m_tokens[:-1]
                b_m_tokens = b_m_tokens[:-1]
            else:
                a_m_tokens = a_m_tokens[1:]
                b_m_tokens = b_m_tokens[1:]
        
        
        a_m_tokens_len = a_m_tokens.shape[0]
        b_m_tokens_len = b_m_tokens.shape[0]

        tasks = self.tasks[task_idx]
        if self.split == 'train':
            return a_speech, a_m_tokens, a_m_tokens_len, None, None, None, None, a_all_captions, tasks, b_speech, b_m_tokens, b_m_tokens_len, b_all_captions, csv_id
        elif self.split == 'val':
            return a_speech, a_motion, a_m_length, None, None, None, None, a_all_captions, tasks, b_speech, b_motion, b_m_length, b_all_captions, csv_id
        elif self.split == 'test':
            return a_speech, a_motion, a_m_length, None, None, None, None, a_all_captions, tasks, b_speech, b_motion, b_m_length, b_all_captions, csv_id