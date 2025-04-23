import os
import re
from typing import List, Union
import numpy as np
import math
import time
import heapq
import torch
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from transformers import AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import random
from typing import Optional
from .tools.token_emb import NewTokenEmb


class MLM(nn.Module):

    def __init__(
        self,
        model_path: str,
        model_type: str = "t5",
        stage: str = "lm_pretrain",
        new_token_type: str = "insert",
        motion_codebook_size: int = 512,
        framerate: float = 30.0,
        down_t: int = 4,
        predict_ratio: float = 0.2,
        inbetween_ratio: float = 0.25,
        max_length: int = 512,
        lora: bool = False,
        quota_ratio: float = 0.5,
        noise_density: float = 0.15,
        mean_noise_span_length: int = 3,
        motion_part='body',
        interleaved=False,
        **kwargs,
    ) -> None:

        super().__init__()

        # Parameters
        self.m_codebook_size = motion_codebook_size
        self.max_length = max_length
        self.framerate = framerate
        self.down_t = down_t
        self.predict_ratio = predict_ratio
        self.inbetween_ratio = inbetween_ratio
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.quota_ratio = quota_ratio
        self.stage = stage
        self.motion_part = motion_part
        self.interleaved = interleaved
        # Instantiate language model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)
        if model_type == "t5":
            self.language_model = T5ForConditionalGeneration.from_pretrained(
                model_path)
            self.lm_type = 'encdec'
        elif model_type == "gpt2":
            self.language_model = GPT2LMHeadModel.from_pretrained(model_path)
            self.lm_type = 'dec'
        else:
            raise ValueError("type must be either seq2seq or conditional")

        if self.lm_type == 'dec':
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add motion tokens
        if self.motion_part != 'body_hand_sep':
            self.tokenizer.add_tokens(
                [f'<body_id_{i}>' for i in range(self.m_codebook_size + 3)])
        else:
            self.tokenizer.add_tokens(
                [f'<body_id_{i}>' for i in range(self.m_codebook_size + 3)])
            self.tokenizer.add_tokens(
                [f'<hand_id_{i}>' for i in range(self.m_codebook_size + 3)])

        
        if self.interleaved or self.motion_part != 'body_hand_sep':
            self.tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<begin_body>', '<end_body>', ]}, replace_additional_special_tokens=False)
        else:
            self.tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<begin_body>', '<end_body>', '<begin_hand>', '<end_hand>']}, replace_additional_special_tokens=False)
        
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<User>', '<LLM>']}, replace_additional_special_tokens=False)
        # self.language_model.resize_token_embeddings(len(self.tokenizer))
        self.added_tokens_type = 'special'
        # print('Speech special tokens added.')
        
        
        
        if new_token_type == "insert":
            self.language_model.resize_token_embeddings(len(self.tokenizer))
        elif new_token_type == "mlp":
            shared = NewTokenEmb(self.language_model.shared,
                                 self.m_codebook_size + 3)
            # lm_head = NewTokenEmb(self.language_model.lm_head,
            #   self.m_codebook_size + 3)
            self.language_model.resize_token_embeddings(len(self.tokenizer))
            self.language_model.shared = shared
            # self.language_model.lm_head = lm_head

        # Lora
        if lora:
            from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
            from peft.utils.other import fsdp_auto_wrap_policy
            peft_config = LoraConfig(
                bias="none",
                task_type="CAUSAL_LM",
                #  inference_mode=False,
                r=8,
                lora_alpha=16,
                lora_dropout=0.05)
            self.language_model = get_peft_model(self.language_model,
                                                 peft_config)

    def add_speech_normal_tokens(self):
        self.tokenizer.add_tokens(['<begin_user_speech>', '<end_user_speech>', '<answer_token>', '<begin_agent_speech>', '<end_agent_speech>',])
        # self.tokenizer.add_special_tokens(
        #     {'additional_special_tokens': ['<speech_start>', '<speech_end>']})
        self.language_model.resize_token_embeddings(len(self.tokenizer))
        self.added_tokens_type = 'normal'
        print('Speech normal tokens added.')
    
    def add_none(self):
        self.added_tokens_type = ''
        
    def add_speech_special_tokens(self):
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<begin_user_speech>', '<end_user_speech>', '<answer_token>', '<begin_agent_speech>', '<end_agent_speech>',]})
        self.language_model.resize_token_embeddings(len(self.tokenizer))
        self.added_tokens_type = 'special'
        print('Speech special tokens added.')

    def forward(self, texts: List[str], motion_tokens: Tensor,
                lengths: List[int], tasks: dict, partner_motion: list, **kwargs):
        if self.lm_type == 'encdec':
            return self.forward_encdec(texts, motion_tokens, lengths, tasks, **kwargs)
        elif self.lm_type == 'dec':
            return self.forward_dec(texts, motion_tokens, lengths, tasks, partner_motion, **kwargs)
        else:
            raise NotImplementedError("Only conditional_multitask supported")

    def forward_encdec(
        self,
        texts: List[str],
        motion_tokens: Tensor,
        lengths: List[int],
        tasks: dict,
    ):

        # Tensor to string
        motion_strings = self.motion_token_to_string(motion_tokens, lengths)

        # Supervised or unsupervised
        # condition = random.choice(
        #     ['text', 'motion', 'supervised', 'supervised', 'supervised'])
        condition = random.choice(['supervised', 'supervised', 'supervised'])

        if condition == 'text':
            inputs = texts
            outputs = texts
        elif condition == 'motion':
            inputs = motion_strings
            outputs = motion_strings
        else:
            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts)

        # Tokenize
        source_encoding = self.tokenizer(inputs,
                                         padding='max_length',
                                         max_length=self.max_length,
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")

        source_attention_mask = source_encoding.attention_mask.to(
            motion_tokens.device)
        source_input_ids = source_encoding.input_ids.to(motion_tokens.device)

        if condition in ['text', 'motion']:
            batch_size, expandend_input_length = source_input_ids.shape
            mask_indices = np.asarray([
                self.random_spans_noise_mask(expandend_input_length)
                for i in range(batch_size)
            ])
            target_mask = ~mask_indices
            input_ids_sentinel = self.create_sentinel_ids(
                mask_indices.astype(np.int8))
            target_sentinel = self.create_sentinel_ids(
                target_mask.astype(np.int8))

            labels_input_ids = self.filter_input_ids(source_input_ids,
                                                     target_sentinel)
            source_input_ids = self.filter_input_ids(source_input_ids,
                                                     input_ids_sentinel)

        else:
            target_inputs = self.tokenizer(outputs,
                                           padding='max_length',
                                           max_length=self.max_length,
                                           truncation=True,
                                           return_attention_mask=True,
                                           add_special_tokens=True,
                                           return_tensors="pt")

            labels_input_ids = target_inputs.input_ids.to(motion_tokens.device)
            lables_attention_mask = target_inputs.attention_mask.to(
                motion_tokens.device)

        labels_input_ids[labels_input_ids == 0] = -100
        outputs = self.language_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask
            if condition == 'supervised' else None,
            labels=labels_input_ids,
            decoder_attention_mask=lables_attention_mask
            if condition == 'supervised' else None,
        )

        return outputs

    def forward_dec(
        self,
        texts: List[str],
        motion_tokens: Tensor,
        lengths: List[int],
        tasks: dict,
        partner_motion: list, 
        **kwargs,
    ):
        self.tokenizer.padding_side = "right"

        if len(kwargs) == 0:
            # Tensor to string
            motion_strings = self.motion_token_to_string_new(motion_tokens, lengths)

            partner_motion_strings = self.motion_token_to_string_new(partner_motion, [256] * len(partner_motion))    

            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts, partner_motion_strings)
            labels = []
            for i in range(len(inputs)):
                labels.append(inputs[i] + outputs[i] +
                            self.tokenizer.eos_token)

            # Tokenize
            inputs = self.tokenizer(labels,
                                    padding='max_length',
                                    max_length=self.max_length,
                                    truncation=True,
                                    return_attention_mask=True,
                                    return_tensors="pt")

            labels_input_ids = inputs.input_ids.to(motion_tokens.device)
            lables_attention_mask = inputs.attention_mask.to(motion_tokens.device)
            outputs = self.language_model(input_ids=labels_input_ids,
                                        attention_mask=lables_attention_mask,
                                        labels=inputs["input_ids"])
        else:
            a_motion_strings = self.motion_token_to_string(motion_tokens, lengths)
            b_motion_strings = self.motion_token_to_string(kwargs['b_motion'], kwargs['b_length'])
            
            inputs_str, outputs_str = self.template_fulfill_interaction(tasks, lengths, a_motion_strings, texts,
                                                                b_motion_strings, kwargs['b_speech'], kwargs['b_length'])
            
            special_token = '<answer_token>' if self.added_tokens_type else '\n'
            labels = []
            for i in range(len(inputs_str)):
                labels.append(inputs_str[i] + special_token + outputs_str[i] +
                            self.tokenizer.eos_token)

            # Tokenize
            inputs = self.tokenizer(labels,
                                    padding='max_length',
                                    max_length=self.max_length,
                                    truncation=True,
                                    return_attention_mask=True,
                                    return_tensors="pt")
            if self.only_answer:
                input_ids_ = inputs.input_ids.clone()
                labels_ids = inputs.input_ids.clone()
                answer_start_index = (labels_ids == self.tokenizer.convert_tokens_to_ids('<answer_token>')).nonzero(as_tuple=True)[1][0] + 1
                labels_ids[:, :answer_start_index] = -100 
                
                attention_mask = inputs.attention_mask.to(motion_tokens.device)
                labels_ids = labels_ids.to(motion_tokens.device)
                input_ids_ = input_ids_.to(motion_tokens.device)
                outputs = self.language_model(input_ids=input_ids_,
                        attention_mask=attention_mask,
                        labels=labels_ids)
            else:
                labels_input_ids = inputs.input_ids.to(motion_tokens.device)
                lables_attention_mask = inputs.attention_mask.to(motion_tokens.device)
                outputs = self.language_model(input_ids=labels_input_ids,
                                            attention_mask=lables_attention_mask,
                                            labels=inputs["input_ids"])

        return outputs

    def generate_direct(self,
                        texts: List[str],
                        max_length: int = 256,
                        num_beams: int = 1,
                        do_sample: bool = True,
                        bad_words_ids: List[int] = None):

        # Device
        self.device = self.language_model.device
        
        # Tokenize
        if self.lm_type == 'dec':
            texts = [text + '<LLM>' for text in texts]
            self.tokenizer.padding_side = 'left'

        source_encoding = self.tokenizer(texts,
                                         padding='max_length',
                                         max_length=self.max_length,
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")

        source_input_ids = source_encoding.input_ids.to(self.device)
        source_attention_mask = source_encoding.attention_mask.to(self.device)

        if self.lm_type == 'encdec':
            outputs = self.language_model.generate(
                source_input_ids,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=do_sample,
                bad_words_ids=bad_words_ids,
            )
        elif self.lm_type == 'dec':
            outputs = self.language_model.generate(
                input_ids=source_input_ids,
                attention_mask=source_attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=do_sample,
                max_new_tokens=max_length)
            # self.tokenizer.padding_side = 'left'

        outputs_string = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        
        outputs_tokens, cleaned_text = self.transfer_motion_text(outputs_string, skip_pad_token=True)

        return outputs_tokens, cleaned_text


    def generate_direct_inter(self,
                            texts: List[str],
                            max_length: int = 256,
                            num_beams: int = 1,
                            do_sample: bool = True,
                            bad_words_ids: List[int] = None):

        # Device
        self.device = self.language_model.device
        if self.right_padding:
            self.tokenizer.padding_side = 'right'
        else:
            self.tokenizer.padding_side = 'left'
        source_encoding = self.tokenizer(texts,
                                         padding='max_length',
                                         max_length=self.max_length,
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")

        source_input_ids = source_encoding.input_ids.to(self.device)
        source_attention_mask = source_encoding.attention_mask.to(self.device)

        outputs = self.language_model.generate(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=do_sample,
            max_new_tokens=max_length)
        
        
        outputs_string = self.tokenizer.batch_decode(outputs,
                                                    skip_special_tokens=False)
        motion, speech = self.transfer_motion_speech(outputs_string)

        # if torch.distributed.get_rank() == 0:
            
        alls_ = len(motion)
        no_motion = 0
        no_speech = 0
        for motion_ in motion:
            if len(motion_) == 1 and motion_[0] == 0:
                no_motion += 1
        for speech_ in speech:
            if speech_ == '':
                no_speech += 1
        print('No motion {}, no speech {} in /{} examples.'.format(no_motion, no_speech, alls_))

        return motion, speech

 
    def extract_LLM_part(self, text: str, N: int=-1):
        if N == -1:
            match = re.findall(r'<LLM>(.*)', text)
            if len(match) > 0:
                return match[-1].strip()
            else:
                return ''
        else:
            match = re.findall(r'<LLM>(.*?)<User>', text)
            if N < len(match):
                return match[N].strip()
            else:
                return ''



    def transfer_motion_text(self, output_strings: List[str], skip_pad_token=True):
        motion_tokens = []
        texts = []
        for i in range(len(output_strings)):
            output_string = output_strings[i]
            if skip_pad_token:
                output_string = output_string.replace(self.tokenizer.pad_token, '')
            
            output_string = self.extract_LLM_part(output_string, N=-1)
            
            text_string = re.sub(r'<body_id_(\d+)>', '', output_string)
            text_string = re.sub(r'<hand_id_(\d+)>', '', text_string)
            text_string = re.sub(r'<begin_body>', '', text_string)
            text_string = re.sub(r'<end_body>', '', text_string)
            text_string = re.sub(r'<begin_hand>', '', text_string)
            text_string = re.sub(r'<end_hand>', '', text_string)
            text_string = re.sub(r'<User>', '', text_string)
            text_string = re.sub(r'<LLM>', '', text_string)
            texts.append(text_string)
            
            if self.motion_part != 'body_hand_sep':
                # extract body_id_xxx from the string in special tokens '<begin_body>', '<end_body>'
                token_ids = self.get_motion_token_list(output_string, '<begin_body>', '<end_body>', r'<body_id_(\d+)>')
                motion_tokens.append(token_ids)
            
            elif self.interleaved:
                token_ids = self.get_motion_token_list(output_string, '<begin_body>', '<end_body>', r'<body_id_(\d+)><hand_id_(\d+)>')
                motion_tokens.append(token_ids)
            else:
                body_ids = self.get_motion_token_list(output_string, '<begin_body>', '<end_body>', r'<body_id_(\d+)>')
                hand_ids = self.get_motion_token_list(output_string, '<begin_hand>', '<end_hand>', r'<hand_id_(\d+)>')
                min_len = min(body_ids.shape[1], hand_ids.shape[1])
                body_ids = body_ids[..., :min_len]
                hand_ids = hand_ids[..., :min_len]
                motion_tokens.append(torch.cat([body_ids, hand_ids], axis=0))
        return motion_tokens, texts
    
    
    def get_motion_token_list(self, content, startStr, endStr, re_format):
        try:
            startIndex = content.index(startStr)
            if startIndex >= 0:
                startIndex += len(startStr)
            endIndex = content.index(endStr)
        except:
            if re_format == r'<body_id_(\d+)><hand_id_(\d+)>':
                return torch.tensor([[0], [0]], dtype=int).to(self.device)
            else:
                return torch.tensor([[0]], dtype=int).to(self.device)

        string = content[startIndex:endIndex]
        motion_ids = re.findall(re_format, string)
        # motion_ids = re.findall(r'<motion_id_(\d+)>', string)
        output_list = []
        if len(motion_ids) == 0:
            if re_format == r'<body_id_(\d+)><hand_id_(\d+)>':
                return torch.tensor([[0], [0]], dtype=int).to(self.device)
            else:
                return torch.tensor([[0]], dtype=int).to(self.device)
        else:
            if isinstance(motion_ids[0], tuple):
                for i in range(len(motion_ids[0])):
                    tmp_ids = []
                    for item in motion_ids:
                        tmp_ids += self.extract_number(item[i])
                    tmp_ids = [int(i) for i in tmp_ids]
                    tmd_id_ts = torch.tensor(tmp_ids, dtype=int).to(self.device)
                    output_list.append(tmd_id_ts)
                output = torch.stack(output_list)
                return output
            else:
                tmp_ids = []
                for item in motion_ids:
                    tmp_ids += self.extract_number(item)
                tmp_ids = [int(i) for i in tmp_ids]
                output = torch.tensor(tmp_ids, dtype=int).to(self.device).reshape(1, -1)
                return output
    
    
    
    def transfer_motion_speech(self, behavior_string: List[str]):
        motion_tokens = []
        speech_tokens = []
        special_token = '<answer_token>' if self.added_tokens_type else '\n'
        if self.added_tokens_type:
            speech_markers = ['<begin_agent_speech>', '<end_agent_speech>']
        else:
            speech_markers = [f'<motion_id_{self.m_codebook_size + 1}>', f'<motion_id_{self.m_codebook_size}>']
            
        for i in range(len(behavior_string)):
                
            token_list = self.get_motion_str(behavior_string[i], f'<motion_id_{self.m_codebook_size}>', f'<motion_id_{self.m_codebook_size + 1}>', start_marker=special_token)
            token_list_padded = torch.tensor(token_list,
                                             dtype=int).to(self.device)
            motion_tokens.append(token_list_padded)
            speech = self.get_speech_str(behavior_string[i], speech_markers[0], speech_markers[1], start_marker=special_token)
            # speech = self.get_second_speech_str(behavior_string[i])
            speech_tokens.append(speech.strip())
        return motion_tokens, speech_tokens
    
    
    def generate_conditional(self,
                             texts: Optional[List[str]] = None,
                             motion_tokens: Optional[Tensor] = None,
                             lengths: Optional[List[int]] = None,
                             task: str = "t2m",
                             with_len: bool = False,
                             stage: str = 'train',
                             tasks: dict = None):

        self.device = self.language_model.device

        if task in ["t2m", "m2m", "pred", "inbetween"]:

            if task == "t2m":
                assert texts is not None
                motion_strings = [''] * len(texts)
                if not with_len:
                    if tasks is None:
                        tasks = [{
                            'input':
                            ['Show me a motion that captures the essence of  Input: <Caption_Placeholder>'],
                            'output': ['']
                        }] * len(texts)

                    lengths = [0] * len(texts)
                else:
                    tasks = [{
                        'input': [
                            'Generate motion with <Frame_Placeholder> frames: <Caption_Placeholder>'
                        ],
                        'output': ['']
                    }] * len(texts)
                    
            elif task == "pred":
                assert motion_tokens is not None and lengths is not None
                texts = [''] * len(lengths)
                tasks = [{
                    'input': ['Predict motion: <Motion_Placeholder_s1>'],
                    'output': ['']
                }] * len(lengths)

                motion_strings_old = self.motion_token_to_string(
                    motion_tokens, lengths)
                motion_strings = []
                for i, length in enumerate(lengths):
                    split = length // 5
                    motion_strings.append(
                        '>'.join(motion_strings_old[i].split('>')[:split]) +
                        '>')

            elif task == "inbetween":
                assert motion_tokens is not None and lengths is not None
                texts = [''] * len(lengths)
                tasks = [{
                    'input': [
                        "Complete the masked motion: <Motion_Placeholder_Masked>"
                    ],
                    'output': ['']
                }] * len(lengths)
                motion_strings = self.motion_token_to_string(
                    motion_tokens, lengths)
            for task in tasks:
                task['class'] = 't2m'
            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts,
                                                    stage)

            outputs_tokens, cleaned_text = self.generate_direct(inputs,
                                                                max_length=128,
                                                                num_beams=1,
                                                                do_sample=True)

            return outputs_tokens

        elif task == "m2t":
            assert motion_tokens is not None and lengths is not None

            motion_strings = self.motion_token_to_string_new(
                motion_tokens, lengths)

            if not with_len:
                tasks = [{
                    'input': ['Generate text for <Motion_Placeholder>:'],
                    'output': ['']
                }] * len(lengths)
            else:
                tasks = [{
                    'input': [
                        'Generate text with <Frame_Placeholder> frames: <Motion_Placeholder>'
                    ],
                    'output': ['']
                }] * len(lengths)

            texts = [''] * len(lengths)
            for task in tasks:
                task['class'] = 't2m'
            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts)
            outputs_tokens, cleaned_text = self.generate_direct(
                inputs,
                max_length=256,
                num_beams=1,
                do_sample=False,
                # bad_words_ids=self.bad_words_ids
            )
            return cleaned_text
    
    
    def generate_inter(self,
                        a_speech: Optional[List[str]] = None,
                        a_m_tokens: Optional[Tensor] = None,
                        a_lengths: Optional[List[int]] = None,
                        tasks: dict = None,):

        self.device = self.language_model.device

        assert a_m_tokens is not None and a_lengths is not None
        motion_strings = self.motion_token_to_string(a_m_tokens, a_lengths)
        
        
        tmp_lengths = [0] * len(a_lengths)
        tmp_speech = texts = ['<motion_id_1>'] * len(a_lengths)
        tmp_motions = [''] * len(a_lengths)
        inputs_, outputs_ = self.template_fulfill_interaction(tasks, a_lengths, motion_strings, a_speech, tmp_motions, tmp_speech, tmp_lengths)
        
        special_token = '<answer_token>' if self.added_tokens_type else '\n'
        inputs = []
        for i in range(len(inputs_)):
            inputs.append(inputs_[i] + special_token)
        
        pred_m_tokens, pred_strings = self.generate_direct_inter(inputs,
                                                            max_length=256,
                                                            num_beams=1,
                                                            do_sample=True)

        return pred_m_tokens, pred_strings
        
    
    def motion_token_to_string_new(self, motion_token, lengths: List[int]):
        motion_string = []
        for i in range(len(motion_token)):
            if motion_token[i] is None:
                motion_string.append('')
                continue
            if self.motion_part != 'body_hand_sep':
                motion_i = motion_token[i].cpu() if motion_token[i].device.type == 'cuda' else motion_token[i]
                motion_list = motion_i[0].tolist()[:lengths[i]]
                motion_string.append(
                    ('<begin_body>' + ''.join([f'<body_id_{int(i)}>' for i in motion_list]) + '<end_body>'))
            elif self.interleaved:
                motion_i = motion_token[i].cpu() if motion_token[i].device.type == 'cuda' else motion_token[i]
                motion_list = motion_i[:, :lengths[i]].tolist()
                motion_string_tmp = '<begin_body>'
                for j in range(len(motion_list[0])):
                    motion_string_tmp += f'<body_id_{int(motion_list[0][j])}><hand_id_{int(motion_list[1][j])}>'
                motion_string_tmp += '<end_body>'
                motion_string.append(motion_string_tmp)
            else:
                motion_i = motion_token[i].cpu() if motion_token[i].device.type == 'cuda' else motion_token[i]
                motion_list = motion_i[:, :lengths[i]].tolist()
                body_list = motion_list[0]
                hand_list = motion_list[1]
                motion_string.append(
                    ('<begin_body>' + ''.join([f'<body_id_{int(i)}>' for i in body_list]) + '<end_body>' +
                    '<begin_hand>' + ''.join([f'<hand_id_{int(i)}>' for i in hand_list]) + '<end_hand>'))
            
        return motion_string
    

    def motion_token_to_string(self, motion_token: Tensor, lengths: List[int]):
        motion_string = []
        for i in range(len(motion_token)):
            motion_i = motion_token[i].cpu(
            ) if motion_token[i].device.type == 'cuda' else motion_token[i]
            motion_list = motion_i.tolist()[:lengths[i]]
            motion_string.append(
                (f'<motion_id_{self.m_codebook_size}>' +
                 ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
                 f'<motion_id_{self.m_codebook_size + 1}>'))
        return motion_string

    def motion_token_list_to_string(self, motion_token: Tensor):
        motion_string = []
        for i in range(len(motion_token)):
            motion_i = motion_token[i].cpu(
            ) if motion_token[i].device.type == 'cuda' else motion_token[i]
            motion_list = motion_i.tolist()
            motion_string.append(
                (f'<motion_id_{self.m_codebook_size}>' +
                 ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
                 f'<motion_id_{self.m_codebook_size + 1}>'))
        return motion_string

    def extract_number(self, s):
        match = re.search(r'\d+', s)
        if match:
            return [ match.group()]
        else:
            return []

    
    def motion_string_to_token(self, motion_string: List[str]):
        motion_tokens = []
        output_string = []
        #TODO motion to string check
        for i in range(len(motion_string)):
            string = self.get_middle_str(
                motion_string[i], f'<motion_id_{self.m_codebook_size}>',
                f'<motion_id_{self.m_codebook_size + 1}>')
            motion_ids = re.findall(r'<motion_id_(\d+)>', string)
            token_list = []
            for motion_id in motion_ids:
                token_list += self.extract_number(motion_id)
            token_list = [int(i) for i in token_list]
            if len(token_list) == 0:
                token_list = [0]
            token_list_padded = torch.tensor(token_list,
                                             dtype=int).to(self.device)
            motion_tokens.append(token_list_padded)
            output_string.append(motion_string[i].replace(
                string, '<Motion_Placeholder>'))

        return motion_tokens, output_string

    def get_middle_str(self, content, startStr, endStr, name='body'):
        try:
            startIndex = content.index(startStr)
            if startIndex >= 0:
                startIndex += len(startStr)
            endIndex = content.index(endStr)
        except:
            return f'<{name}_id_0>'

        return content[startIndex:endIndex]


    def placeholder_fulfill(self, prompt: str, length: int, motion_string: str,
                            text: str, is_input=False):

        if random.random() < self.quota_ratio:
            text = f'\"{text}\"'

        if type(text) is float:
            text = ' '

        prompt = prompt.replace('<Caption_Placeholder>', text).replace('<Motion_Placeholder>', motion_string)

        if is_input:
            prompt = '<User>' + prompt
        else:
            prompt = '<LLM>' + prompt
        
        return prompt
     
    def placeholder_fulfill_interaction(self, prompt: str, length: int, motion_string: str,
                            text: str, is_input=False):

        seconds = math.floor(length / self.framerate)
        motion_splited = motion_string.split('>')
        token_length = length / self.down_t
        predict_head = int(token_length * self.predict_ratio + 1)
        masked_head = int(token_length * self.inbetween_ratio + 1)
        masked_tail = int(token_length * (1 - self.inbetween_ratio) + 1)
        
        motion_predict_head = '>'.join(
            motion_splited[:predict_head]
        ) + f'><motion_id_{self.m_codebook_size+1}>'
        motion_predict_last = f'<motion_id_{self.m_codebook_size}>' + '>'.join(
            motion_splited[predict_head:])

        motion_masked = '>'.join(
            motion_splited[:masked_head]
        ) + '>' + f'<motion_id_{self.m_codebook_size+2}>' * (
            masked_tail - masked_head) + '>'.join(motion_splited[masked_tail:])

        if random.random() < self.quota_ratio:
            text = f'{text}'

        if type(text) is float:
            text = ' '
        
        if self.added_tokens_type:
            if is_input:
                text = '<begin_user_speech>' + text + '<end_user_speech>'
            else:
                text = '<begin_agent_speech>' + text + '<end_agent_speech>'
                
        prompt = prompt.replace('<Speech_Placeholder>', text).replace('<Motion_Placeholder>', motion_string)

        return prompt
    
    def template_fulfill(self,
                         tasks,
                         lengths,
                         motion_strings,
                         texts,
                         partner_motion_strings=None,
                         stage='test'):
        inputs = []
        outputs = []
        for i in range(len(lengths)):
            input_template = random.choice(tasks[i]['input'])
            output_template = random.choice(tasks[i]['output'])
            length = lengths[i]
            inputs.append(
                self.placeholder_fulfill(input_template, length,
                                         motion_strings[i], texts[i], is_input=True))
            if tasks[i]['class'] == 'interactive':
                outputs.append(
                    self.placeholder_fulfill(output_template, length,
                                         partner_motion_strings[i], texts[i], is_input=False))
            outputs.append(
                self.placeholder_fulfill(output_template, length,
                                         motion_strings[i], texts[i], is_input=False))

        return inputs, outputs
    
    def template_fulfill_interaction(self,
                         tasks,
                         a_lengths,
                         a_motion_strings,
                         a_speech,
                         b_motion_strings,
                         b_speech,
                         b_lengths,
                         stage='test'):
        inputs = []
        outputs = []
        for i in range(len(a_lengths)):
            input_template = random.choice(tasks[i]['input'])
            output_template = random.choice(tasks[i]['output'])
            inputs.append(
                self.placeholder_fulfill_interaction(input_template, a_lengths[i],
                                         a_motion_strings[i], a_speech[i], is_input=True))
            outputs.append(
                self.placeholder_fulfill_interaction(output_template, b_lengths[i],
                                         b_motion_strings[i], b_speech[i], is_input=False))

        return inputs, outputs
      
    
    def get_motion_str(self, content, startStr, endStr, start_marker=''):
        try:
            if start_marker:
                start_index_ = content.index(start_marker)
            else:
                start_index_ = 0
            start_index = content.index(startStr, start_index_) + len(startStr)
            end_index =  content.index(endStr, start_index)
            motion_string = f'<motion_id_{self.m_codebook_size}>' + content[start_index:end_index] + f'<motion_id_{self.m_codebook_size+1}>'
            motion_ids = re.findall(r'<motion_id_(\d+)>', motion_string)
            token_list = []
            for motion_id in motion_ids:
                token_list += self.extract_number(motion_id)
            token_list = [int(i) for i in token_list]
            if len(token_list) == 0:
                token_list = [0]
        except:
            return [0]

        return token_list
    
    def get_speech_str(self, content, startStr: str, endStr: str, start_marker='\n'):
        try:
            if start_marker == '\n':
                start_index_ = content.index(start_marker)
                start_index = content.index(startStr, start_index_) + len(startStr)
                end_index =  content.index(endStr, start_index)
                return content[start_index:end_index]
            else:
                min_str = ''
                start_index_ = 0
                while True:
                    start_index_ = content.find(startStr, start_index_)
                    if start_index_ == -1:
                        break
                    
                    end_index_ = content.find(endStr, start_index_ + len(startStr))
                    if end_index_ == -1:
                        break
                        
                    curr_substr = content[start_index_ + len(startStr):end_index_]
                    if len(curr_substr) < len(min_str) or min_str == '':
                        min_str = curr_substr
                    
                    start_index_ += len(startStr)
                        
                return min_str
        except:
            return ''

        

    def random_spans_noise_mask(self, length):
        # From https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(
            np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens,
                                                  num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens,
                                                     num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length, ), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

    def create_sentinel_ids(self, mask_indices):
        # From https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py
        start_indices = mask_indices - np.roll(mask_indices, 1,
                                               axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0,
                                np.cumsum(start_indices, axis=-1),
                                start_indices)
        sentinel_ids = np.where(sentinel_ids != 0,
                                (len(self.tokenizer) - sentinel_ids - (self.m_codebook_size + 3)), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        # From https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids,
                                  input_ids.to('cpu'))

        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape(
            (batch_size, -1))
        input_ids = np.concatenate(
            [
                input_ids,
                np.full((batch_size, 1),
                        self.tokenizer.eos_token_id,
                        dtype=np.int32),
            ],
            axis=-1,
        )

        input_ids = torch.tensor(input_ids, device=self.device)

        return input_ids
