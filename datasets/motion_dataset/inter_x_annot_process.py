import os
from openai import AzureOpenAI, OpenAI
import json
import argparse
import pickle
import logging
import time
from tqdm import tqdm


AZURE_OPENAI_API_KEY = '$YOUR_API_KEY'
AZURE_OPENAI_ENDPOINT = '$YOUR_ENDPOINT'
openai_api_key = '$YOUR_OPENAI_API_KEY' # todo
openai_base_url = 'https://api.openai.com/v1'


USE_AZURE = True
TOKENS_BLOG = {}

def ChatGPT_request_messages(messages, 
                             model="gpt-4o",
                             api_key='',
                             base_url='https://api.openai.com/v1',
                             temperature=1.,
                             presence_penalty=0.0,
                             client='azure',
                             tokens_blog={}, 
                             json_format=False,
                             **kwargs): 
  try: 
    if client == 'openai':
        client = OpenAI(api_key=api_key, base_url=base_url)
    elif client == 'azure':
        if api_key == '':
            api_key = AZURE_OPENAI_API_KEY
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        else:
            azure_endpoint = base_url
        client = AzureOpenAI(api_key=api_key, azure_endpoint=azure_endpoint, api_version="2024-05-01-preview")
    else:
        raise ValueError("Invalid client")
    response_format = {"type": "json_object" if json_format else "text"}
    if type(messages) is not list:
        messages = [{"role": "user", "content": messages}]
        completion = client.chat.completions.create(
        model=model, 
        messages=messages,
        response_format=response_format,
        presence_penalty=presence_penalty,
        temperature=temperature,
        )
    else:
        completion = client.chat.completions.create(
        model=model, 
        messages=messages,
        response_format=response_format,
        presence_penalty=presence_penalty,
        temperature=temperature,
        )
    
    if model not in tokens_blog.keys():
        tokens_blog[model] = {'completion': 0, 'prompt': 0}
    tokens_blog[model]['completion'] += completion.usage.completion_tokens
    tokens_blog[model]['prompt'] += completion.usage.prompt_tokens
    if json_format:
        return json.loads(completion.choices[0].message.content)
    else:
        return completion.choices[0].message.content
  except: 
    print ("ChatGPT RETURN ERROR")
    return False


def check_format(item):
    if type(item) is not dict:
        return False
    if 'actor' not in item or 'reactor' not in item:
        return False
    for key in ['actor', 'reactor']:
        if type(item[key]) is not str:
            return False
    return True
    

def annots_to_single(file_path, save_dir, order=0, repeat=5, logger=None):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
        
    actor = 'P1' if order == 1 else 'P2'
    reactor = 'P2' if order == 1 else 'P1'
    file_name = os.path.basename(file_path).split('.')[0]
    split_texts = {}
    split_texts['actor'] = actor
    split_texts['reactor'] = reactor
    split_texts['file_name'] = file_name
    split_texts['texts'] = []
    
    
    for line in lines:
        prompts = f"""
        Please split the overall action description of two people into individual action descriptions for each person.
        Among these two people, there is an actor and a reactor. 
        The actor is the one who actively performs actions, and the reactor is the one who reacts to the actor.
        Whether it is an actor depends on whether the person is the one who initiates the action.
        Now the overall action of the two person is:
        """
        
        prompts += f"{line}\n"
        
        prompts += """
        Based on the overall description above, please split the action descriptions of actor and reactor.
        In the action description of a single person, when describing another person, always use 'the other person', not 'actor', 'reactor', or 'second person', 'first person' etc.
        Attention: Format the output in a json dictionary.
        The dictionary should have a key "actor" for the actor and a key "reactor" for the reactor, with their respective single action descriptions as the values.
        """
        
        prompts += """
        Here are examples for reference:
        
        Example 1:
        ### The Overall Descriptions:
        The first person raises his/her right hand and waves it happily from side to side towards the second person. Then, the second person enthusiastically waves back with his/her right hand. After a brief moment, the second person lowers his/her hand, and the first person also puts his/her hand down shortly after.
        ### The output should be:
        {"actor": "Raises his/her right hand and waves it happily from side to side towards the second person and puts his/her hand down shortly after", "reactor": "Enthusiastically waves back with the right hand. After a brief moment, he/she lowers the hand"}
        
        Example 2:
        ### The Overall Descriptions:
        One person raises his/her left hand in front of his/her head, shakes it several times, and then puts it down, while the other person stays still.
        ### The output should be:
        {"actor": "Raises his/her left hand in front of his/her head, shakes it several times, and then puts it down", "reactor": "Stays still"}
        """
        
        
        flag = False
        for _ in range(repeat):
            try:
                messages = prompts
                response = ChatGPT_request_messages(messages, model="gpt-4o", client='azure', json_format=True, tokens_blog=TOKENS_BLOG)
                time.sleep(2)
                if check_format(response):
                    texts_tmp = {
                        'both': line,
                        'actor': response['actor'],
                        'reactor': response['reactor']
                    }
                    split_texts['texts'].append(texts_tmp)
                    flag = True
                    break
            except:
                pass
        if not flag:
            logger.info(f"GPT-4o Error: {file_path} Description: {line}")
    if len(split_texts['texts']) >= 1:
        with open(os.path.join(save_dir, f'{file_name}.json'), 'w', encoding='utf-8') as file:
            json.dump(split_texts, file, indent=4)
    else:
        logger.info(f"Error in : {file_path}")


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
    parser = argparse.ArgumentParser('Inter-X annotation process')
    parser.add_argument('--period', type=int, default=8)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    
    text_root_dir = "SOLAMI_data/Inter-X/texts"
    save_dir = "SOLAMI_data/SEA_processed/Inter-X/texts_processed"
    order_path = 'SOLAMI_data/Inter-X/annots/interaction_order.pkl'

    with open(order_path, 'rb') as f:
        orders = pickle.load(f)
        
    os.makedirs(save_dir, exist_ok=True)
    
    names = os.listdir(text_root_dir)
    names.sort()
    file_paths_all = [os.path.join(text_root_dir, name) for name in names]
    file_paths_tmp = file_paths_all[args.part::args.period]
    file_paths = []
    for file_path in file_paths_tmp:
        base_name = os.path.basename(file_path).split('.')[0]
        json_path = os.path.join(save_dir, f'{base_name}.json')
        if not os.path.exists(json_path):
            file_paths.append(file_path)
            
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    log_file_path = os.path.join(os.path.dirname(save_dir),
                                 'process_text_period{}_part{}.log'.format(args.period, args.part))
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
    
    
    if args.debug:
        file_paths = file_paths[:3]
    
    for file_path in tqdm(file_paths):
        annots_to_single(file_path, save_dir, order=orders[os.path.basename(file_path).split('.')[0]], logger=logger)
        
    cost_sum, costs = calculate_api_cost(TOKENS_BLOG)
    logger.info(f"Cost: {cost_sum}")
    logger.info(json.dumps(costs, indent=4))