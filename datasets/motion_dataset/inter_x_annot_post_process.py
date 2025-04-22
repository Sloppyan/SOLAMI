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
    if 'option' not in item:
        return False
    if item['option'] not in ['Y', 'N', 'R']:
        return False
    return True
    

def check_actor_reactor(file_path, save_dir, repeat=5, logger=None):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)    
    
    both_descriptions = [item['both'] for item in data['texts']]
    saved_data = data.copy()
    lens = len(both_descriptions)
    for idx, item in enumerate(data['texts']):
        saved_data['texts'][idx]['option'] = 'Y'
        saved_data['texts'][idx]['votes'] = {'Y': 0, 'R': 0, 'N': 0}
        saved_data['texts'][idx]['modified'] = False
        
        prompts = f"""Here are {lens} different descriptions of the same two-person interactive action:\n"""
        for line_idx, line in enumerate(both_descriptions):
            prompts += f"Desciption {line_idx+1}: {line}\n"
       
        prompts += f"""\nAmong them, a data annotator has split this two-person interactive action into two individual actions based on the {idx+1}-th description. 
        Actor: {item['actor']}
        Reactor: {item['reactor']}
        
        The actor is the person who initiates the action, and the reactor is the person who responds passively. 
        Is this split reasonable? 
        You now have three options:
        Option "Y" means it is reasonable, option "R" means the actions of the actor and reactor are annotated in reverse, and option "N" means the annotation is not reasonable.
        Please response in a json dict format. The dict only has one key: "option", and it only has three options: "Y", "N", "R".
        """
        
        
        sum_votes = 0
        for _ in range(repeat):
            try:
                messages = prompts
                response = ChatGPT_request_messages(messages, model="gpt-4o", client='azure', json_format=True, tokens_blog=TOKENS_BLOG)
                time.sleep(1)
                if check_format(response):
                    saved_data['texts'][idx]['votes'][response['option']] += 1
                    sum_votes += 1
            except:
                pass
        if sum_votes < repeat-1:
            logger.info(f"GPT-4o Error: {file_path} Description: {line}")
        else:
            votes = list(saved_data['texts'][idx]['votes'].values())
            if votes[0] >= votes[1] and votes[0] >= votes[2]:
                saved_data['texts'][idx]['modified'] = True
                saved_data['texts'][idx]['option'] = 'Y'
            elif votes[1] > votes[0] and votes[1] >= votes[2]:
                saved_data['texts'][idx]['modified'] = True
                saved_data['texts'][idx]['option'] = 'R'
                saved_data['texts'][idx]['actor'], saved_data['texts'][idx]['reactor'] = saved_data['texts'][idx]['reactor'], saved_data['texts'][idx]['actor']
            else:
                saved_data['texts'][idx]['modified'] = True
                saved_data['texts'][idx]['option'] = 'N'
    processed_items = [item for item in saved_data['texts'] if item['modified']]
    if len(processed_items) >= 1:
        file_name = os.path.basename(file_path).split('.')[0]
        with open(os.path.join(save_dir, f'{file_name}.json'), 'w', encoding='utf-8') as file:
            json.dump(saved_data, file, indent=4)
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
    parser = argparse.ArgumentParser('Inter-X annotation post process')
    parser.add_argument('--period', type=int, default=12)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    
    text_root_dir = "SOLAMI_data/SEA_processed/Inter-X/texts_processed"
    save_dir = "SOLAMI_data/SEA_processed/Inter-X/texts_post_processed"
    # order_path = 'SOLAMI_data/Inter-X/annots/interaction_order.pkl'
 
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
                                 'post_process_text_period{}_part{}.log'.format(args.period, args.part))
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
        check_actor_reactor(file_path, save_dir, logger=logger)
        
    cost_sum, costs = calculate_api_cost(TOKENS_BLOG)
    logger.info(f"Cost: {cost_sum}")
    logger.info(json.dumps(costs, indent=4))