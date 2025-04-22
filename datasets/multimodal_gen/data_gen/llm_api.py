from openai import OpenAI, AzureOpenAI
import os
import json

def ChatGPT_request_messages(messages, 
                             model="gpt-4o",
                             api_key='',
                             base_url='https://api.openai.com/v1',
                             temperature=1.,
                             presence_penalty=0.0,
                             client='openai',
                             tokens_blog={}, 
                             json_format=False,
                             **kwargs): 
    try: 
        if client == 'openai':
            client = OpenAI(api_key=api_key, base_url=base_url)
        elif client == 'azure':
            if api_key == '':
                api_key = os.environ["AZURE_OPENAI_API_KEY"]
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
            else:
                azure_endpoint = base_url
            client = AzureOpenAI(api_key=api_key, azure_endpoint=azure_endpoint, api_version="2024-05-01-preview")
        else:
            raise ValueError("Invalid client")
        # client = OpenAI(api_key=api_key, base_url=base_url)
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


def get_embedding(text, 
                  model="text-embedding-ada-002", 
                  api_key='',
                  base_url='https://api.openai.com/v1',
                  client='openai',
                  tokens_blog={}, 
                  **kwargs): 
    if client == 'openai':
        client = OpenAI(api_key=api_key, base_url=base_url)
    elif client == 'azure':
        if api_key == '':
            api_key = os.environ["AZURE_OPENAI_API_KEY"]
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
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
            'prompt': 2.5,
            'completion': 10.,
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