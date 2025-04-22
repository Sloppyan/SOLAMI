"""
Step 1: load the topic

Step 2: filter the topic

Step 3: save the topic

topic sources: google trends, zhihu, jike
"""
import os
import re
import sys
import pandas as pd
import json
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from data_gen.llm_api import ChatGPT_request_messages, calculate_api_cost

data_dirs = {
    'google_trends': r'data\topics\google_trends',
    'zhihu': r'data\topics\zhihu',
    'jike': r'data\topics\jike',
}
TOKENS_BLOG = {}
SAVE_EVERY_STEPS = 5

if not os.path.exists(r'data\topics\origin_topics.csv'):
    data_items = []

    ### define unified data format
    # data source, category, topic, traffic, sub_topics, date

    # data_item = {
    #     'data_source': 'google_trends',
    #     'category': 'general',
    #     'topic': 'coronavirus',
    #     'traffic': '100',
    #     'sub_topics': 'covid-19; virus; pandemic',
    #     'date': '2021-01-01'
    # }

    ### load google trends data
    if 'google_trends' in data_dirs:
        data_paths = os.listdir(data_dirs['google_trends'])
        for data_path in data_paths:
            if data_path.endswith('.csv'):
                data = pd.read_csv(os.path.join(data_dirs['google_trends'], data_path))
                for index, row in data.iterrows():
                    data_item = {
                        'data_source': 'google_trends',
                        'category': '',
                        'topic': row['topic'],
                        'traffic': row['traffic'],
                        'sub_topics': row['sub_topics'],
                        'date': row['date']
                    }
                    data_items.append(data_item)

    ### load zhihu data
    def extract_topics(line):
        match = re.search(r'^\d+\.(.+)$', line)
        if match:
            return match.group(1)
        return None

    if 'zhihu' in data_dirs:
        data_paths = os.listdir(data_dirs['zhihu'])
        for data_path in data_paths:
            if data_path.endswith('.txt'):
                data = pd.read_csv(os.path.join(data_dirs['zhihu'], data_path), header=None, names=['topic'])
                data['topic'] = data['topic'].apply(extract_topics)
                for index, row in data.iterrows():
                    if row['topic'] is None:
                        continue
                    data_item = {
                        'data_source': 'zhihu',
                        'category': data_path.split('.')[0],
                        'topic': row['topic'],
                        'traffic': '',
                        'sub_topics': '',
                        'date': '2024-06-19'
                    }
                    data_items.append(data_item)

    ### load jike data
    if 'jike' in data_dirs:
        data_paths = os.listdir(data_dirs['jike'])
        for data_path in data_paths:
            if data_path.endswith('.txt'):
                data = pd.read_csv(os.path.join(data_dirs['jike'], data_path), header=None, names=['Topic'])
                for index, row in data.iterrows():
                    data_item = {
                        'data_source': 'jike',
                        'category': '',
                        'topic': row['Topic'],
                        'traffic': '',
                        'sub_topics': '',
                        'date': '2024-06-19'
                    }
                    data_items.append(data_item)

    ### save origin data
    df = pd.DataFrame(data_items)
    df.to_csv(r'data\topics\origin_topics.csv', index=False)
else:
    df = pd.read_csv(r'data\topics\origin_topics.csv')
    data_items = df.to_dict(orient='records')

### filter the topic

filter_data_path = r'data\topics\filter_topics.csv'


def filter_google_trends(data_item, filtered_data_items, database_id):
    prompt = """
    Now there are two characters A and B in social interaction in a 3D role-playing AI application.
    Character A is a normal person of a 3D role-playing AI application.
    Character B is the 3D virtual companion of character A. Character B has all the capabilities of a normal AI assistant. 
    In addition, it can understand the human\'s body language, interact with human in real time, and perform sports, dance, and other skills with its body.
    Later I will provide you a possible topic from the Google Trends for the social coversation between them A and B.
    Based on the settings above, you need to judge whether the topic is appropriate and rewrite the topic in a bit more detail in English, about 10-15 words.
    
    Your response must be in a json dict format.
    The key 'option' means whether the topic is appropriate, and you have two options: 'yes' or 'no'.
    Another key 'detailed topic' is the detailed translation of the topic.
    
    Now the topic is: [{topic}], similar words to the topic are:[{sub_topic}]. 
    
    Here are some examples:
    ###
    Example1:
    If the input is:
    Topic: NVIDIA Stock
    Similar words:
    Then your response is like this:
    {{"option": "yes", "detailed topic": "The stock price of NVIDIA is rising."}}
    Example2:
    If the input is:
    Topic: tornado
    Similar words: Tornadoes on the East Coast
    Then your response is like this:
    {{"option": "no", "detailed topic": "The tornadoes on the East Coast are too dangerous."}}
    ###
    Be sure to output the json dict in the correct format.
    """
    prompts = prompt.format(topic=data_item['topic'], sub_topic=data_item['sub_topics'])
    
    for _ in range(5): 
        try:            
            curr_gpt_response_json = ChatGPT_request_messages(prompts, client='azure', json_format=True, tokens_blog=TOKENS_BLOG)
            curr_gpt_response = json.loads(curr_gpt_response_json)
            if curr_gpt_response['option'] in ['yes', 'no'] and 'detailed topic' in curr_gpt_response:
                if curr_gpt_response['option'] == 'yes':
                    data_item_modifed = data_item.copy()
                    data_item_modifed['modified_topic'] = curr_gpt_response['detailed topic']
                    data_item_modifed['database_id'] = database_id
                    filtered_data_items.append(data_item_modifed)            
                break
        except:
            pass
    return filtered_data_items


def filter_zhihu(data_item, filtered_data_items, database_id):
    prompt = """
    Now there are two characters A and B in social interaction in a 3D role-playing AI application.
    Character A is a normal person of a 3D role-playing AI application.
    Character B is the 3D virtual companion of character A. Character B has all the capabilities of a normal AI assistant. 
    In addition, it can understand the human\'s body language, interact with human in real time, and perform sports, dance, and other skills with its body.
    Later I will provide you a possible topic in Chinese for the social coversation between them A and B.
    Based on the settings above, you need to judge whether the topic is appropriate and rewrite the topic in a bit more detail in English, about 10-15 words.
    
    Your response must be in a json dict format.
    The key 'option' means whether the topic is appropriate, and you have two options: 'yes' or 'no'.
    Another key 'detailed topic' is the detailed translation of the topic.
    
    Now the topic is: [{topic}], the category of the topic is:[{category}]. 
    
    Here are some examples:
    ###
    Example1:
    If the input is:
    Topic: 为什么大量人类会喜欢猫？
    Category: 宠物
    Then your response is like this:
    {{
        "option": "yes",
        "detailed topic": "Why do so many people like cats?"
    }}
    Example2:
    If the input is:
    Topic: 为什么总叫央视六套六公主？
    Category: 数码
    Then your response is like this:
    {{
        "option": "no",
        "detailed topic": "Why do we always call CCTV 6 sets 6 princess?"
    }}
    Example3:
    If the input is:
    Topic: 2024年的欧洲杯冠军会是谁?
    Category: 体育
    Then your response is like this:
    {{
        "option": "yes",
        "detailed topic": "Predicting the champions of Euro 2024"
    }}
    ###
    Be sure to output the json dict in the correct format.
    """
    prompts = prompt.format(topic=data_item['topic'], category=data_item['category'])
    
    for _ in range(3): 
        try:            
            curr_gpt_response_json = ChatGPT_request_messages(prompts, client='azure', json_format=True, tokens_blog=TOKENS_BLOG)
            curr_gpt_response = json.loads(curr_gpt_response_json)
            if curr_gpt_response['option'] in ['yes', 'no'] and 'detailed topic' in curr_gpt_response:
                if curr_gpt_response['option'] == 'yes':
                    data_item_modifed = data_item.copy()
                    data_item_modifed['modified_topic'] = curr_gpt_response['detailed topic']
                    data_item_modifed['database_id'] = database_id
                    filtered_data_items.append(data_item_modifed)            
                break
        except:
            pass
    return filtered_data_items



def filter_jike(data_item, filtered_data_items, database_id):
    prompt = """
    Now there are two characters A and B in social interaction in a 3D role-playing AI application.
    Character A is a normal person of a 3D role-playing AI application.
    Character B is the 3D virtual companion of character A. Character B has all the capabilities of a normal AI assistant. 
    In addition, it can understand the human\'s body language, interact with human in real time, and perform sports, dance, and other skills with its body.
    Later I will provide you a possible topic in Chinese for the social coversation between them A and B.
    Based on the settings above, you need to judge whether the topic is appropriate and rewrite the topic in a bit more detail in English, about 10-15 words.
    
    Your response must be in a json dict format.
    The key 'option' means whether the topic is appropriate, and you have two options: 'yes' or 'no'.
    Another key 'detailed topic' is the detailed translation of the topic.
    
    Now the topic is: [{topic}]. 
    
    Here are some examples:
    ###
    Example1:
    If the input is:
    Topic: 你看过最难忘的一部电影是?
    Then your response is like this:
    {{
        "option": "yes",
        "detailed topic": "One of the most memorable movies."
    }}
    Example2:
    If the input is:
    Topic: 最近读了什么书?
    Then your response is like this:
    {{
        "option": "yes",
        "detailed topic": "The book you read recently."
    }}
    ###
    Be sure to output the json dict in the correct format.
    """
    prompts = prompt.format(topic=data_item['topic'])
    
    for _ in range(3): 
        try:            
            curr_gpt_response_json = ChatGPT_request_messages(prompts, client='azure', json_format=True, tokens_blog=TOKENS_BLOG)
            curr_gpt_response = json.loads(curr_gpt_response_json)
            if curr_gpt_response['option'] in ['yes', 'no'] and 'detailed topic' in curr_gpt_response:
                if curr_gpt_response['option'] == 'yes':
                    data_item_modifed = data_item.copy()
                    data_item_modifed['modified_topic'] = curr_gpt_response['detailed topic']
                    data_item_modifed['database_id'] = database_id
                    filtered_data_items.append(data_item_modifed)            
                break
        except:
            pass
    return filtered_data_items


def write_data(filter_data_path, current_data_items):
    df_output = pd.DataFrame(current_data_items)
    df_output.to_csv(filter_data_path, index=False)



if os.path.exists(filter_data_path):
    df = pd.read_csv(filter_data_path)
    filtered_data_items = df.to_dict(orient='records')
    curr_id = filtered_data_items[-1]['database_id'] + 1
else:
    filtered_data_items = []
    curr_id = 0


google_trends_max = 2000
zhihu_max = 2000
jike_max = 2000
google_trends_count = 0
zhihu_count = 0
jike_count = 0

lens = len(data_items[curr_id:])

for id in tqdm(range(lens), total=lens):
    data_item = data_items[id + curr_id]
    if data_item['data_source'] == 'google_trends':
        if google_trends_count >= google_trends_max:
            continue
        filtered_data_items = filter_google_trends(data_item, filtered_data_items, id + curr_id)
        google_trends_count += 1
        if filtered_data_items and len(filtered_data_items) % SAVE_EVERY_STEPS == 0:
            write_data(filter_data_path, filtered_data_items)
    elif data_item['data_source'] == 'zhihu':
        if zhihu_count >= zhihu_max:
            continue
        filtered_data_items = filter_zhihu(data_item, filtered_data_items, id + curr_id)
        zhihu_count += 1
        if filtered_data_items and len(filtered_data_items) % SAVE_EVERY_STEPS == 0:
            write_data(filter_data_path, filtered_data_items)
    elif data_item['data_source'] == 'jike':
        if jike_count >= jike_max:
            continue
        filtered_data_items = filter_jike(data_item, filtered_data_items, id + curr_id)
        jike_count += 1
        if filtered_data_items and len(filtered_data_items) % SAVE_EVERY_STEPS == 0:
            write_data(filter_data_path, filtered_data_items)

write_data(filter_data_path, filtered_data_items)

print("Filtering topics completed.")
print(f"Filtered topics saved to {filter_data_path}")
print(f"Total topics filtered: {len(filtered_data_items)}")

cost_all, costs = calculate_api_cost(TOKENS_BLOG)
print('Total cost ($): ', cost_all)
print('Detail costs ($): ', costs)