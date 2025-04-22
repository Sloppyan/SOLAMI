from llm_api import ChatGPT_request_messages
from utils import compare_types

class Behavior:
    def __init__(self, role, motion, expression, speech, action=None, round_id=None) -> None:
        self.role = role
        self.motion = motion
        self.expression = expression
        self.speech = speech
        self.action = action
        self.round_id = round_id
    
    def __str__(self) -> str:
        return f"<round>{self.round_id}<role>{self.role}<motion>{self.motion}<speech>{self.speech}<expression>{self.expression}"
    
    def get_save_dicts(self):
        return {'role': self.role, 
                'motion': self.motion, 
                'expression': self.expression, 
                'speech': self.speech, 
                'action_index': self.action['motion_name'], 
                'action_dataset': self.action['dataset'],
                'round_id': self.round_id,
                'action_des': self.action['text']}


def check_format_behavior(gpt_response):
    example = {'round': '0', 'role': 'A','speech': 'How are you?', 'expression': 'smile', 'motion': 'wave hands'}
    behavior_dict = {}
    gpt_response = gpt_response.strip()
    for item in gpt_response.split('<'):
        if item == '':
            continue
        if '>' in item and len(item) >= 2:
            key, value = item.split('>')
            if key in example.keys():
                behavior_dict[key] = value
    if compare_types(behavior_dict, example):
        return True
    else:
        return False



def check_inter_response(gpt_response):
    if type(gpt_response) != dict:
        return False
    if 'inter' not in gpt_response or 'behavior' not in gpt_response:
        return False
    if gpt_response['inter'] not in ['Y', 'N']:
        return False
    
    gpt_response = gpt_response['behavior']
    if check_format_behavior(gpt_response):
        return True
    else:
        return False


def response_reformat(response, example, tokens_blog={}, **kwargs):
    prompt = f"""Modify the format of the input string according to the format of the sample.
    Note that the input content cannot be changed, but the format of the input must be consistent with that of the sample.
    
    Here are some examples:
    ### Example 1:
    If the sample format is:
    <round>PPPP<role>QQQQ<motion>XXXX<speech>YYYY<expression>ZZZZ
    It means that PPPP is the round number, QQQQ is the role name, XXXX is the motion description, YYYY is the speech, and ZZZZ is the expression.
    
    If the input is:
    <round>0<role>A<Motion>Wave hands<speech>Hello<expression>Smile <speech>How are you?
    
    Then the output should be:
    <round>0<role>A<Motion>Wave hands<speech>Hello<expression>Smile
    
    Because <Motion> should be <motion>, and there is an extra <speech>How are you? at the end. 
    
    ### Example 2:
    If the sample format is:
    {{'index': 1, 'behavior': '<round>PPPP<role>QQQQ<motion>XXXX<speech>YYYY<expression>ZZZZ'}}
    It means that PPPP is the round number, QQQQ is the role name, XXXX is the motion description, YYYY is the speech, and ZZZZ is the expression.
    The motion and speech are generated based on the first (index 1) provided motion in the previous prompt.
    
    If the input is:
    2<round>0<role>A<Motion>Wave hands<speech>Hello<expression>Smile <speech>How are you?
    
    Then the output should be:
    {{'index': 2, 'behavior': '<round>0<role>A<Motion>Wave hands<speech>Hello<expression>Smile <speech>How are you?'}}
    
    Because the first row should be only an index number, and the motion and speech are generated based on the first (index 1) provided motion in the previous prompt.
    """
    
    prompt += f"\n Now the input string is: \n{response}\n"
    prompt += f"\nOutput the result directly, without any superfluous content.So the output should be \n"
    curr_gpt_response = ChatGPT_request_messages(prompt, tokens_blog=tokens_blog, **kwargs)
    return curr_gpt_response

        
def transfer_response_to_behavior(response, tokens_blog={}, **kwargs):
    check_count = 0
    while not check_inter_response(response):
        print("Reformatting response: ", response)
        response = response_reformat(response, "{'inter': 'N', 'behavior': '<round>PPPP<role>QQQQ<motion>XXXX<speech>YYYY<expression>ZZZZ'}", tokens_blog=tokens_blog, **kwargs)
        print("Reformatted response: ", response)
        check_count += 1
        if check_count > 3:
            return Exception("Reformatting response failed")
    inter = response['inter']
    behavior_dict = {}
    response = response['behavior'].strip()
    for item in response.split('<'):
        if item == '':
            continue
        if '>' in item and len(item) >= 2:
            key, value = item.split('>')
            behavior_dict[key] = value
    return inter, Behavior(behavior_dict['role'], behavior_dict['motion'], behavior_dict['expression'], behavior_dict['speech'], round_id=behavior_dict['round'])


def check_format_regenerated_behavior(gpt_response, top_k=4):
    if type(gpt_response) != dict:
        return False
    if 'index' not in gpt_response or 'behavior' not in gpt_response:
        return False
    if gpt_response['index'] not in [i+1 for i in range(top_k)]:
        return False
    if not check_format_behavior(gpt_response['behavior']):
        return False
    return True


def transfer_response_to_regenerated_behavior(response, top_k=4, tokens_blog={}, **kwargs):
    check_count = 0
    while not check_format_regenerated_behavior(response, top_k=top_k):
        print("Reformatting response: ", response)
        response = response_reformat(response, "{'index': 1, 'behavior': '<round>PPPP<role>QQQQ<motion>XXXX<speech>YYYY<expression>ZZZZ'}", tokens_blog=tokens_blog, **kwargs)
        print("Reformatted response: ", response)
        check_count += 1
        if check_count > 3:
            return Exception("Reformatting response failed")
    # response = response.strip()
    # response_split = response.split('\n')
    index  = int(response['index'])
    behavior_dict = {}
    for item in response['behavior'].split('<'):
        if item == '':
            continue
        if '>' in item and len(item) >= 2:
            key, value = item.split('>')
            behavior_dict[key] = value
    behavior = Behavior(behavior_dict['role'], behavior_dict['motion'], behavior_dict['expression'], behavior_dict['speech'], round_id=behavior_dict['round'])
    return index, behavior


def transfer_script_to_behavior(response, tokens_blog={}, **kwargs):
    check_count = 0
    while not check_format_behavior(response):
        print("Reformatting response: ", response)
        response = response_reformat(response, "<round>PPPP<role>QQQQ<motion>XXXX<speech>YYYY<expression>ZZZZ", tokens_blog=tokens_blog, **kwargs)
        print("Reformatted response: ", response)
        check_count += 1
        if check_count > 3:
            return Exception("Reformatting response failed")
    behavior_dict = {}
    response = response.strip()
    for item in response.split('<'):
        if item == '':
            continue
        if '>' in item and len(item) >= 2:
            key, value = item.split('>')
            behavior_dict[key] = value
    return Behavior(behavior_dict['role'], behavior_dict['motion'], behavior_dict['expression'], behavior_dict['speech'], round_id=behavior_dict['round'])