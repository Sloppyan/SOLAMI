import numpy as np
import logging
import re
import ast
import datetime
from llm_api import ChatGPT_request_messages, get_embedding
from behavior import transfer_response_to_behavior, transfer_response_to_regenerated_behavior, transfer_script_to_behavior
from utils import func_validate, compare_types

def retrieve_motion(text_to_actions_dicts={}):
    actions = []
    # weights = []
    # for dataset_name in text_to_actions_dicts.keys():
    #     for action_list in text_to_actions_dicts[dataset_name].values():
    #         actions.extend(action_list)
    #         weights.extend([dataset_item_weights[dataset_name]] * len(action_list))
    ## sample action 
    for action_list in text_to_actions_dicts.values():
            actions.extend(action_list)
    action = np.random.choice(actions)
    return action


def generate_topic(prompts, motion=None, tokens_blog={}, llm_settings={}, logger=None):
    prompt_generate_topic = prompts['background'] + prompts['user_settings'] + prompts['agent_settings']
    prompt_generate_topic += prompts['space_limitation'] + prompts['locomotion_limitation']
    if motion is not None:
        texts_len = len(motion['text'])
        text_index = np.random.randint(texts_len)
        motion['current_text'] = motion['text'][text_index]
        prompt_generate_topic += f"""Now one of the characters are acting out the motion:  + {motion['current_text']}."""
    prompt_generate_topic += """Please generate a specific topic to start the conversation.
            The topic should be specific, informative, and plausible based on the provided settings above.
            It should be a python dict string with the key 'topic' and the value as the topic description in json format.

            Here are some exemplary topics as outputs in json format:
           
            Example 1: {"topic": "How to celebrate A's birthday"} 

            Example 2: {"topic": "Character A's exhausting job and his crazy boss"}
            
            Example 3: {"topic": "Basketball star Kobe Bryant"}
            
            Example 4: {"topic": "The new movie 'The Matrix Resurrections'"}
            
            Example 5: {"topic": "Dancing skills for a coming party"}
            
            Be sure to output one topic according to the json format of the examples.
            """
    
    for i in range(llm_settings['repeat_times']):
        try:
            current_settings = llm_settings.copy()
            current_settings['json_format'] = True
            response = ChatGPT_request_messages(prompt_generate_topic, tokens_blog=tokens_blog, **current_settings)
            logging.info("Trial {}: output of the proposed topic: \n {}".format(i, response))
            if compare_types(response, {'topic': 'test'}):
                des_topic = response['topic']
                break
        except:
            pass
    if des_topic == '':
        logger.info("Failed to generate topic!")
    return des_topic



def generate_by_scripts_completion( prompts, 
                                    datasets=None,
                                    content_type: str ='common',
                                    conversation_settings: dict = {},
                                    llm_settings: dict = {},
                                    text_embedding_settings: dict = {},
                                    tokens_blog = {},
                                    logger=None,
                                   **kwargs
                                   ):
    ## retrieved motion
    if conversation_settings['start_by_retrieved_motion']:
        retrieved_motion = retrieve_motion(datasets.text_to_actions)
    else:
        retrieved_motion = None
    
    ### generate topic or not
    topic_description = ''
    if prompts['topic_type'] == 'none' or prompts['topic_type'] == '':
        topic_description = ''
    elif prompts['topic_type'] == 'preset':
        topic_description = prompts['topic']
    elif prompts['topic_type'] == 'generate':
        topic_description = generate_topic(prompts=prompts, 
                                           motion=retrieved_motion, 
                                           tokens_blog=tokens_blog, 
                                           llm_settings=llm_settings,
                                           logger=logger)
    
    dialogs = []
    
    ## generate next round
    for turn in range(conversation_settings['NUM_ROUNDS'] * 2):
        ## generate next round
        current_round = str(turn // 2)
        curr_role = 'A' if (len(dialogs) == 0 or dialogs[-1].role == 'B') else 'B'
        
        dialogs_des = '\n'.join([str(dialog) for dialog in dialogs])
        
        messages = []
        
        prompt_system = prompts['background'] + prompts['user_settings'] + prompts['agent_settings']
        prompt_system += prompts['space_limitation'] + prompts['locomotion_limitation'] + prompts['behavior_illustration']
        if topic_description != '':
            prompt_system += f"""The topic of the social conversation is: {topic_description}."""
        
        messages.append({"role": 'system', "content": prompt_system})
        
        inter_type = False
        
        if retrieved_motion is not None and turn == 0:
            if 'current_text' not in retrieved_motion.keys():
                texts_len = len(retrieved_motion['text'])
                text_index = np.random.randint(texts_len)
                retrieved_motion['current_text'] = retrieved_motion['text'][text_index]
            prompt_generate = f"""Character A is acting out the motion to start the conversation:  '{retrieved_motion['current_text']}.'
            Please generate the role {curr_role}'s behavior(speech, motion, expression) for this round (round 0) to start the conversation.
            It should be plausible and reasonable with the daily life and the topic.
            Your response must be a json dict format.
            One key is 'inter' and its value must be 'N'. Another key is 'behavior', which is the behavior script of role {curr_role} according to the format of the case's response.
            
            ### Example:
            Example 1:
            If role A's motion description is like this: Raises hand and waves at B
            Then the response is like this:
            {{'inter': 'N', 'behavior': '<round>0<role>A<motion>Raises hand and waves at B<speech>Hello over there!<expression>Smile'}}
            
            Example 2:
            If role A's motion description is like this: Yawn, shoulder drop
            Then the response is like this:
            {{'inter': 'N', 'behavior': '<round>0<role>A<motion>Yawn, shoulder drop<speech>I'm finally back.<expression>Tired'}}
            
            Be sure to output the result according to the format of the case's response.
            """
            pass
        else:
            prompt_generate = f"""Below are the interaction scripts between characters A and B.
            Please generate the role {curr_role}'s behavior(speech, motion, expression) for the next round that can respond to the previous bahaviors of his/her partner.
            It should be plausible and reasonable with the daily life.
            Your response must be a json dict format."""
            
            action_tmp = dialogs[-1].action
            if action_tmp['next_partner_motion_name'] is not None:
                inter_type = True
            if inter_type:
                next_partner = datasets.dataset_items[action_tmp['next_partner_motion_name']]
                texts_len = len(next_partner['text'])
                text_index = np.random.randint(texts_len)
                next_partner['current_text'] = next_partner['text'][text_index]
                prompt_generate += f"""In some situation, when one person performs the action: '{dialogs[-1].motion}',
                the other person may/tends to response with the action: '{next_partner['current_text']} depending on the situation'.
                If the role {curr_role} response with the action: '{next_partner['current_text']}', 
                then you should output like : {{'inter': 'Y', 'behavior': '<round>{current_round}<role>XX<motion>{next_partner['current_text']}<speech>ZZZZZ<expression>QQQQ'}}.
                If the role {curr_role} do not response with the mentioned action, you should output like : {{'inter': 'N', 'behavior': '<round>{current_round}<role>XX<motion>YYYYY<speech>ZZZZZ<expression>QQQQ'}}.
                """
            else:
                prompt_generate += f"""Here you should output like : {{'inter': 'N', 'behavior': '<round>{current_round}<role>XX<motion>YYYYY<speech>ZZZZZ<expression>QQQQ'}}.
                """
            prompt_generate += f"""
            The key 'inter' means whether the role {curr_role} response with the mentioned action ('Y' mean yes, and 'N' mean no), 
            and the key 'behavior' is the behavior script of role {curr_role}.
            Be sure to output in json dict format according to the format of the case's response!
            The behavior must be role {curr_role}'s behavior respond to his/her partner in round {current_round}.
            """
                
            prompt_generate += f"""
            ### Interaction scripts:
            {dialogs_des}
            
            ### Example:
            If the input scripts are like this:
            <round>0<role>A<motion>Raises hand and waves at B<speech>Hello over there!<expression>Smile
            <round>0<role>B<motion>Waves back<speech>Hey, I see! What a day!<expression>Excited
            <round>1<role>A<motion>Yawn, shoulder drop<speech>I'm finally back.<expression>Tired
            
            Then your response in json format is like this:
            {{'inter': 'N', 'behavior': '<round>1<role>B<motion>Walk forward<speech>Why do you look so tired?<expression>Curious'}}
            """

        messages.append({"role": 'user', "content": prompt_generate})
        
        check_generate = False
        for i in range(llm_settings['repeat_times']):
            try:
                current_settings = llm_settings.copy()
                current_settings['json_format'] = True
                response = ChatGPT_request_messages(messages, tokens_blog=tokens_blog, **current_settings)
                logger.info("Trial {}: Behavior Generated: \n {}".format(i, response))
                if i >= llm_settings['repeat_times'] - 2:
                    pass
                # response = response.strip()
                inter_used, behavior = transfer_response_to_behavior(response, tokens_blog=tokens_blog, **llm_settings)
                if behavior.role != curr_role or behavior.round_id != current_round:
                    continue
                if inter_type and inter_used == 'Y':
                    behavior.action = next_partner
                dialogs.append(behavior)
                check_generate = True
                break
            except:
                pass
        
        if not check_generate:
            logger.critical("Failed to generate the behavior for the next round![Topic]{}".format(topic_description))
            return {}

        if inter_type and inter_used == 'Y':
            continue
    
        last_generated_behavior = dialogs[-1]
        curr_role = last_generated_behavior.role
        last_generated_behavior_motion = last_generated_behavior.motion
        embedding_last_generated_behavior = get_embedding(last_generated_behavior_motion, tokens_blog=tokens_blog, **text_embedding_settings)
        
        similarity = np.dot(embedding_last_generated_behavior, datasets.embeddings_np)
        
        top_k = conversation_settings['refine']['top_k']
        top_k_ids = np.argsort(similarity)[::-1][:top_k]
        top_k_actions = [np.random.choice(datasets.text_to_actions[datasets.embeddings_text[id]]) for id in top_k_ids]
        
        for action in top_k_actions:
            text_len = len(action['text'])
            text_index = np.random.randint(text_len)
            action['current_text'] = action['text'][text_index]
        
        top_k_actions_des = '\n'.join([f"""{idx+1}. {top_k_actions[idx]['current_text']}""" for idx in range(len(top_k_actions))])
        dialogs_des = '\n'.join([str(dialog) for dialog in dialogs])
        
        ## refine the last generated behavior
        messages = []
        prompt_system = prompts['background'] + prompts['user_settings'] + prompts['agent_settings']
        prompt_system += prompts['space_limitation'] + prompts['locomotion_limitation'] + prompts['behavior_illustration']
        if topic_description != '':
            prompt_system += f"""The topic of the social conversation is: {topic_description}."""
        messages.append({"role": 'system', "content": prompt_system})
        
        prompt_refine = f"""Below are the interaction scripts between roles A and B.
        ####
        Attention:The last row is the interaction scripts generated by the AI systems about role {curr_role}'s behavior, and the rest rows are true recordings between A and B.
        In real life, the motion of last row can only be selected from following top {top_k} most similar motions to the last motion.
        ####
        Please select the most plausible and reasonable motion from the following top {top_k} most similar motions (just give a number index), and regenerate the behavior of role {curr_role}.
        Your response must be json dict format. One key is 'index', which is the number index (int type) of the selected motion, and the other key is 'behavior', which is the behavior script of role {curr_role} according to the format of the case's response.

        Top {top_k} most similar motions to the last motion:
        {top_k_actions_des}

        Interaction scripts:
        {dialogs_des}


        Here are some examples:
        ###
        Example 1:
        If the input is:
        Top 4 most similar motions to the last motion:
        1. Raises hand 
        2. Waves back
        3. Shaking hands
        4. Raises hand and smiles

        Interaction scripts:
        <round>0<role>A<motion>Raises hand and waves at B<speech>Hello over there!<expression>Smile
        <round>0<role>B<motion>Raises hand<speech>Hey, I see! What a day!<expression>Excited

        Then your response is like this:
        {{'index': 2, 'behavior': "<role>B<motion>Waves back<speech>Yes, I saw you!<expression>Smile"}}
        
        
        Example 2:
        If the input is:
        Top 3 most similar motions to the last motion:
        1. go left and do a moonwalk 
        2. walks around
        3. walks backward

        Interaction scripts:
        <round>0<role>A<motion>Wave hands<speech>Can you do moonwalk?<expression>Smile
        <round>0<role>B<motion>Shake head<speech>No. Can you show me how to do it?<expression>Excited
        <round>1<role>A<motion>moonwalk<speech>Just like this<expression>Smile

        Then your response is like this:
        {{'index': 1, "behavior": "<role>A<motion>go left and do a moonwalk<speech>Just like this<expression>Smile"}}
        ###
        
        Be sure to output the behavior script of role {curr_role} according to the format of the case's response! 
        The behavior should be in format like '<round>{str(dialogs[-1].round_id)}<role>{curr_role}<motion>YYYYY<speech>ZZZZZ<expression>QQQQ'.
        """
        messages.append({"role": 'user', "content": prompt_refine})
        
        check_generate = False
        for i in range(llm_settings['repeat_times']):
            try:
                current_settings = llm_settings.copy()
                current_settings['json_format'] = True
                response = ChatGPT_request_messages(messages, tokens_blog=tokens_blog, **current_settings)
                logger.info("Trial {}: Refined Behavior: \n {}".format(i, response))
                if i >= llm_settings['repeat_times'] - 2:
                    pass
                index, behavior = transfer_response_to_regenerated_behavior(response, top_k=top_k, tokens_blog=tokens_blog, **llm_settings)
                behavior.action = top_k_actions[index-1]
                if behavior.role != curr_role or behavior.round_id != current_round:
                    continue
                
                logger.info("Update motion from [{}] to [{}]".format(last_generated_behavior.motion, behavior.motion))
                dialogs[-1] = behavior
                check_generate = True
                break
            except:
                pass
        
        if not check_generate:
            logger.critical("Failed to refine the behavior! [Topic]{}".format(topic_description))
            return {}
    
    generate_res = {}
    generate_res['dialogs_raw'] = dialogs
    generate_res['topic'] = topic_description
    # generate_res['background'] = prompts['background']
    generate_res['user_settings'] = prompts['user_settings']
    generate_res['agent_settings'] = prompts['agent_settings']
    # generate_res['space_limitation'] = prompts['space_limitation']
    # generate_res['locomotion'] = prompts['locomotion_limitation']
    generate_res['content_type'] = content_type
    
    generate_res['generation_method'] = 'script completion'
    generate_res['llm_model'] = llm_settings['model']
    generate_res['time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    generate_res['text_embedding_model'] = text_embedding_settings['model']
    generate_res['start_by_retrieved_motion'] = conversation_settings['start_by_retrieved_motion']
    return generate_res


def generate_by_agent_conversation(
    prompts, 
    datasets=None,
    content_type: str ='common',
    conversation_settings: dict = {},
    llm_settings: dict = {},
    text_embedding_settings: dict = {},
    tokens_blog = {},
    logger=None,
    **kwargs
):
     ## retrieved motion
    if conversation_settings['start_by_retrieved_motion']:
        retrieved_motion = retrieve_motion(datasets.text_to_actions)
    else:
        retrieved_motion = None
    
    ### generate topic or not
    topic_description = ''
    if prompts['topic_type'] == 'none' or prompts['topic_type'] == '':
        topic_description = ''
    elif prompts['topic_type'] == 'preset':
        topic_description = prompts['topic']
    elif prompts['topic_type'] == 'generate':
        topic_description = generate_topic(prompts=prompts, 
                                           motion=retrieved_motion, 
                                           tokens_blog=tokens_blog, 
                                           llm_settings=llm_settings,
                                           logger=logger)
    
    
    dialogs = []
    for turn in range(conversation_settings['NUM_ROUNDS'] * 2):
        ## generate next round
        current_round = str(turn // 2)
        curr_role = 'A' if (len(dialogs) == 0 or dialogs[-1].role == 'B') else 'B'
        
        messages = []
        prompt_generate = f"""Let's do a role play."""
        prompt_generate += prompts['background']
        if curr_role == 'A':
            prompt_generate += f"""You are Character A.""" + prompts['user_settings']
            prompt_generate += f"""I'm Character B.""" + prompts['agent_settings']
        else:
            prompt_generate += f"""You are Character B.""" + prompts['agent_settings']
            prompt_generate += f"""I'm Character A.""" + prompts['user_settings']
        prompt_generate += prompts['space_limitation'] + prompts['locomotion_limitation'] + prompts['behavior_illustration']
        
        if topic_description != '':
                prompt_generate += f"""The topic of the social conversation is: {topic_description}."""
        
        prompt_generate += f"""Below are the interaction scripts between us."""
        messages.append({"role": 'system', "content": prompt_generate})
        
        inter_type = False
        
        for dialog in dialogs:
            if dialog.role == curr_role:
                messages.append({"role": 'assistant', "content": str(dialog)})
            else:
                messages.append({"role": 'user', "content": str(dialog)})
        
        if len(messages) == 1:
                messages.append({"role": 'user', "content": ""})
                
        user_prompt = messages[-1]['content'] + '\n'
        
        
        if retrieved_motion is not None and turn == 0:
            if 'current_text' not in retrieved_motion.keys():
                texts_len = len(retrieved_motion['text'])
                text_index = np.random.randint(texts_len)
                retrieved_motion['current_text'] = retrieved_motion['text'][text_index]
            user_prompt += f"""You are Character {curr_role}. You are acting out this motion to start the conversation:  '{retrieved_motion['current_text']}.'
            Please generate your behavior(speech, motion, expression) for this round (round 0) to start the conversation.
            It should be plausible and reasonable with the daily life and the topic.
            Your response must be a json dict format.
            One key is 'inter' and its value must be 'N'. Another key is 'behavior', which is the behavior script of role {curr_role} according to the format of the case's response.
            
            ### Example:
            Example 1:
            If your motion description is like this: Raises hand and waves at B
            Then the response is like this:
            {{'inter': 'N', 'behavior': '<round>0<role>A<motion>Raises hand and waves at B<speech>Hello over there!<expression>Smile'}}
            
            Example 2:
            If your motion description is like this: Yawn, shoulder drop
            Then the response is like this:
            {{'inter': 'N', 'behavior': '<round>0<role>A<motion>Yawn, shoulder drop<speech>I'm finally back.<expression>Tired'}}
            
            Be sure to output the script according to the format of the case's response!
            """
        else:
            user_prompt += f"""You are Character {curr_role}. 
            Please generate your behavior(speech, motion, expression) for the next round that can respond to the previous bahaviors of your partner.
            It should be plausible and reasonable with the daily life. 
            Your response must be a json dict format."""
            
            action_tmp = dialogs[-1].action
            if action_tmp['next_partner_motion_name'] is not None:
                inter_type = True
            if inter_type:
                next_partner = datasets.dataset_items[action_tmp['next_partner_motion_name']]
                texts_len = len(next_partner['text'])
                text_index = np.random.randint(texts_len)
                next_partner['current_text'] = next_partner['text'][text_index]
                user_prompt += f"""In some situation, when one person performs the action: '{dialogs[-1].motion}',
                the other person may/tends to response with the action: '{next_partner['current_text']} depending on the situation'.
                If the role {curr_role} response with the action: '{next_partner['current_text']}', 
                then you should output like : {{'inter': 'Y', 'behavior': '<round>{current_round}<role>XX<motion>{next_partner['current_text']}<speech>ZZZZZ<expression>QQQQ'}}.
                If the role {curr_role} do not response with the mentioned action, you should output like : {{'inter': 'N', 'behavior': '<round>{current_round}<role>XX<motion>YYYYY<speech>ZZZZZ<expression>QQQQ'}}.
                """
            else:
                user_prompt += f"""Here you should output like : {{'inter': 'N', 'behavior': '<round>{current_round}<role>XX<motion>YYYYY<speech>ZZZZZ<expression>QQQQ'}}."""
            
            user_prompt += f"""\n
            The key 'inter' means whether the role {curr_role} response with the mentioned action ('Y' mean yes, and 'N' mean no), 
            and the key 'behavior' is the behavior script of role {curr_role}.
            Be sure to output in json dict format according to the format of the case's response!
            The behavior must be role {curr_role}'s behavior respond to his/her partner in round {current_round}.
            
            ### Example:
            Example 1:
            {{'inter': 'Y', 'behavior': '<round>0<role>{curr_role}<motion>Raises hand and waves at B<speech>Hello over there!<expression>Smile'}}
            
            Example 2:
            {{'inter': 'N', 'behavior': '<round>1<role>{curr_role}<motion>Walk forward<speech>Why do you look so tired?<expression>Curious'}}
            """
        
        messages[-1]['content'] = user_prompt
        
        
        check_generate = False
        for i in range(llm_settings['repeat_times']):
            try:
                current_settings = llm_settings.copy()
                current_settings['json_format'] = True
                response = ChatGPT_request_messages(messages, tokens_blog=tokens_blog, **current_settings)
                logger.info("Trial {}: Behavior Generated: \n {}".format(i, response))
                if i >= llm_settings['repeat_times'] - 2:
                    pass
                inter_used, behavior = transfer_response_to_behavior(response, tokens_blog=tokens_blog, **llm_settings)
                if behavior.role != curr_role or behavior.round_id != current_round:
                    continue
                if inter_type and inter_used == 'Y':
                    behavior.action = next_partner
                dialogs.append(behavior)
                check_generate = True
                break
            except:
                pass
        
        if not check_generate:
            logger.critical("Failed to generate the behavior for the next round![Topic]{}".format(topic_description))
            return {}
        
        if inter_type and inter_used == 'Y':
            continue
        
        last_generated_behavior = dialogs[-1]
        curr_role = last_generated_behavior.role
        last_generated_behavior_motion = last_generated_behavior.motion
        embedding_last_generated_behavior = get_embedding(last_generated_behavior_motion, tokens_blog=tokens_blog, **text_embedding_settings)
        
        similarity = np.dot(embedding_last_generated_behavior, datasets.embeddings_np)
        
        top_k = conversation_settings['refine']['top_k']
        top_k_ids = np.argsort(similarity)[::-1][:top_k]
        top_k_actions = [np.random.choice(datasets.text_to_actions[datasets.embeddings_text[id]]) for id in top_k_ids]
        
        for action in top_k_actions:
            text_len = len(action['text'])
            text_index = np.random.randint(text_len)
            action['current_text'] = action['text'][text_index]
        
        top_k_actions_des = '\n'.join([f"""{idx+1}. {top_k_actions[idx]['current_text']}""" for idx in range(len(top_k_actions))])
        
        messages = []
        prompt_generate = f"""Let's do a role play."""
        prompt_generate += prompts['background']
        if curr_role == 'A':
            prompt_generate += f"""You are Character A.""" + prompts['user_settings']
            prompt_generate += f"""I'm Character B.""" + prompts['agent_settings']
        else:
            prompt_generate += f"""You are Character B.""" + prompts['agent_settings']
            prompt_generate += f"""I'm Character A.""" + prompts['user_settings']
        prompt_generate += prompts['space_limitation'] + prompts['locomotion_limitation'] + prompts['behavior_illustration']
        
        if topic_description != '':
                prompt_generate += f"""The topic of the social conversation is: {topic_description}."""
        
        prompt_generate += f"""Below are the interaction scripts between us."""
        messages.append({"role": 'system', "content": prompt_generate})
        
        for dialog in dialogs:
            if dialog.role == curr_role:
                messages.append({"role": 'assistant', "content": str(dialog)})
            else:
                messages.append({"role": 'user', "content": str(dialog)})

                
        user_prompt = f"""Your response is {str(messages[-1]['content'])}. 
            While in real life, your last response motion can only be selected from following top {top_k} most similar motions to the last motion.
            Please select the most plausible and reasonable motion from the following top {top_k} most similar motions (just give a number index), 
            and regenerate your behavior.
            Your response must be json dict format. One key is 'index', which is the number index (int type) of the selected motion, and the other key is 'behavior', 
            which is the behavior script of role {curr_role} according to the format of the case's response.
            
            Top {top_k} most similar motions to the last motion:
            {top_k_actions_des}
            
             Here are some examples:
            ###
            Example 1:
            If the input is:
            Top 4 most similar motions to the last motion:
            1. Raises hand 
            2. Waves back
            3. Shaking hands
            4. Raises hand and smiles

            And your last response is like this:
            <role>{curr_role}<motion>Raises hand<speech>Hey, I see! What a day!<expression>Excited

            Then your response is like this:
            {{'index': 2, 'behavior': "<role>{curr_role}<motion>Waves back<speech>Yes, I saw you!<expression>Smile"}}
            
            
            Example 2:
            If the input is:
            Top 3 most similar motions to the last motion:
            1. go left and do a moonwalk 
            2. walks around
            3. walks backward

            And your last response is like this:
            <role>{curr_role}<motion>moonwalk<speech>Just like this<expression>Smile

            Then your response is like this:
            {{'index': 1, "behavior": "<role>{curr_role}<motion>go left and do a moonwalk<speech>Just like this<expression>Smile"}}
            
            ###
            Be sure to output the behavior script of role {curr_role} according to the format of the case's response! 
            The behavior should be in format like '<round>{str(dialogs[-1].round_id)}<role>{curr_role}<motion>YYYYY<speech>ZZZZZ<expression>QQQQ'.
            """
        
        messages.append({"role": 'user', "content": user_prompt})
        
        
        check_generate = False
        for i in range(llm_settings['repeat_times']):
            try:
                current_settings = llm_settings.copy()
                current_settings['json_format'] = True
                response = ChatGPT_request_messages(messages, tokens_blog=tokens_blog, **current_settings)
                logger.info("Trial {}: Refined Behavior: \n {}".format(i, response))
                if i >= llm_settings['repeat_times'] - 2:
                    pass
                index, behavior = transfer_response_to_regenerated_behavior(response, top_k=top_k, tokens_blog=tokens_blog, **llm_settings)                    
                behavior.action = top_k_actions[index-1]
                if behavior.role != curr_role or behavior.round_id != current_round:
                    continue

                logger.info("Update motion from [{}] to [{}]".format(last_generated_behavior.motion, behavior.motion))
                dialogs[-1] = behavior
                check_generate = True
                break
            except:
                pass
        
        if not check_generate:
            logger.critical("Failed to refine the behavior! [Topic]{}".format(topic_description))
            return {}
        
    generate_res = {}
    generate_res['dialogs_raw'] = dialogs
    generate_res['topic'] = topic_description
    # generate_res['background'] = prompts['background']
    generate_res['user_settings'] = prompts['user_settings']
    generate_res['agent_settings'] = prompts['agent_settings']
    # generate_res['space_limitation'] = prompts['space_limitation']
    # generate_res['locomotion'] = prompts['locomotion_limitation']
    generate_res['content_type'] = content_type
    
    generate_res['generation_method'] = 'agent conversation'
    generate_res['llm_model'] = llm_settings['model']
    generate_res['time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    generate_res['text_embedding_model'] = text_embedding_settings['model']
    generate_res['start_by_retrieved_motion'] = conversation_settings['start_by_retrieved_motion']
    return generate_res


def generate_by_generate_once(
    prompts,
    datasets=None,
    content_type: str ='common',
    conversation_settings: dict = {},
    llm_settings: dict = {},
    text_embedding_settings: dict = {},
    tokens_blog = {},
    logger=None,
    **kwargs
):
    ## retrieved motion
    if conversation_settings['start_by_retrieved_motion']:
        retrieved_motion = retrieve_motion(datasets.text_to_actions)
    else:
        retrieved_motion = None
    
    dialogs = []
    
    ### generate topic or not
    topic_description = ''
    if prompts['topic_type'] == 'none' or prompts['topic_type'] == '':
        topic_description = ''
    elif prompts['topic_type'] == 'preset':
        topic_description = prompts['topic']
    elif prompts['topic_type'] == 'generate':
        topic_description = generate_topic(prompts=prompts, 
                                           motion=retrieved_motion, 
                                           tokens_blog=tokens_blog, 
                                           llm_settings=llm_settings,
                                           logger=logger)
        
    prompt_generate = prompts['background'] + prompts['user_settings'] + prompts['agent_settings']
    prompt_generate += prompts['space_limitation'] + prompts['locomotion_limitation'] + prompts['behavior_illustration']

    if topic_description != '':
        prompt_generate += f"""The topic of the social conversation that Character A want to start with Character B is: {topic_description}."""
    
    prompt_generate += f"""
        Please generate the interaction scripts between characters A and B.
        The scripts should have {conversation_settings['NUM_ROUNDS'] * 2} lines of interaction, 
        and each line should have one behavior(speech, motion, expression) for one of the characters.
        The interactions of characters A and B must alternate, with A acting first.
        
     ### Examples
        Example 1 (8 lines of interaction):
        <round>0<role>A<motion>sitting down with hands near face<speech>Just been feeling kinda off lately. You know?<expression>Melancholy
        <round>0<role>B<motion>put his left hand on someone's shoulder<speech>What's been on your mind?<expression>Concerned
        <round>1<role>A<motion>appears to be disappointed and sighs<speech>Just stress from work and life, I guess. It's been a lot to handle.<expression>Weary
        <round>1<role>B<motion>person turns and then sits down<speech>Have you thought about talking to someone, like a therapist or a friend?<expression>Supportive
        <round>2<role>A<motion>a person slowly walks forward with their head down<speech>I've considered it, but I don't know where to start.<expression>Uncertain
        <round>2<role>B<motion>gestures with both arms<speech>Let's go for a walk. Sometimes a change in scenery can help clear the mind.<expression>Encouraging
        <round>3<role>A<motion>starts walking forward slowly<speech>Yeah, that sounds like a good idea. Thanks for suggesting it.<expression>Grateful
        <round>3<role>B<motion>points at a direction<speech>How about going to the central park?<expression>Excited
     
        Example 2 (6 lines of interaction):
        <round>0<role>A<motion>look around curiously and put hands on hips<speech>Did you know the moon has quakes just like the Earth?<expression>Intrigued
        <round>0<role>B<motion>raises one eyebrow and tilts head slightly<speech>Moonquakes, huh? How strong do they get?<expression>Curious
        <round>1<role>A<motion>gestures with hands like waves<speech>Some can be pretty strong, up to a magnitude of 5.5!<expression>Amazed
        <round>1<role>B<motion>nods and rubs chin thoughtfully<speech>That's fascinating. I wonder if any future lunar bases will have to be quake-proof.<expression>Thoughtful
        <round>2<role>A<motion>claps hands together once<speech>Imagine living on the moon and having a quake-proof house. That would be wild!<expression>Excited
        <round>2<role>B<motion>mimics walking in low gravity<speech>And walking around in low gravity all day would be a bonus!<expression>Playful
    
        Be sure to output the script according to the format of the case's response!
        Your response must be {conversation_settings['NUM_ROUNDS'] * 2} lines without any superfluous content and each line is in the format of the behavior (like '<round>PP<role>XX<motion>YYYYY<speech>ZZZZZ<expression>QQQQ').
    """
    
    check_generate = False
    for i in range(llm_settings['repeat_times']):
        try:
            response = ChatGPT_request_messages(prompt_generate, tokens_blog=tokens_blog, **llm_settings)
            logger.info("Trial {}: Behavior Generated: \n {}".format(i, response))
            response = response.strip().split('\n')
            dialogs_raw = []
            for resp in response:
                if resp.startswith('<round>'):
                    dialogs_raw.append(resp)
            
            for i in range(conversation_settings['NUM_ROUNDS'] * 2):
                behavior = transfer_script_to_behavior(dialogs_raw[i], tokens_blog=tokens_blog, **llm_settings)
                if i % 2 == 0 and (behavior.role != 'A' or behavior.round_id != str(i // 2)):
                    continue
                if i % 2 == 1 and (behavior.role != 'B' or behavior.round_id != str(i // 2)):
                    continue
                dialogs.append(behavior)
            check_generate = True
            break
        except:
            pass
    
    if not check_generate:
        logger.critical("Failed to generate the interaction scripts!")
        return {}
        
    for dialog_id in range(len(dialogs)):
        
        curr_generated_behavior = dialogs[dialog_id]
        curr_role = curr_generated_behavior.role
        curr_generated_behavior_motion = curr_generated_behavior.motion
        embedding_curr_generated_behavior = get_embedding(curr_generated_behavior_motion, tokens_blog=tokens_blog, **text_embedding_settings)
        
        similarity = np.dot(embedding_curr_generated_behavior, datasets.embeddings_np)
        
        top_k = conversation_settings['refine']['top_k']
        top_k_ids = np.argsort(similarity)[::-1][:top_k]
        top_k_actions = [np.random.choice(datasets.text_to_actions[datasets.embeddings_text[id]]) for id in top_k_ids]
        
        for action in top_k_actions:
            text_len = len(action['text'])
            text_index = np.random.randint(text_len)
            action['current_text'] = action['text'][text_index]
        
        top_k_actions_des = '\n'.join([f"""{idx+1}. {top_k_actions[idx]['current_text']}""" for idx in range(len(top_k_actions))])
        dialogs_des = '\n'.join([str(dialog) for dialog in dialogs])
        
        messages = []
        prompt_system = prompts['background'] + prompts['user_settings'] + prompts['agent_settings']
        prompt_system += prompts['space_limitation'] + prompts['locomotion_limitation'] + prompts['behavior_illustration']
        if topic_description != '':
            prompt_system += f"""The topic of the social conversation that Character A want to start with Character B is: {topic_description}."""
        messages.append({"role": 'system', "content": prompt_system})
        
        prompt_refine = f"""Below are the interaction scripts between roles A and B.
        
        ### Interaction scripts:
        {dialogs_des}
        
        Attention:
        The {dialog_id+1}-th line is the interaction scripts generated by the AI systems about role {curr_role}'s behavior, and the rest lines are true recordings between A and B.
        This AI-generated line is:
        {str(curr_generated_behavior)}
        
        In real life, the motion of this line can only be selected from following top {top_k} most similar motions to the AI-generated motion in {dialog_id+1}-th line.
        Please select the most plausible and reasonable motion from the following top {top_k} most similar motions (just give a number index), and regenerate the behavior of role {curr_role}.
        Your response must be json dict format. One key is 'index', which is the number index (int type) of the selected motion, 
        and the other key is 'behavior', which is the behavior script of role {curr_role} according to the format of the case's response.
        
        ### Top {top_k} most similar motions to the last motion:
        {top_k_actions_des}
        
        Here are some examples:
        ###
        Example 1:
        If the top 4 most similar motions is:
        1. Raises hand 
        2. Waves back
        3. Shaking hands
        4. Raises hand and smiles

        AI-generated interaction scripts is:
        <round>0<role>B<motion>Raises hand<speech>Hey, I see! What a day!<expression>Excited

        Then your response is like this:
        {{'index': 2, 'behavior': "<role>B<motion>Waves back<speech>Yes, I saw you!<expression>Smile"}}
        
        
        Example 2:
        If the top 3 most similar motions to the AI-generated motion is:
        1. go left and do a moonwalk 
        2. walks around
        3. walks backward

        AI-generated interaction scripts is:
        <round>1<role>A<motion>moonwalk<speech>Just like this<expression>Smile

        Then your response is like this:
        {{'index': 1, "behavior": "<role>A<motion>go left and do a moonwalk<speech>Just like this<expression>Smile"}}
        ###
        Be sure to output the behavior script of role {curr_role} according to the format of the case's response! 
        The behavior should be in format like '<round>{str(dialogs[-1].round_id)}<role>{curr_role}<motion>YYYYY<speech>ZZZZZ<expression>QQQQ'.
        """

        messages.append({"role": 'user', "content": prompt_refine})
        check_generate = False
        for i in range(llm_settings['repeat_times']):
            try:
                current_settings = llm_settings.copy()
                current_settings['json_format'] = True
                response = ChatGPT_request_messages(messages, tokens_blog=tokens_blog, **current_settings)
                logger.info("Trial {}: Refined Behavior: \n {}".format(i, response))
                if i == llm_settings['repeat_times'] - 2:
                    pass
                index, behavior = transfer_response_to_regenerated_behavior(response, top_k=top_k, tokens_blog=tokens_blog, **llm_settings)
                behavior.action = top_k_actions[index-1]
                
                if behavior.role != curr_role or behavior.round_id != str(dialog_id // 2):
                    continue
                
                logger.info("Update motion from [{}] to [{}]".format(curr_generated_behavior.motion, behavior.motion))
                dialogs[dialog_id] = behavior
                check_generate = True
                break
            except:
                pass
        
        if not check_generate:
            logger.critical("Failed to refine the behavior! [Topic]{}".format(topic_description))
            return {}
    
    generate_res = {}
    generate_res['dialogs_raw'] = dialogs
    generate_res['topic'] = topic_description
    # generate_res['background'] = prompts['background']
    generate_res['user_settings'] = prompts['user_settings']
    generate_res['agent_settings'] = prompts['agent_settings']
    # generate_res['space_limitation'] = prompts['space_limitation']
    # generate_res['locomotion'] = prompts['locomotion_limitation']
    generate_res['content_type'] = content_type
    
    generate_res['generation_method'] = 'generate once'
    generate_res['llm_model'] = llm_settings['model']
    generate_res['time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    generate_res['text_embedding_model'] = text_embedding_settings['model']
    generate_res['start_by_retrieved_motion'] = conversation_settings['start_by_retrieved_motion']
    return generate_res