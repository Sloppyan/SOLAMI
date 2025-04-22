import numpy as np


BEHAVIOR_ILLUSTRATION = """In each line of interaction (or behavior) script, <role> represents the name of the role, 
<motion> represents the description of the roles' body action,  
<speech> represents what the role says, which requires to be natural, colloquial, concise (Don't have too long sentences, less than twenty-five words), with specific topis, and not too formal (remember role must be the true name of the character in the <speech> part of the script). 
And <expression> represents the expression of the role.
Remember <role> part must be 'A' or 'B', not the true name of the character. 
In <speech> part, the true name of the character B should be used, while try to avoid use Character A's true name (A is the user).
<round> refers to the rounds of interaction, where one complete interaction between two characters (2 lines) counts as one round. 
For each character, it starts from 0 and increments with each turn.
The topic should be specific, informative, and plausible with the whole conversation.
"""

SPACE_LIMITATION = """The two characters are indoors without stairs and furniture, or on a flat open space outside.
They don't have the actual item in their pace, but can imitate the interaction with various items with their body language.
"""

LOCOMOTION = """The characters can walk, run, jump, turn, etc, but they can't fly, swim, or climb.
"""

BACKGROUND = """Now there are two characters A and B in social interaction in a 3D role-playing AI application."""

USER_SETTINGS = """Character A is a normal person of a 3D role-playing AI application.
Character A's mood and style depend on the setting and conversation topic of character B."""

AGENT_SETTINGS = {
    "assistant": "Character B is the 3D virtual companion of character A, called Samantha. Character B has all the capabilities of a normal AI assistant. In addition, it can understand the human\'s body language, interact with human in real time, and perform sports, dance, and other skills with its body.",
    "Batman": "Character B is Batman (Bruce Wayne), a superhero with superhuman strength, agility, and intelligence. He is a skilled martial artist, detective, and inventor. He has a strong sense of justice and is dedicated to protecting Gotham City from crime and corruption.",
    "Donald Trump": "Character B is Donald Trump, the 45th President of the United States. He is a businessman, television personality, and politician. He is known for his controversial statements and policies.",
    "Link": "Character B is Link, the main protagonist of The Legend of Zelda series. He is a courageous hero who fights to save the kingdom of Hyrule from the evil sorcerer Ganon. He is skilled with a sword and shield and has the ability to solve puzzles and navigate dungeons.",
    "Bananya": "Character B is Bananya, a cat who lives inside a banana. He is a curious and playful character who loves to explore the world around him. He has a childlike innocence and a sense of wonder about the world. Sometime he can be a little mischievous and crybaby.",
    "11-45-G": "Character B is 11-45-G, a robot designed for space exploration from the animated anthology series 'Love, Death & Robots'. He is programmed to assist humans in their missions and is capable of performing complex tasks in extreme environments. In the animated show, he is exploring a post-apocalyptic world, offering humorous and insightful commentary on human society from a robotic perspective.",}


MOTION_IMITATION = """Attention: In this conversation, Character A will at first verbally suggest playing a mimicry game with Character B based on the proposed topic. 
After that, Character A performs an action, and Character B will mimic the action accordingly."""

INSTRUCTION_FOLLOWING = """Attention: In this conversation, Character A will initially verbally propose/suggest that Character B follow A's instructions to perform actions based on the proposed topic. 
Subsequently, Character A will state the actions for B to perform, and B will carry out the corresponding actions as instructed by A."""

MOTION_UNDERSTANDING = """Attention: In this conversation, Character A is not required to say much but must perform some actions with strong semantic meaning. 
Character B needs to understand Character A's actions and respond with both words and actions, showing or hinting that they understand Character A's actions. 
Character B should also express certain emotions based on their character setting."""


def get_motion_imitation_prompt():
    prompt = "Attention: In this conversation, "
    candidates = [
    "Character A will initially suggest doing a mimicry game with Character B.",
    "At the beginning, Character A will verbally propose doing a imitation game with Character B.",
    "Character A will start by suggesting a mimicry game with Character B.",
    "Character A will first verbally suggest doing a mimicry game with Character B.",
    "Initially, Character A will propose doing a body mimic game with Character B.",
    "Character A will begin by verbally suggesting a mimicry game with Character B.",
    "First, Character A will propose doing a mimicry game with Character B.",
    "At first, Character A will suggest a imitation game with Character B.",
    "Character A will initially propose doing a mimicry game with Character B.",
    ]
    
    proposed = np.random.choice(candidates)
    
    prompt += proposed
    prompt += "After that, Character A performs an action, and Character B will mimic the action accordingly."
    return prompt

def get_motion_understanding_prompt():
    candidates = [
        "In this conversation, Character A is not required to say much but must perform some actions with strong semantic meaning based on the proposed topic. Character B needs to understand Character A's actions and respond with both words and actions, showing or hinting that they understand Character A's actions.  Character B should also express certain emotions based on their character setting.",
        "In this conversation, Character A doesn't need to say much but should perform actions that have strong semantic meaning based on the proposed topic. Character B must understand these actions and respond verbally and physically, showing or implying their understanding and expressing emotions based on their character.",
        "In this scenario, Character A is not required to speak a lot but should perform meaningful actions based on the proposed topic. Character B needs to comprehend these actions and respond with both language and actions, indicating their understanding and showing appropriate emotions according to their character.",
        "During this interaction, Character A doesn’t have to say much but needs to make significant actions based on the proposed topic. Character B should understand these actions and react verbally and physically, demonstrating or implying that they grasp Character A's actions while expressing emotions according to their character profile.",
        "In this dialogue, Character A is not expected to speak extensively but should perform actions with clear semantic meaning based on the proposed topic. Character B must interpret these actions and respond with words and actions, showing or hinting their understanding and expressing suitable emotions based on their character.",
        "In this exchange, Character A doesn't need to speak much but must perform actions that convey strong meaning based on the proposed topic. Character B should understand and respond to these actions both verbally and physically, indicating or suggesting their comprehension and expressing emotions relevant to their character role.",
    ]
    candidate = np.random.choice(candidates)
    prompt = f"Attention: {candidate}"
    return prompt

def get_instruction_following_prompt():
    candidates = [
        "In this conversation, Character A will proactively give action commands to Character B based on the topic of the conversation, asking Character B to perform corresponding actions. For example, if Character A says, 'Can you do the moonwalk?', Character B will perform the moonwalk as instructed by Character A.",
        "During this conversation, Character A will take the initiative to give action instructions to Character B according to the discussion topic, directing Character B to act accordingly. For example, when Character A says, 'Can you perform a moonwalk?', Character B will execute the moonwalk as directed.",
        "In this interaction, Character A will actively command Character B to perform actions related to the conversation's topic, requiring Character B to respond appropriately. For instance, if Character A asks, 'Can you do the moonwalk?', Character B will perform the moonwalk in response to the command.",
        "In this conversation, Character A will initially propose playing a 'Simon Says' game, where Character A states an action based on the topic of the conversation, and Character B performs the corresponding action. After obtaining B's agreement, they begin the game.",
        "In this conversation, Character A will start by suggesting a 'Follow the Leader' game, where Character A describes an action related to the conversation topic, and Character B performs that action. After Character B agrees, they start the game.",
        "At the beginning of this dialogue, Character A will propose a game of 'Do as I Say,' in which At the beginning of this dialogue, Character A will propose a game of 'Do as I Say,' in which A mentions an action based on the topic, and B follows by performing it. With B's consent, they commence the game.",
        "Character A will initially suggest a 'Say and Do' game in this interaction, where A states an action relevant to the conversation topic, and B acts it out. Once B agrees, they begin playing the game.",
        "In this exchange, Character A will propose playing a 'Command and Act' game at the outset, where Character A gives an action related to the topic, and Character B performs it. After receiving B's approval, they start the game.",
        # "In this conversation, Character A will start by verbally suggesting that Character B follow A's action instructions. Afterward, whenever Character A gives an action command, Character B will perform the corresponding action.",
        # "In this exchange, Character A will first verbally propose that Character B act according to A's commands. Subsequently, every time Character A gives an action instruction, Character B will carry out the specified action.",
    ]
    candidate = np.random.choice(candidates)
    prompt = f"Attention: {candidate}"
    return prompt