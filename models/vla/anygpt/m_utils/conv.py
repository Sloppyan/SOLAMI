from enum import auto, IntEnum
from typing import List, Any, Dict, Union, Tuple


class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()



class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""
    ### init the class
    
    def __init__(self,
        name: str, 
        system_template: str = "{system_info}", 
        system_message: dict = {}, 
        character: str='',
        method='solami',
        agent_voice_prompt: str = None,
        user_voice_prompt: str = None,
        roles: Tuple[str] = ("USER", "ASSISTANT"), 
        messages: List[List[str]] = [], 
        offset: int = 0, 
        sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE, 
        sep: str = "\n", 
        sep2: str = None, 
        stop_str: Union[str, List[str]] = None, 
        stop_token_ids: List[int] = None):
    
        self.name = name
        self.method = method
        self.system_template = system_template
        self.system_message = system_message
        self.character = character
        self.roles = roles
        self.messages = messages
        self.offset = offset
        self.sep_style = sep_style
        self.sep = sep
        self.sep2 = sep2
        self.stop_str = stop_str
        self.stop_token_ids = stop_token_ids
        self.agent_voice_prompt = agent_voice_prompt
        self.user_voice_prompt = user_voice_prompt
    

    def get_prompt(self, agent_role, start_rounds=0) -> str:
        """Get the prompt for generation."""
        if start_rounds >= len(self.messages):
            return ""
        system_prompt = self.system_template.format(system_info=self.system_message[agent_role])
        ret = system_prompt + "\n"
        for i in range(start_rounds, len(self.messages)):
            role, message = self.messages[i]
            if message and i % 2 == 0:
                ret += role + ": " + message + self.sep + "\n"
            elif message and i % 2 == 1:
                ret += role + ": " + message + self.sep2
                if i != len(self.messages) - 1:
                    ret += "\n"
            else:
                ret += role + ":"
        if self.messages[-1][0] == self.roles[0]:
            ret += self.roles[1] + ":"
        return ret
    
    def get_history(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        ret = system_prompt + "\n"
        for i in range(len(self.messages)):
            role, message = self.messages[i]
            if message and i % 2 == 0:
                ret += role + ": " + message + self.sep
                if i != len(self.messages) - 1:
                    ret += "\n"
            elif message and i % 2 == 1:
                ret += role + ": " + message + self.sep2
                if i != len(self.messages) - 1:
                    ret += "\n"
            else:
                ret += role + ":"
        return ret
    
    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def delete_message(self, idx: int):
        """Delete a message."""
        self.messages.pop(idx)

    def get_message_rounds(self,):
        return len(self.messages)

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def to_llama_chat(self, start_rounds=0):
        conversation = []
        conversation.append({"role": "system", "content": self.system_message[self.character]})
        for i, (role, msg) in enumerate(self.messages[start_rounds :]):
            if i % 2 == 0:
                conversation.append({"role": 'user', "content": msg})
            else:
                conversation.append({"role": 'assistant', "content": msg})
                
        return conversation

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }

    def reset(self):
        self.messages = []
        self.offset = 0