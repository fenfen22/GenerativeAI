from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict
from pathlib import Path
import json


class ChatMessage:
    def __init__(self, sender, message):
        self.sender = sender
        self.message = message

with Path("message.json").open("r") as f:
    data = json.load(f)

messages = []
for conversation_thread in data:
    for message_info in conversation_thread:
        message_type = message_info['type']
        message_content = message_info['data']['content']

        # Create a ChatMessage object for each message
        message = ChatMessage(
            sender=message_type,
            message=message_content
        )
        messages.append(message)
    

new_list = []

for i in messages:
    message_dict = {
        "type": i.sender,
        "data": i.message
    }
    new_list.append(message_dict)

# print(messages_from_dict(new_list))
# history = ChatMessageHistory(message=messages_from_dict(new_list))
# print(history)
