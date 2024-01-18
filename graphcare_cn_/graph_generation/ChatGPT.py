import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="../../resources/.env")

class ChatGPT:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url="https://api.chatanywhere.com.cn/v1"
        )

    def chat(self, message):
        chat_completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": message},
                    ],
                }
            ],
        )
        return chat_completion.choices[0].message.content
    
if __name__ == '__main__':
    prompt = "What is the meaning of life?"
    chatGPT = ChatGPT()
    response = chatGPT.chat(prompt)
    print(response)