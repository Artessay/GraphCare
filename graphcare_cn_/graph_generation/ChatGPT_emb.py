import os
import json
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path="../../resources/.env")

key = os.environ.get("OPENAI_API_KEY")

def embedding_retriever(term):
    # Set up the API endpoint URL and request headers
    url = "https://api.chatanywhere.com.cn/v1/embeddings" # "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }

    # Set up the request payload with the text string to embed and the model to use
    payload = {
        "input": term,
        "model": "text-embedding-ada-002"
    }

    # Send the request and retrieve the response
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Extract the text embeddings from the response JSON
    embedding = response.json()["data"][0]['embedding']

    return embedding

if __name__ == '__main__':
    term = "happy"
    embedding = embedding_retriever(term)
    print(embedding)