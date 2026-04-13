import os
import requests
from dotenv import load_dotenv

load_dotenv("Phase3/.env")

url = f"{os.getenv('AZURE_OPENAI_ENDPOINT')}openai/deployments/{os.getenv('AZURE_OPENAI_DEPLOYMENT')}/chat/completions?api-version={os.getenv('AZURE_OPENAI_API_VERSION')}"

headers = {
    "Content-Type": "application/json",
    "api-key": os.getenv("AZURE_OPENAI_API_KEY"),
}

data = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one sentence."}
    ]
}

response = requests.post(url, headers=headers, json=data)
print("Status code:", response.status_code)
print("Reply:", response.json()["choices"][0]["message"]["content"])