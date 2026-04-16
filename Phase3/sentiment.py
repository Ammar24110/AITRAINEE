import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv("Phase3/.env")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

response = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    messages=[
        {
            "role": "system",
            "content": "You are a sentiment analysis assistant. Reply with only one word: Positive, Negative, or Neutral."
        },
        {
            "role": "user",
            "content": "Analyze the sentiment of this sentence: I like football its my favorite sport."
        }
    ],
)

print(response.choices[0].message.content)