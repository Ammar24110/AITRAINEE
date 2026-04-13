import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

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
            "content": "You are a helpful assistant that summarizes text clearly and briefly."
        },
        {
            "role": "user",
            "content": "Summarize this text in 2 sentences: Azure OpenAI is a Microsoft Azure service that gives developers access to powerful OpenAI models for chat, text generation, summarization, and more. It helps organizations build AI applications with Azure security, scalability, and enterprise features."
        }
    ],
)

print(response.choices[0].message.content)