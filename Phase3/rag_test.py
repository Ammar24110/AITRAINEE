import os
import requests
from dotenv import load_dotenv
from openai import AzureOpenAI

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

search_service = os.getenv("AZURE_SEARCH_SERVICE")
search_index = os.getenv("AZURE_SEARCH_INDEX")
search_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

question = input("You: ")

search_url = f"https://{search_service}.search.windows.net/indexes/{search_index}/docs/search?api-version=2024-07-01"

search_headers = {
    "Content-Type": "application/json",
    "api-key": search_key,
}

search_payload = {
    "search": question,
    "top": 2
}

search_response = requests.post(search_url, headers=search_headers, json=search_payload)
results = search_response.json()["value"]

context = "\n\n".join(doc.get("chunk", "") for doc in results[:2])

response = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    messages=[
        {
            "role": "system",
            "content": "You are a helpful RAG assistant. Answer only from the provided context. If the answer is not in the context, say you do not know."
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nContext:\n{context}"
        }
    ],
)

print("\nAssistant:", response.choices[0].message.content)