import os
import requests
from dotenv import load_dotenv

load_dotenv("Phase3/.env")

service_name = os.getenv("AZURE_SEARCH_SERVICE")
index_name = os.getenv("AZURE_SEARCH_INDEX")
admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")

print("SERVICE:", service_name)
print("INDEX:", index_name)
print("KEY EXISTS:", admin_key is not None)

url = f"https://{service_name}.search.windows.net/indexes/{index_name}/docs/search?api-version=2024-07-01"

headers = {
    "Content-Type": "application/json",
    "api-key": admin_key,
}

payload = {
    "search": "How does the architecture work?",
    "top": 3
}

response = requests.post(url, headers=headers, json=payload)

print("Status code:", response.status_code)
results = response.json()["value"]

for i, doc in enumerate(results[:2], start=1):
    print(f"\nResult {i}")
    print("Title:", doc.get("title"))
    print("Score:", doc.get("@search.score"))
    print("Chunk:", doc.get("chunk"))