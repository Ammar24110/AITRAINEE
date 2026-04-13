import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import AzureOpenAI

load_dotenv("Phase3/.env")

app = FastAPI()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {"message": "Phase 3 Azure OpenAI app is running"}


@app.post("/chat")
def chat(request: ChatRequest):
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
           {"role": "system", "content": "You are a helpful assistant. Answer clearly and briefly in 3 sentences maximum."},
            {"role": "user", "content": request.message}
        ],
    )

    return {"reply": response.choices[0].message.content}