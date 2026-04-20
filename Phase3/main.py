import os
from typing import Optional, Tuple, Any, Dict, List

import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from openai import AzureOpenAI
from pydantic import BaseModel
from sqlalchemy.orm import Session
from langdetect import detect

from Phase3.db import ChatSessionModel, Message, get_db, init_db

load_dotenv("Phase3/.env")

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

search_service = os.getenv("AZURE_SEARCH_SERVICE")
search_index = os.getenv("AZURE_SEARCH_INDEX")
search_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

azure_custom_skill_url = os.getenv("AZURE_CUSTOM_SKILL_URL")
azure_custom_skill_key = os.getenv("AZURE_CUSTOM_SKILL_KEY")

SEARCH_API_VERSION = "2024-07-01"
TOP_K = 4
REQUEST_TIMEOUT = 30
HISTORY_LIMIT = 10


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[int] = None


class SkillRecord(BaseModel):
    recordId: str
    data: Dict[str, Any]


class SkillRequest(BaseModel):
    values: List[SkillRecord]


class AzureSkillTestRequest(BaseModel):
    texts: List[str]


@app.on_event("startup")
def startup_event():
    init_db()


@app.get("/")
def home():
    return {"message": "Phase 3 smart chat app is running"}


def get_or_create_session(db: Session, session_id: Optional[int]) -> ChatSessionModel:
    if session_id is None:
        session = ChatSessionModel()
        db.add(session)
        db.commit()
        db.refresh(session)
        return session

    session = db.query(ChatSessionModel).filter(ChatSessionModel.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    return session


def build_history_text(session: ChatSessionModel) -> str:
    recent_messages = session.messages[-HISTORY_LIMIT:]
    history_lines = []

    for msg in recent_messages:
        role_name = "User" if msg.role == "user" else "Assistant"
        history_lines.append(f"{role_name}: {msg.content}")

    return "\n".join(history_lines)


def save_message(db: Session, session_id: int, role: str, content: str):
    message = Message(
        session_id=session_id,
        role=role,
        content=content,
    )
    db.add(message)
    db.commit()


def decide_mode(history_text: str, current_message: str) -> str:
    response = client.chat.completions.create(
        model=openai_deployment,
        messages=[
            {
                "role": "system",
                "content": (
                    "Decide how the assistant should answer the latest user message. "
                    "Return only one word: CHAT or RAG.\n"
                    "Return CHAT when the message should mainly be answered from normal conversation "
                    "or memory in the chat history.\n"
                    "Return RAG when the message is factual, technical, document-based, or a follow-up "
                    "to a factual or technical topic that needs retrieval.\n"
                    "Return only CHAT or RAG."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{history_text}\n\n"
                    f"Latest message:\n{current_message}"
                ),
            },
        ],
    )

    mode = response.choices[0].message.content.strip().upper()
    if mode not in {"CHAT", "RAG"}:
        return "CHAT"
    return mode


def rewrite_search_query(history_text: str, current_message: str) -> str:
    if not history_text:
        return current_message

    response = client.chat.completions.create(
        model=openai_deployment,
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the latest user message into one clear standalone search query "
                    "for Azure AI Search. Use conversation history to resolve references like "
                    "'it', 'that', 'this', and short follow-up questions. "
                    "Do not answer the question. Return only the rewritten search query."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{history_text}\n\n"
                    f"Latest message:\n{current_message}"
                ),
            },
        ],
    )

    rewritten = response.choices[0].message.content.strip()
    return rewritten if rewritten else current_message


def search_documents(search_query: str):
    if not search_service or not search_index or not search_key:
        raise HTTPException(
            status_code=500,
            detail="Azure AI Search environment variables are missing.",
        )

    search_url = (
        f"https://{search_service}.search.windows.net/"
        f"indexes/{search_index}/docs/search?api-version={SEARCH_API_VERSION}"
    )

    search_headers = {
        "Content-Type": "application/json",
        "api-key": search_key,
    }

    embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=search_query,
    )
    query_vector = embedding_response.data[0].embedding

    search_payload = {
        "search": search_query,
        "top": TOP_K,
        "select": "chunk_id,parent_id,chunk,title",
        "queryType": "semantic",
        "semanticConfiguration": "phase3-rag-semantic-configuration",
        "vectorQueries": [
            {
                "kind": "vector",
                "vector": query_vector,
                "fields": "text_vector",
                "k": 50,
            }
        ],
    }

    try:
        search_response = requests.post(
            search_url,
            headers=search_headers,
            json=search_payload,
            timeout=REQUEST_TIMEOUT,
        )

        if not search_response.ok:
            raise HTTPException(
                status_code=500,
                detail=f"Azure AI Search request failed: {search_response.status_code} - {search_response.text}",
            )

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Azure AI Search request failed: {str(exc)}",
        ) from exc

    return search_response.json().get("value", [])


def build_context_and_sources(results) -> Tuple[str, list]:
    context_parts = []
    unique_sources = []
    seen_sources = set()

    for doc in results[:TOP_K]:
        title = doc.get("title", "Unknown source")
        chunk = (doc.get("chunk") or "").strip()

        if title not in seen_sources:
            seen_sources.add(title)
            unique_sources.append(title)

        if chunk:
            context_parts.append(f"Source: {title}\n{chunk}")

    context = "\n\n".join(context_parts)
    return context, unique_sources


def generate_chat_reply(history_text: str, current_message: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Use the conversation history when relevant. "
                "Answer clearly and briefly in 3 sentences maximum."
            ),
        }
    ]

    if history_text:
        messages.append(
            {
                "role": "system",
                "content": f"Conversation history:\n{history_text}",
            }
        )

    messages.append(
        {
            "role": "user",
            "content": current_message,
        }
    )

    response = client.chat.completions.create(
        model=openai_deployment,
        messages=messages,
    )

    return response.choices[0].message.content


def generate_rag_reply(history_text: str, current_message: str, context: str) -> str:
    user_content = f"Question: {current_message}\n\nContext:\n{context}"

    if history_text:
        user_content = (
            f"Conversation history:\n{history_text}\n\n"
            f"Question: {current_message}\n\nContext:\n{context}"
        )

    response = client.chat.completions.create(
        model=openai_deployment,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful RAG assistant. "
                    "Answer clearly and briefly using only the provided context. "
                    "Use conversation history only to understand references and follow-up intent. "
                    "If the answer is not in the context, say you do not know. "
                    "If the user asks for a list, return a short clear list."
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
    )

    return response.choices[0].message.content


def handle_smart_chat(request: ChatRequest, db: Session):
    session = get_or_create_session(db, request.session_id)
    history_text = build_history_text(session)

    mode = decide_mode(history_text, request.message)

    if mode == "RAG":
        search_query = rewrite_search_query(history_text, request.message)
        results = search_documents(search_query)
        context, sources = build_context_and_sources(results)

        if context:
            reply = generate_rag_reply(history_text, request.message, context)
        else:
            reply = generate_chat_reply(history_text, request.message)
            sources = []
            mode = "CHAT"
    else:
        reply = generate_chat_reply(history_text, request.message)
        sources = []

    save_message(db, session.id, "user", request.message)
    save_message(db, session.id, "assistant", reply)

    return {
        "session_id": session.id,
        "mode": mode,
        "reply": reply,
        "sources": sources,
    }


@app.post("/custom-skill/detect-language")
def detect_language_skill(request: SkillRequest):
    response_values = []

    language_map = {
        "ar": "Arabic",
        "en": "English",
        "fr": "French",
        "pt": "Portuguese",
        "es": "Spanish",
        "de": "German",
        "it": "Italian",
        "tr": "Turkish",
    }

    for record in request.values:
        text = record.data.get("text", "").strip()

        try:
            detected_code = detect(text) if text else "unknown"
            language = language_map.get(detected_code, detected_code)
        except Exception:
            language = "Unknown"

        response_values.append(
            {
                "recordId": record.recordId,
                "data": {
                    "detected_language": language
                },
                "errors": None,
                "warnings": None,
            }
        )

    return {"values": response_values}


def call_deployed_custom_skill(texts: List[str]) -> Dict[str, Any]:
    if not azure_custom_skill_url or not azure_custom_skill_key:
        raise HTTPException(
            status_code=500,
            detail="Azure custom skill URL or key is missing from environment variables.",
        )

    payload = {
        "values": [
            {
                "recordId": str(i + 1),
                "data": {"text": text},
            }
            for i, text in enumerate(texts)
        ]
    }

    url = f"{azure_custom_skill_url}?code={azure_custom_skill_key}"

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Azure deployed custom skill request failed: {str(exc)}",
        ) from exc

    return response.json()


@app.post("/test-azure-custom-skill")
def test_azure_custom_skill(request: AzureSkillTestRequest):
    return call_deployed_custom_skill(request.texts)


@app.get("/ui", response_class=HTMLResponse)
def ui_home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "reply": "",
            "sources": [],
            "session_id": "",
            "message": "",
            "detected_languages": [],
            "detect_texts": "",
        },
    )


@app.post("/ui", response_class=HTMLResponse)
def ui_chat(
    request: Request,
    message: str = Form(...),
    session_id: str = Form(""),
    db: Session = Depends(get_db),
):
    parsed_session_id = int(session_id) if session_id.strip() else None

    chat_request = ChatRequest(
        message=message,
        session_id=parsed_session_id,
    )

    result = handle_smart_chat(chat_request, db)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "reply": result["reply"],
            "sources": result["sources"],
            "session_id": result["session_id"],
            "message": message,
            "detected_languages": [],
            "detect_texts": "",
        },
    )


@app.post("/ui/detect-language", response_class=HTMLResponse)
def ui_detect_language(
    request: Request,
    detect_texts: str = Form(...),
):
    texts = [line.strip() for line in detect_texts.splitlines() if line.strip()]
    result = call_deployed_custom_skill(texts)

    detected_languages = []
    for item in result.get("values", []):
        detected_languages.append(
            {
                "recordId": item.get("recordId"),
                "language": item.get("data", {}).get("detected_language", "Unknown"),
            }
        )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "reply": "",
            "sources": [],
            "session_id": "",
            "message": "",
            "detected_languages": detected_languages,
            "detect_texts": detect_texts,
        },
    )


@app.post("/chat")
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    return handle_smart_chat(request, db)


@app.post("/rag-chat")
def rag_chat(request: ChatRequest, db: Session = Depends(get_db)):
    return handle_smart_chat(request, db)