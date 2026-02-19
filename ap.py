from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import rag
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from db import init_db, get_db, ChatSessionModel, Message, RetrievedContext
SUMMARY_TRIGGER = 5   # start summarizing after this many messages in session
SUMMARY_CHUNK = 10     # summarize this many old messages at a time
RECENT_KEEP = 6        # keep last 6 messages as "recent"
app = FastAPI(title="Milestone 3 - Data Preparation and Indexing")

init_db()
class IndexRequest(BaseModel):
    rebuild: bool = True


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

class AskRequest(BaseModel):
    query: str
    top_k: int = 3
    session_id: int | None = None

class ChatRequest(BaseModel):
    message: str
    top_k: int = 3
    session_id: int | None = None
    history_k: int = 6  # last 6 messages


@app.get("/")
def health():
    return {"status": "ok", "message": "RAG API running"}


@app.post("/index")
def index_docs(body: IndexRequest):
    try:
        rag.get_index(force_rebuild=body.rebuild)
        return {"status": "indexed", "rebuild": body.rebuild, "persist_dir": rag.PERSIST_DIR}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def search_docs(body: SearchRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        results = rag.retrieve(body.query, top_k=body.top_k)
        return {"query": body.query, "top_k": body.top_k, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/ask")
def ask_llm(body: AskRequest, db: Session = Depends(get_db)):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # 1) Create or load session
    if body.session_id is None:
        chat_session = ChatSessionModel(title=body.query[:50])
        db.add(chat_session)
        db.commit()
        db.refresh(chat_session)
    else:
        chat_session = db.query(ChatSessionModel).filter(ChatSessionModel.id == body.session_id).first()
        if not chat_session:
            raise HTTPException(status_code=404, detail="Session not found")

    # 2) Save user message
    user_msg = Message(
        session_id=chat_session.id,
        role="user",
        content=body.query
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    # 3) Run RAG ask() to get answer + sources
    result = rag.ask(body.query, top_k=body.top_k)

    # 4) Save assistant message
    assistant_msg = Message(
        session_id=chat_session.id,
        role="assistant",
        content=result["answer"]
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    
    for i, src in enumerate(result["sources"], start=1):
        ctx = RetrievedContext(
            message_id=assistant_msg.id,
            source_index=i,
            text=src.get("text"),
            similarity=src.get("similarity"),
            source_metadata=src.get("metadata")  
        )
        db.add(ctx)

    db.commit()

    return {
        "session_id": chat_session.id,
        **result
    }
def format_history(msgs):
    lines = []
    for m in msgs:
        if m.role == "user":          # ✅ only user messages
            lines.append(f"USER: {m.content}")
    return "\n".join(lines)

def is_question(text: str) -> bool:
    t = text.strip().lower()
    if t.endswith("?"):
        return True
    starters = ("what", "why", "how", "when", "where", "who", "which", "do ", "does ", "did ", "can ", "could ", "is ", "are ", "am ", "will ", "would ")
    return t.startswith(starters)

def is_statement(text: str) -> bool:
    t = text.strip().lower()
    if t.endswith("?"):
        return False
    starters = (
        "my name is", "i am", "i'm", "i like", "i love", "i live", "i work",
        "my favorite", "my favourite", "i prefer"
    )
    return t.startswith(starters)
@app.post("/chat")
def chat_endpoint(body: ChatRequest, db: Session = Depends(get_db)):
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # 1) Create or load session
    if body.session_id is None:
        chat_session = ChatSessionModel(title=body.message[:50])
        db.add(chat_session)
        db.commit()
        db.refresh(chat_session)
    else:
        chat_session = db.query(ChatSessionModel).filter(ChatSessionModel.id == body.session_id).first()
        if not chat_session:
            raise HTTPException(status_code=404, detail="Session not found")

    # 2) Fetch history BEFORE saving the new user message
    # 2) Fetch recent history (for context)
    recent_msgs = (
    db.query(Message)
    .filter(Message.session_id == chat_session.id)
    .order_by(Message.created_at.desc())
    .limit(body.history_k)
    .all()
)
    recent_msgs = list(reversed(recent_msgs))
    recent_text = format_history(recent_msgs)

    if (chat_session.summary or "").strip():
       history_text = f"SUMMARY:\n{chat_session.summary}\n\nRECENT:\n{recent_text}"
    else:
      history_text = recent_text

    result = rag.chat(body.message, history_text=history_text, top_k=body.top_k)
    # 3) Save user message
    user_msg = Message(session_id=chat_session.id, role="user", content=body.message)
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)
    # If it's NOT a question, just acknowledge and skip RAG
    # If it's NOT a question, just acknowledge and skip RAG
    if not is_question(body.message):
        assistant_text = "Got it. I’ll remember that."
        assistant_msg = Message(session_id=chat_session.id, role="assistant", content=assistant_text)
        db.add(assistant_msg)
        db.commit()
        db.refresh(assistant_msg)

        return {
           "session_id": chat_session.id,
           "answer": assistant_text,
           "sources": []
    }
    # 4) Add the NEW user message to history_text (so model sees it in history too)
    if history_text.strip():
        history_text = history_text + f"\nUSER: {body.message}"
    else:
        history_text = f"USER: {body.message}"

    # 5) Run RAG + history
    result = rag.chat(body.message, history_text=history_text, top_k=body.top_k)

    # 6) Save assistant message
    assistant_msg = Message(session_id=chat_session.id, role="assistant", content=result["answer"])
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)
        # --- 3A) Summarize old history if session is long ---
    total_msgs = db.query(Message).filter(Message.session_id == chat_session.id).count()

    if total_msgs > SUMMARY_TRIGGER:
        # last RECENT_KEEP messages (we never summarize these)
        recent = (
            db.query(Message)
            .filter(Message.session_id == chat_session.id)
            .order_by(Message.created_at.desc())
            .limit(RECENT_KEEP)
            .all()
        )
        recent_ids = {m.id for m in recent}

        # candidates for summarization: not summarized yet + not in recent
        candidates = (
            db.query(Message)
            .filter(
                Message.session_id == chat_session.id,
                Message.id > chat_session.summary_last_message_id
            )
            .order_by(Message.created_at.asc())
            .all()
        )
        candidates = [m for m in candidates if m.id not in recent_ids]

        chunk = candidates[:SUMMARY_CHUNK]
        if chunk:
            chunk_text = format_history(chunk)
            chat_session.summary = rag.summarize_history(chat_session.summary or "", chunk_text)
            chat_session.summary_last_message_id = chunk[-1].id
            db.commit()
    # 7) Save retrieved context (linked to assistant message)
    for i, src in enumerate(result["sources"], start=1):
        ctx = RetrievedContext(
            message_id=assistant_msg.id,
            source_index=i,
            text=src.get("text"),
            similarity=src.get("similarity"),
            source_metadata=src.get("metadata"),
        )
        db.add(ctx)

    db.commit()

    return {
        "session_id": chat_session.id,
        "answer": result["answer"],
        "sources": result["sources"],
    }

@app.get("/history/{session_id}")
def get_history(session_id: int, db: Session = Depends(get_db)):
    chat_session = db.query(ChatSessionModel).filter(ChatSessionModel.id == session_id).first()
    if not chat_session:
        raise HTTPException(status_code=404, detail="Session not found")

    msgs = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .all()
    )

    return {
        "session_id": session_id,
        "title": chat_session.title,
        "summary": getattr(chat_session, "summary", ""), 
        "messages": [
            {"id": m.id, "role": m.role, "content": m.content, "created_at": m.created_at}
            for m in msgs
        ]
    }

