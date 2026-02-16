from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import rag
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from db import init_db, get_db, ChatSessionModel, Message, RetrievedContext

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