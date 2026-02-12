from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import rag

app = FastAPI(title="Milestone 3 - Data Preparation and Indexing")


class IndexRequest(BaseModel):
    rebuild: bool = True


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


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