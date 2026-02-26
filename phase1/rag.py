import os
import faiss
from typing import Optional, List, Dict, Any
from llama_index.core.node_parser import SentenceSplitter
import time
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
_RETRIEVE_CACHE: dict[tuple[str, int], List[Dict[str, Any]]] = {}
LLM_MODEL = "gemma3:4b"
EMBED_MODEL = "nomic-embed-text"
PERSIST_DIR = os.getenv("PERSIST_DIR", "storage")
DATA_DIR = os.getenv("DATA_DIR", "datasets/Data/data_a")
PROMPT_MODE = "fewshot"  # options: "basic", "strict", "fewshot"

_index: Optional[VectorStoreIndex] = None
_EMBED_DIM: int | None = None

def setup_models():
    Settings.llm = Ollama(model=LLM_MODEL, temperature=0)
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)

def get_embed_dim() -> int:
    global _EMBED_DIM
    if _EMBED_DIM is not None:
        return _EMBED_DIM
    setup_models()
    vec = Settings.embed_model.get_text_embedding("dimension test")
    _EMBED_DIM = len(vec)
    return _EMBED_DIM

def is_personal_query(q: str) -> bool:
    q = (q or "").lower()
    triggers = [
        "my ", "i ", "i'm", "im ",
        "me ", "mine",
        "my name", "what is my name",
        "where do i live", "where am i from",
        "what do i like", "what do i love",
        "remember", "do you remember"
    ]
    return any(t in q for t in triggers)
def build_index() -> VectorStoreIndex:
    setup_models()
    documents = SimpleDirectoryReader(DATA_DIR).load_data()

    splitter = SentenceSplitter(chunk_size=350, chunk_overlap=70)
    nodes = splitter.get_nodes_from_documents(documents)

    dimension = get_embed_dim()
    faiss_index = faiss.IndexFlatL2(dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    #  Build index from nodes instead of full documents
    index = VectorStoreIndex(nodes, storage_context=storage_context)

    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index

def load_index() -> VectorStoreIndex:
    setup_models()
    vector_store = FaissVectorStore.from_persist_dir(PERSIST_DIR)
    storage_context = StorageContext.from_defaults(
        persist_dir=PERSIST_DIR,
        vector_store=vector_store
    )
    return load_index_from_storage(storage_context)


def get_index(force_rebuild: bool = False) -> VectorStoreIndex:
    global _index

    if force_rebuild:
        _index = build_index()
        return _index

    if _index is not None:
        return _index

    if os.path.exists(PERSIST_DIR):
        _index = load_index()
    else:
        _index = build_index()

    return _index


def retrieve(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    key = (query.strip().lower(), top_k)
    if key in _RETRIEVE_CACHE:
       return _RETRIEVE_CACHE[key]
    t0 = time.perf_counter()

    idx = get_index(force_rebuild=False)
    t1 = time.perf_counter()

    retriever = idx.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    t2 = time.perf_counter()

    print(f"[PERF] get_index={(t1-t0)*1000:.1f}ms | retrieve={(t2-t1)*1000:.1f}ms | total={(t2-t0)*1000:.1f}ms")

    results: List[Dict[str, Any]] = []
    for item in nodes:
        node = item.node
        score = getattr(item, "score", None)  # for L2, lower is better

        text = node.get_content() if hasattr(node, "get_content") else str(node)
        metadata = getattr(node, "metadata", {}) or {}

        results.append({
            "text": text,
            "similarity": float(score) if score is not None else None,
            "metadata": metadata
        })

    # For L2: lower is better
    results.sort(key=lambda r: float("inf") if r["similarity"] is None else r["similarity"])
    _RETRIEVE_CACHE[key] = results
    return results


def ask(query: str, history_text: str = "", top_k: int = 3) -> Dict[str, Any]:
    sources = retrieve(query, top_k=top_k)

    context = "\n\n".join(
        [f"[Source {i+1} | {s['metadata'].get('file_name','unknown')}]\n{s['text']}"
         for i, s in enumerate(sources)]
    )

    prompt = f"""
You are a conversational assistant.

You have TWO sources:
1) CHAT HISTORY (user personal info like name, preferences, previous messages).
2) DOCUMENT CONTEXT (retrieved chunks from files).

RULES:
- If the answer is in CHAT HISTORY, answer using CHAT HISTORY.
- Else if the answer is in DOCUMENT CONTEXT, answer using DOCUMENT CONTEXT.
- Else say exactly: I don't know based on the provided history and documents.
- Keep the answer to 1-2 sentences.

EXAMPLES:

Example 1:
CHAT HISTORY: User: My name is Ammar.
DOCUMENT CONTEXT: (empty)
QUESTION: What is my name?
ANSWER: Your name is Ammar.

Example 2:
CHAT HISTORY: User: I like football.
DOCUMENT CONTEXT: (empty)
QUESTION: What do I like?
ANSWER: You like football.

Example 3:
CHAT HISTORY: (empty)
DOCUMENT CONTEXT: FastAPI is a Python framework for building APIs.
QUESTION: What is FastAPI?
ANSWER: FastAPI is a Python framework for building APIs.

Example 4:
CHAT HISTORY: (empty)
DOCUMENT CONTEXT: API stands for Application Programming Interface.
QUESTION: What does SQL stand for?
ANSWER: I don't know based on the provided history and documents.

NOW YOUR TURN:

CHAT HISTORY:
{history_text}

DOCUMENT CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
    setup_models()
    response = Settings.llm.complete(prompt)

    return {
        "query": query,
        "top_k": top_k,
        "answer": str(response),
        "sources": sources
    }

def chat(query: str, history_text: str, top_k: int = 3) -> Dict[str, Any]:
    sources = retrieve(query, top_k=top_k)

    context = "\n\n".join(
        [f"[Source {i+1} | {s['metadata'].get('file_name','unknown')}]\n{s['text']}"
         for i, s in enumerate(sources)]
    )

    prompt = f"""
You are a conversational assistant.

You have TWO sources:
1) CHAT HISTORY (facts the user said before).
2) DOCUMENT CONTEXT (retrieved chunks from files).

RULES:

1) Determine the question type first.

2) If the question is about personal information 
   (name, age, location, preferences, relationships, past statements),
   answer ONLY using CHAT HISTORY.
   - Do NOT use DOCUMENT CONTEXT.
   - Rewrite the answer cleanly.
   - Convert "I ..." to "You ..." when appropriate.

3) If the question is technical, factual, or about external knowledge,
   IGNORE CHAT HISTORY completely.
   Answer ONLY using DOCUMENT CONTEXT.

4) Never mix CHAT HISTORY and DOCUMENT CONTEXT in the same answer.

5) If the answer is not found in the correct source,
   say exactly:
   I don't know based on the provided history and documents.

6) Keep the answer to 1-2 sentences.
7) NEVER mention chat history unless the question is explicitly about the user (name/location/preferences).
8)NEVER start an answer with a personal fact unless the question is personal.
EXAMPLES:

Example 1:
CHAT HISTORY:
User: My name is Ammar.
QUESTION: What is my name?
ANSWER: Your name is Ammar.

Example 2:
CHAT HISTORY:
User: I like football.
QUESTION: What do I like?
ANSWER: You like football.

Example 3:
CHAT HISTORY:
User: I live in Qatar.
QUESTION: Where do I live?
ANSWER: You live in Qatar.

Example 4:
CHAT HISTORY: (empty)
DOCUMENT CONTEXT: FastAPI is a Python framework for building APIs.
QUESTION: What is FastAPI?
ANSWER: FastAPI is a Python framework for building APIs.

NOW YOUR TURN:

CHAT HISTORY:
{history_text}

DOCUMENT CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
    if not is_personal_query(query):
      history_text = ""
    setup_models()
    response = Settings.llm.complete(prompt)

    return {
        "query": query,
        "top_k": top_k,
        "answer": str(response).strip(),
        "sources": sources
    }

def summarize_text(old_summary: str, msgs: list[str]) -> str:
    setup_models()

    chunk_text = "\n".join(msgs).strip()

    prompt = f"""
You are summarizing chat history for memory.

RULES:
- Update the EXISTING SUMMARY using the NEW MESSAGES.
- Keep it short (max 6 bullet points).
- Only keep stable user facts/preferences and important decisions.
- Do NOT invent anything.

EXISTING SUMMARY:
{old_summary}

NEW MESSAGES:
{chunk_text}

UPDATED SUMMARY:
"""

    resp = Settings.llm.complete(prompt)
    return str(resp).strip()

def summarize_history(old_summary: str, new_chunk: str) -> str:
    """
    Returns an updated running summary that includes new_chunk.
    Uses your local LLM (same one you use in rag.chat).
    """
    old_summary = (old_summary or "").strip()
    new_chunk = (new_chunk or "").strip()

    if not new_chunk:
        return old_summary

    prompt = f"""
You are maintaining a SHORT memory summary for a chat session.

Current summary (may be empty):
{old_summary if old_summary else "(empty)"}

New conversation chunk to incorporate:
{new_chunk}

Update the summary with the new information.
Rules:
- Keep it concise (5-10 bullet points max).
- Keep stable facts (name, preferences, dislikes, goals).
- Remove repetition.
- If something is corrected later, keep the latest version.

Return ONLY the updated summary text.
""".strip()

    setup_models()
    resp = Settings.llm.complete(prompt)
    return str(resp).strip()