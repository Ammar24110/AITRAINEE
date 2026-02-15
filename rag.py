import os
import faiss
from typing import Optional, List, Dict, Any
from llama_index.core.node_parser import SentenceSplitter

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

LLM_MODEL = "gemma3:4b"
EMBED_MODEL = "nomic-embed-text"
PERSIST_DIR = "storage"
DATA_DIR = "Data"

_index: Optional[VectorStoreIndex] = None


def setup_models():
    Settings.llm = Ollama(model=LLM_MODEL, temperature=0)
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)


def build_index() -> VectorStoreIndex:
    setup_models()
    documents = SimpleDirectoryReader(DATA_DIR).load_data()

    splitter = SentenceSplitter(chunk_size=200, chunk_overlap=30)
    nodes = splitter.get_nodes_from_documents(documents)

    dimension = 768
    faiss_index = faiss.IndexFlatIP(dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # ✅ Build index from nodes instead of full documents
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
    idx = get_index(force_rebuild=False)
    retriever = idx.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)

    results: List[Dict[str, Any]] = []
    for item in nodes:
        node = item.node
        score = getattr(item, "score", None)  # now similarity (higher is better)

        text = node.get_content() if hasattr(node, "get_content") else str(node)
        metadata = getattr(node, "metadata", {}) or {}

        results.append({
            "text": text,
            "similarity": float(score) if score is not None else None,
            "metadata": metadata
        })

    # Sort explicitly: highest similarity first
    results.sort(key=lambda r: float("-inf") if r["similarity"] is None else r["similarity"], reverse=True)
    return results

def ask(query: str, top_k: int = 3) -> Dict[str, Any]:
    sources = retrieve(query, top_k=top_k)

    context = "\n\n".join(
        [f"[Source {i+1} | {s['metadata'].get('file_name','unknown')}]\n{s['text']}"
         for i, s in enumerate(sources)]
    )

    prompt = f"""You are a helpful assistant.
Use the context below to answer the question.

Rules:
- If the context contains the answer, answer in 1-2 sentences.
- If the context does NOT contain the answer, say exactly: I don't know based on the provided documents.
- Do NOT refuse if the context clearly mentions the answer.

Context:
{context}

Question: {query}

Answer (1-2 sentences):"""

    setup_models()
    response = Settings.llm.complete(prompt)

    return {
        "query": query,
        "top_k": top_k,
        "answer": str(response),
        "sources": sources
    }