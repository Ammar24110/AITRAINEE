import os
import faiss
from typing import Optional, List, Dict

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

    dimension = 768
    faiss_index = faiss.IndexFlatL2(dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index


def load_index() -> VectorStoreIndex:
    setup_models()
    vector_store = FaissVectorStore.from_persist_dir(PERSIST_DIR)
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR, vector_store=vector_store)
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


def retrieve(query: str, top_k: int = 3) -> List[Dict]:
    index = get_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)

    results = []
    for n in nodes:
        results.append({
            "score": float(n.score) if n.score is not None else None,
            "text": n.node.get_content(),
            "metadata": dict(n.node.metadata) if n.node.metadata else {},
        })
    return results