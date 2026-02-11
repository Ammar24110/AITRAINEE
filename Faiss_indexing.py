import os
import faiss

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

def build_or_load_index():
    Settings.llm = Ollama(model=LLM_MODEL, temperature=0)
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)

    #  LOAD (FAISS)
    if os.path.exists(PERSIST_DIR):
        vector_store = FaissVectorStore.from_persist_dir(PERSIST_DIR)
        storage_context = StorageContext.from_defaults(
            persist_dir=PERSIST_DIR,
            vector_store=vector_store,
        )
        return load_index_from_storage(storage_context)

    #  BUILD (FAISS)
    documents = SimpleDirectoryReader("Data").load_data()

    dimension = 768
    faiss_index = faiss.IndexFlatL2(dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index

def main():
    index = build_or_load_index()
    query_engine = index.as_query_engine(similarity_top_k=3)

    print("FAISS Index ready. Ask questions (type 'exit' to quit).")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() == "exit":
            break
        answer = query_engine.query(q)
        print("\nAssistant:", answer.response)

if __name__ == "__main__":
    main()