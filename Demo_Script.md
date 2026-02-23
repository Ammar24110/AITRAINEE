

1) Introduction

Retrieval-Augmented Generation (RAG) system built using FastAPI, FAISS, and a local LLM running through Ollama.

The system supports document indexing, semantic retrieval, grounded answer generation, and session-based conversational memory.

---

2) Indexing
I rebuild the index using the `/index` endpoint.

- Documents are loaded from the dataset folder.
- Each document is split into chunks using SentenceSplitter with overlap.
- Embeddings are generated using the `nomic-embed-text` model.
- The embeddings are stored inside a FAISS vector index.
- The index is persisted locally for reuse.

---

3) Retrieval
I test semantic search using the `/search` endpoint.

- The user query is converted into an embedding.
- FAISS performs similarity search using L2 distance.
- The system retrieves the most relevant chunks.
- Metadata such as file name and similarity score is returned.

Lower L2 distance indicates higher similarity.

---

4) RAG Generation

Now I demonstrate grounded answer generation using the `/chat` endpoint.

- The system retrieves top-k relevant chunks.
- Retrieved chunks are injected into a structured prompt.
- The LLM generates an answer strictly based on retrieved context.
- Sources are returned for transparency.

This reduces hallucinations and improves factual accuracy.

---

5) Session Memory

The system supports session-based memory.

- Each session has a unique session_id.
- Conversation history is stored and summarized.
- Personal facts are prioritized from chat history.
- This enables multi-turn conversational continuity.

---

6) Performance

Performance logs show:

- Index loading time
- Retrieval time
- Total processing time

Repeated queries are faster due to caching.

---

7) Architecture Overview

The system architecture includes:

1. Document Loader
2. Embedding Model (Ollama)
3. FAISS Vector Store
4. LLM Generator
5. FastAPI Endpoints

Retriever and generator are decoupled following standard RAG architecture.

---

8) Conclusion

This demonstrates a complete end-to-end RAG pipeline including:

- Multi-dataset indexing
- Semantic retrieval
- Grounded generation
- Conversational memory
- Performance optimization

The system runs fully locally using FastAPI and Ollama.
