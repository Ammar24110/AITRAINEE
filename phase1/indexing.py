from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

LLM_MODEL = "gemma3:4b"
EMBED_MODEL = "nomic-embed-text"

def main():
    Settings.llm = Ollama(model=LLM_MODEL, temperature=0)
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)

    documents = SimpleDirectoryReader("Data").load_data()
    index = VectorStoreIndex.from_documents(documents)

    qa_template = PromptTemplate(
        "Answer the question using ONLY the context below.\n"
        "If the answer is not in the context, say: Not in document.\n\n"
        "Context:\n"
        "{context_str}\n\n"
        "Question: {query_str}\n"
        "Answer:"
    )

    query_engine = index.as_query_engine(
        similarity_top_k=3,
        response_mode="compact",
        text_qa_template=qa_template
    )

    print("Index built. Ask questions (type 'exit' to quit).")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() == "exit":
            break

        answer = query_engine.query(q)
        print("\nAssistant:", answer.response)

if __name__ == "__main__":
    main()