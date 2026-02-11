import ollama

resp = ollama.chat(
    model="llama3",
    messages=[{"role": "user", "content": "Explain RAG in one sentence."}]
)

print(resp["message"]["content"])
