import ollama

MODEL = "gemma3:4b"

print(f"Local LLM chat using Ollama | model={MODEL}")
print("Type 'exit' to quit.\n")

while True:
    user_text = input("You: ").strip()

    if user_text.lower() == "exit":
        break

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": user_text}],
    )

    print("\nAssistant:", response["message"]["content"], "\n")