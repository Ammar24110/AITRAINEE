import ollama

response = ollama.chat(
    model="gemma3:4b",
    messages=[
        {"role": "user", "content": "Explain Integration in simple words"}
    ]
)

print(response["message"]["content"])