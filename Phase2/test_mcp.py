import requests

# Step 1: create session
session_id = "test123"

# Step 2: send tool call
url = f"http://localhost:8002/messages/?session_id={session_id}"

payload = {
    "message": {
        "content": {
            "tool_name": "send_notification",
            "arguments": {
                "message": "🔥 TEST EMAIL FROM MCP 🔥"
            }
        }
    }
}

response = requests.post(url, json=payload)

print("STATUS:", response.status_code)
print("RESPONSE:", response.text)