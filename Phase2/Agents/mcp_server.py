from fastmcp import FastMCP
from Phase2.Agents.models.services.notification_service import NotificationService

import sys
sys.stdout.reconfigure(encoding='utf-8')

mcp = FastMCP("mcp")

notification_service = NotificationService()

@mcp.tool(
    name="send_notification",
    description="Send email notification after task operations. Input must be one plain text message string."
)
def send_notification(message: str) -> dict:
    print("MCP TOOL RECEIVED:", message)

    if not message or str(message).strip() == "":
        return {
            "success": False,
            "message": "Message is required"
        }

    return notification_service.send_notification(message.strip())


if __name__ == "__main__":
    print("MCP SERVER STARTED")
    mcp.run(transport="sse", port=8002)