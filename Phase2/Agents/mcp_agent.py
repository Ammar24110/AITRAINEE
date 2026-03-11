from Phase2.Agents.models.agent_response import AgentResponse
from Phase2.Agents.models.intent import Intent
from Phase2.Agents.models.services.notification_service import NotificationService

class MCPAgent:

    def __init__(self):
        self.notification_service = NotificationService()


    def handle_request(self, intent: Intent) -> AgentResponse:
        
        title = intent.params.get("title")
        person = intent.params.get("person_assigned")

        if intent.name in ["CREATE_TASK", "UPDATE_TASK"]:
            self.notification_service.send_notification("intent is related to tasks")
        else:
            return AgentResponse (success= False, message=" no notification needed",agent_name="mcp_agent")

        return AgentResponse(success=True, message="Notification sent",agent_name="mcp_agent")