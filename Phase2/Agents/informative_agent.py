from Phase2.Agents.models.agent_response import AgentResponse
from Phase2.Agents.models.intent import Intent
from Phase2.Agents.models.services.rag_service import RagService
from typing import Optional

class InformativeAgent:
    def __init__(self):
        self.rag_service = RagService()


    def handle_request(self, intent: Intent) -> AgentResponse:
        query = intent.params.get("query")
        if query is None:
            return AgentResponse(success=False,message="query is required",agent_name="informative_agent", data= None)
        answer = self.rag_service.query(query)
        return AgentResponse( success= True, message= "knowledge gained",agent_name="informative_agent", data= answer) 