"""Orchestrator Agent Module.
This module contains the OrchestratorAgent class, responsible for
receiving user requests, detecting intent, and delegating the request
to the appropriate system agent (DB, Informative, or MCP).
The orchestrator follows the orchestration pattern to ensure clean
separation of responsibilities between agents.
"""
from Agents.models.intent import Intent
from Agents.models.agent_response import AgentResponse
from Agents.Db_agent import DbAgent
from Agents.informative_agent import InformativeAgent
from Agents.mcp_agent import MCPAgent

class OrchestratorAgent:
    """Central coordinator for routing user requests.The OrchestratorAgent detects the user's intent and delegates
    the request to the appropriate agent:
        - DbAgent for task operations
        - InformativeAgent for knowledge queries
        - MCPAgent for notifications
    """
    def __init__(self):
        self.db_agent = DbAgent()
        self.informative_agent = InformativeAgent()
        self.mcp_agent = MCPAgent()

    def handle_request(self, user_input: str) -> AgentResponse:
        """Processes a user request and delegates it to the correct agent."""
        intent = Intent.from_text(user_input)

        if intent in [Intent.CREATE_TASK, Intent.UPDATE_TASK, Intent.DELETE_TASK]:
          db_response= self.db_agent.handle_request(user_input)
          if db_response.success:
              self.mcp_agent.handle_request(user_input)
          return db_response
        
        elif intent == Intent.KNOWLEDGE_QUERY:
            return self.informative_agent.handle_request(user_input)
        elif intent == Intent.NOTIFY:
            return self.mcp_agent.handle_request(user_input)
        else:
           return AgentResponse( success=False, message="sorry request not understood", agent_name= "orchestrator")
               


