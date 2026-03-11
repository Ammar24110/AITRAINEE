"""Orchestrator Agent Module.
This module contains the OrchestratorAgent class, responsible for
receiving user requests, detecting intent, and delegating the request
to the appropriate system agent (DB, Informative, or MCP).
The orchestrator follows the orchestration pattern to ensure clean
separation of responsibilities between agents.
"""
from Phase2.Agents.models.intent import Intent
from Phase2.Agents.models.agent_response import AgentResponse
from Phase2.Agents.Db_agent import DbAgent
from Phase2.Agents.informative_agent import InformativeAgent
from Phase2.Agents.mcp_agent import MCPAgent

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

        if intent.name in ["CREATE_TASK", "UPDATE_TASK", "DELETE_TASK", "LIST_TASKS"]:
          db_response = self.db_agent.handle_request(intent)
          if db_response.success:
              self.mcp_agent.handle_request(intent)
          return db_response
        
        elif intent.name == "KNOWLEDGE_QUERY":
            return self.informative_agent.handle_request(intent)
        elif intent.name == "NOTIFY":
            return self.mcp_agent.handle_request(intent)
        else:
           return AgentResponse( success=False, message="sorry request not understood", agent_name= "orchestrator")
               


