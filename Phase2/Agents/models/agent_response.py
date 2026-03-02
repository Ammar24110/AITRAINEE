""""This module contains the AgentResponse data structure used by theOrchestrator, DB Agent,
 Informative Agent, and MCP Agent to ensureconsistent communication between components."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class Agentresponse:
    """Represents a standardized response returned by system agents."""

    success: bool
    message: str
    agent_name: str
    data: Optional[Dict[str, Any]] = None


    def to_dict(self):

        return({
            "success": self.success,
            "message": self.message,
            "agent_name": self.agent_name,
            "data": self.data
        })
        
