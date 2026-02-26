"""This module defines the task data structure shared across the DB Agent,Orchestrator Agent, and MCP Agent."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
@dataclass
class Task:
    "represents a single task that is assigned to the employee"
    task_id: int
    person_assigned: str
    title:str
    description: Optional[str] = None
    status: str="pending"
    due_date: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


    def mark_done(self) -> None:
      self.status = "done"
      self.updated_at = datetime.now()


    def update_date(self, new_due_date: str) -> None:
       self.due_date= new_due_date
       self.updated_at= datetime.now()
    
    def to_dict(self):
       """Converts the Task object into a dictionary representation."""
       return("task_id": self.task_id, "person_assigned": self.person_assigned,"title": self.title,"description": self.description,"status": self.status,"due_date": self.due_date,"created_at": self.created_at.isoformat(),"updated_at": self.updated_at.isoformat())