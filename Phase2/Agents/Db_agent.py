from Phase2.Agents.models.intent import Intent
from Phase2.Agents.models.agent_response import AgentResponse
from Phase2.Agents.models.task import Task
from Phase2.Agents.models.services.task_repository import TaskRepository

class DbAgent:
    """Handles create, update, and delete operations for tasks."""

    def __init__(self):
        self.repo = TaskRepository()
        self.agent_name = "db_agent"


    def handle_request(self,intent: Intent) -> AgentResponse:
        if intent.name == "CREATE_TASK":
           return self._create_task(intent)

        elif intent.name == "UPDATE_TASK":
           return self._update_task(intent)

        elif intent.name == "DELETE_TASK":
           return self._delete_task(intent)

        else:
           return AgentResponse("error")
        
    def _create_task(self, intent)-> AgentResponse:
       title = intent.params.get("title")
       if not title:
          return AgentResponse("Title is required")
       task= Task(task_id)
       self.repo.add_task(task)
       return AgentResponse(success=True, data=task)
       