from Phase2.Agents.models.intent import Intent
from Phase2.Agents.models.agent_response import AgentResponse
from Phase2.Agents.models.task import Task
from typing import Optional
from Phase2.Agents.models.services.task_repository import TaskRepository

class DbAgent:
    """Handles create, update, and delete operations for tasks."""

    def __init__(self):
        self.repo = TaskRepository()
        self.agent_name = "db_agent"
        self._next_id = 1

    def handle_request(self,intent: Intent) -> AgentResponse:
        if intent.name == "CREATE_TASK":
           return self._create_task(intent)

        elif intent.name == "UPDATE_TASK":
           return self._update_task(intent)

        elif intent.name == "DELETE_TASK":
           return self._delete_task(intent)
        elif intent.name == "LIST_TASKS":
           return self._list_tasks(intent)
        else:
           return AgentResponse(
    success=False,
    message="Unsupported DB operation",
    agent_name="db_agent"
)
        
    def _create_task(self, intent: Intent) -> AgentResponse:
        """Creates a new task."""

        title = intent.params.get("title")
        person_assigned = intent.params.get("person_assigned")
        description = intent.params.get("description")
        due_date = intent.params.get("due_date")

        if not title:
            return AgentResponse(success=False, message="Title is required", agent_name="db_agent")

        if not person_assigned:
            return AgentResponse(success=False, message="Person assigned is required", agent_name="db_agent")

        task = Task(
            task_id=self._next_id,
            person_assigned=person_assigned,
            title=title,
            description=description,
            due_date=due_date,
        )

        self.repo.add_task(task)
        self._next_id += 1

        return AgentResponse(
            success=True,
            message="Task created successfully",
            agent_name="db_agent",
            data={"task_id": task.task_id},
            )
    def _update_task(self, intent: Intent) -> AgentResponse:
        """Updates an existing task."""

        task_id = intent.params.get("task_id")
        if not task_id:
            return AgentResponse(success=False, message="Task ID is required", agent_name="db_agent")

        task = self.repo.get_task(task_id)
        if not task:
            return AgentResponse(success=False, message="Task not found", agent_name="db_agent")

        # Update fields if provided
        if "title" in intent.params:
            task.title = intent.params["title"]

        if "description" in intent.params:
            task.description = intent.params["description"]

        if "due_date" in intent.params:
            task.update_date(intent.params["due_date"])

        if "status" in intent.params:
            task.status = intent.params["status"]

        self.repo.update_task(task)

        return AgentResponse(
            success=True,
            message="Task updated successfully",
            agent_name="db_agent", 
            data={"task_id": task.task_id},
        )
    def _delete_task(self, intent: Intent) -> AgentResponse:
        """Deletes a task."""

        task_id = intent.params.get("task_id")
        if not task_id:
            return AgentResponse(success=False, message="Task ID is required", agent_name="db_agent")

        task = self.repo.get_task(task_id)
        if not task:
            return AgentResponse(success=False, message="Task not found", agent_name="db_agent")

        self.repo.delete_task(task_id)

        return AgentResponse(
            success=True,
            message="Task deleted successfully", agent_name="db_agent",
            data={"task_id": task_id},
        )
    def _list_tasks(self, intent: Intent) -> AgentResponse:
        """Returns all tasks (optionally filtered by person)."""

        person_assigned: Optional[str] = intent.params.get("person_assigned")
        tasks = self.repo.get_all_tasks()

        if person_assigned:
            tasks = [t for t in tasks if t.person_assigned == person_assigned]

        return AgentResponse(
            success=True,
            message="Tasks retrieved successfully", agent_name="db_agent",
            data=[task.to_dict() for task in tasks],
        )
       