from typing import Annotated
from semantic_kernel.functions import kernel_function

from Phase2.Agents.models.services.task_repository import TaskRepository
from Phase2.Agents.models.task import Task


class TaskPlugin:
    """Semantic Kernel plugin for task management."""

    def __init__(self, repo: TaskRepository):
        self.repo = repo
        self.counter = 1
        self.last_action_message = ""
        self.last_action_type = ""

    @kernel_function
    async def create_task(
      self,
      title: Annotated[str, "Task title"],
      person_assigned: Annotated[str, "Person assigned to the task"]) -> str:
      """Create a new task."""

      if not title or not person_assigned:
        self.last_action_message = ""
        self.last_action_type = ""
        return "Invalid task input"

      existing_tasks = self.repo.get_tasks_by_person(person_assigned)

      for t in existing_tasks:
        if t.title.strip().lower() == title.strip().lower():
            self.last_action_message = ""
            self.last_action_type = ""
            return f"Task already exists for {person_assigned}: {title}"

      task = Task(
        task_id=self.counter,
        title=title,
        person_assigned=person_assigned,
        status="pending"
    )

      self.counter += 1
      self.repo.add_task(task)

      message = f"Task created for {person_assigned}: {title}"
      self.last_action_message = message
      self.last_action_type = "create"

      return message

    @kernel_function
    async def list_tasks(
      self,
      person_assigned: Annotated[str, "Person to filter tasks"] = None) -> str:
      """List tasks (optionally filtered by person)."""

      self.last_action_message = ""
      self.last_action_type = ""

      if person_assigned:
        tasks = self.repo.get_tasks_by_person(person_assigned)
      else:
        tasks = self.repo.get_all_tasks()  

      if not tasks:
        return "No tasks found."

      return "\n".join(
        f"{t.task_id}: {t.title} ({t.status}) - {t.person_assigned}"
        for t in tasks
    )

    @kernel_function
    async def update_task(
        self,
        task_id: Annotated[int, "Task ID"],
        title: Annotated[str, "New title"]
    ) -> str:
        """Update task title."""

        task = self.repo.get_task(task_id)

        if not task:
            self.last_action_message = ""
            self.last_action_type = ""
            return "Task not found"

        task.title = title
        self.repo.update_task(task)

        message = f"Task updated {task_id}: {title}"
        self.last_action_message = message
        self.last_action_type = "update"

        return message

    @kernel_function
    async def delete_task(
        self,
        task_id: Annotated[int, "Task ID"]
    ) -> str:
        """Delete a task."""

        success = self.repo.delete_task(task_id)

        if not success:
            self.last_action_message = ""
            self.last_action_type = ""
            return f"Task not found: {task_id}"

        message = f"Task deleted {task_id}"
        self.last_action_message = message
        self.last_action_type = "delete"

        return message