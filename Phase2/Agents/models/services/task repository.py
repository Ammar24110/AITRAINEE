from typing import Dict, List, Optional
from Agents.models.task import Task

class TaskRepository:
    """In-memory repository for storing and managing Task objects."""
    def __init__(self):
        """Initializes the internal task storage dictionary."""
        self._tasks = {}

    def add_task(self,task) -> Task:
        """Adds a new task to the repository and returns it."""

        self._tasks[task.task_id] = task
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieves a task by its ID if it exists."""

        return self._tasks.get(task_id)

    def get_all_tasks(self) -> List[Task]:
        """Returns a list of all stored tasks."""

        return list(self._tasks.values())