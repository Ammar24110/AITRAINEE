from typing import List, Optional
from Phase2.Agents.models.task import Task


class TaskRepository:

    def __init__(self):
        self._tasks: dict[int, Task] = {}

    def add_task(self, task: Task) -> Task:
        self._tasks[task.task_id] = task
        return task

    def get_task(self, task_id: int) -> Optional[Task]:
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> List[Task]:
       return list(self._tasks.values())

    def update_task(self, task: Task) -> Optional[Task]:
        if task.task_id in self._tasks:
            self._tasks[task.task_id] = task
            return task
        return None

    def delete_task(self, task_id: int) -> bool:
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False

    def get_tasks_by_person(self, person: str) -> List[Task]:
        return [
            task for task in self._tasks.values()
            if task.person_assigned
            and task.person_assigned.lower() == person.lower()
        ]