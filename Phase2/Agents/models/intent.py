from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Intent:
    name: str
    params: Dict[str, Any]

    @staticmethod
    def from_text(text: str):
        """Simple rule-based intent detection."""

        text_lower = text.lower()

        # CREATE TASK
        if "add" in text_lower or "create" in text_lower:

            person = None
            title = text

            if "for " in text_lower:
                try:
                    person = text.split("for ")[1].split(":")[0].strip()
                except:
                    person = None

            if ":" in text:
                title = text.split(":")[1].strip()

            return Intent(
                name="CREATE_TASK",
                params={
                    "title": title,
                    "person_assigned": person
                }
            )

        # UPDATE TASK
        if "update" in text_lower:
            return Intent(
                name="UPDATE_TASK",
                params={"title": text}
            )

        # LIST TASKS
        if "list" in text_lower:

            person = None

            if "for " in text_lower:
                try:
                    person = text.split("for ")[1].strip()
                except:
                    person = None

            return Intent(
                name="LIST_TASKS",
                params={
                    "person_assigned": person
                }
            )

        # KNOWLEDGE QUERY (RAG)
        return Intent(
            name="KNOWLEDGE_QUERY",
            params={"query": text}
        )