from typing import Dict, Any
from phase1.rag import chat, get_index

class RagService:

    def __init__(self):
        self.index =get_index()

    def query(self, query: str, history_text: str = "") -> str:
        result = chat(query, history_text)

        return result["answer"]