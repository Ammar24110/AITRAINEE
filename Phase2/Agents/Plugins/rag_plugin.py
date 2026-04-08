from typing import TypedDict, Annotated
from semantic_kernel.functions import kernel_function

from Phase2.Agents.models.services.rag_service import RagService


class RAGResponse(TypedDict):
    answer: str
    found: bool


class RAGPlugin:
    """Semantic Kernel plugin for knowledge retrieval using RAG."""

    def __init__(self):
        self.rag_service = RagService()

    @kernel_function
    async def search_knowledge(
        self,
        query: Annotated[str, "User question"]
    ) -> RAGResponse:
        """Search knowledge base and return answer."""

        if not query:
            return {
                "answer": "Query is required",
                "found": False
            }

        answer = self.rag_service.query(query)

        if not answer or str(answer).strip() == "":
            return {
                "answer": "No relevant information found.",
                "found": False
            }

        return {
            "answer": str(answer).strip(),
            "found": True
        }