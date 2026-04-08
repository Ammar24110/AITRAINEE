from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

from Phase2.Agents.Plugins.rag_plugin import RAGPlugin


class InformativeAgent:
    def __init__(self) -> None:
        self.kernel = Kernel()

        self.chat_completion = AzureChatCompletion(
    deployment_name="gpt-5.4-mini",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview"
)

        self.kernel.add_service(self.chat_completion)

        self.rag_plugin = RAGPlugin()

        self.kernel.add_plugin(self.rag_plugin, plugin_name="rag")

        self.history = ChatHistory()

        self.settings = AzureChatPromptExecutionSettings()
        self.settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    async def handle_request(self, user_input: str) -> str:
        system_prompt = """
You are a Knowledge Assistant.

Rules:
- ALWAYS use the rag.search_knowledge tool for answering questions
- NEVER answer from your own knowledge
- ONLY return tool results
"""

        if len(self.history.messages) == 0:
            self.history.add_system_message(system_prompt)

        self.history.add_user_message(user_input)

        result = await self.chat_completion.get_chat_message_content(
            chat_history=self.history,
            settings=self.settings,
            kernel=self.kernel,
        )

        self.history.add_message(result)

        return result