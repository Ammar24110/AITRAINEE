from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.connectors.mcp import MCPSsePlugin


from Phase2.Agents.Plugins.task_plugin import TaskPlugin
from Phase2.Agents.models.services.task_repository import TaskRepository
import sys
import asyncio
import os

class DbAgent:
    def __init__(self) -> None:
        self.kernel = Kernel()

        self.chat_completion = AzureChatCompletion(
    deployment_name="gpt-5.4-mini",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview"
)
        self.kernel.add_service(self.chat_completion)

        # TASK PLUGIN
        self.repo = TaskRepository()
        self.task_plugin = TaskPlugin(self.repo)
        self.kernel.add_plugin(self.task_plugin, plugin_name="tasks")

    
        self.mcp_plugin = MCPSsePlugin(
    name="mcp",
    description="Notification tools",
    url="http://localhost:8002/sse"
)

        self.kernel.add_plugin(self.mcp_plugin, plugin_name="mcp")

        asyncio.create_task(self.mcp_plugin.connect())

        self.history = ChatHistory()

        self.settings = AzureChatPromptExecutionSettings()
        self.settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    async def handle_request(self, user_input: str) -> str:

        system_prompt = """
You handle all task operations.

Rules:
- ALWAYS use task tools for task operations
- AFTER creating or updating a task, you MUST call mcp.send_notification
- NEVER skip the notification step
- Return a natural language response to the user
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

        return str(result)