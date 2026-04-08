import os
import sys
import asyncio
from semantic_kernel.kernel import Kernel
from semantic_kernel.agents import (
    ChatCompletionAgent,
    OrchestrationHandoffs,
    HandoffOrchestration,
)
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.mcp import MCPSsePlugin
from Phase2.Agents.Plugins.rag_plugin import RAGPlugin
from Phase2.Agents.Plugins.task_plugin import TaskPlugin
from Phase2.Agents.models.services.task_repository import TaskRepository
from Phase2.Agents.models.services.notification_service import NotificationService


class OrchestratorAgent:
    def __init__(self) -> None:
        self.kernel = Kernel()

        # AI SERVICE
        self.chat_completion = AzureChatCompletion(
    deployment_name="gpt-5.4-mini",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview"
)

        self.kernel.add_service(self.service)

        # PLUGINS / SERVICES
        self.repo = TaskRepository()
        self.task_plugin = TaskPlugin(self.repo)
        self.rag_plugin = RAGPlugin()
        self.notification_service = NotificationService()

        self.mcp_plugin = MCPSsePlugin(
            name="mcp",
            description="Notification tools",
            url="http://localhost:8002/sse",
            load_prompts=False,
        )

        # STATE
        self.current_input = ""
        self.last_agent_response = ""
        self.request_handled = False
        self.notification_sent = False

        # ORCHESTRATOR
        self.orchestrator_agent = ChatCompletionAgent(
            name="Orchestratoragent",
            description="Routes requests to correct agent.",
            instructions="""
Route ONLY ONCE.

Rules:
- Task → DbAgent
- Info → InformativeAgent
- DO NOT loop
""",
            service=self.service,
        )

        # DB AGENT
        self.db_agent = ChatCompletionAgent(
            name="DbAgent",
            description="Handles task operations.",
            instructions="""
You are a task agent.

Rules:
- ALWAYS use TaskPlugin
- AFTER create/update/delete → handoff to MCPAgent
- DO NOT send notification yourself
- Only list tasks directly
""",
            service=self.service,
            plugins=[self.task_plugin],
        )

        # MCP AGENT
        self.mcp_agent = ChatCompletionAgent(
            name="MCPAgent",
            description="Handles notifications.",
            instructions="""
You are a notification agent.
When you are reached, the system will handle the actual notification.
Do not loop.
Do not call any other agent.
Stop when finished.
""",
            service=self.service,
            plugins=[self.mcp_plugin],
        )

        # INFO AGENT
        self.info_agent = ChatCompletionAgent(
            name="InformativeAgent",
            description="Handles knowledge questions.",
            instructions="Use RAG plugin and answer directly.",
            service=self.service,
            plugins=[self.rag_plugin],
        )

        # HANDOFFS
        self.handoffs = (
            OrchestrationHandoffs()
            .add_many(
                source_agent=self.orchestrator_agent.name,
                target_agents={
                    self.db_agent.name: "task",
                    self.info_agent.name: "info",
                },
            )
            .add(
                source_agent=self.db_agent.name,
                target_agent=self.mcp_agent.name,
                description="Call MCP ONLY ONCE and STOP",
            )
        )

        # ORCHESTRATION
        self.orchestration = HandoffOrchestration(
            members=[
                self.orchestrator_agent,
                self.db_agent,
                self.mcp_agent,
                self.info_agent,
            ],
            handoffs=self.handoffs,
            agent_response_callback=self.agent_response_callback,
        )

        self.runtime = InProcessRuntime()
        self.runtime.start()
        asyncio.create_task(self.initialize_mcp())

    async def initialize_mcp(self):
        try:
            await self.mcp_plugin.connect()
            print(" MCP connected")
            print(" MCP READY (tools auto-loaded)")
        except Exception as e:
            print("MCP failed:", e)

    def agent_response_callback(self, message: ChatMessageContent) -> None:
        text = ""

        if hasattr(message, "content") and message.content:
            text = str(message.content).strip()
        elif hasattr(message, "items") and message.items:
            try:
                text = "".join(
                    [str(i.text) for i in message.items if hasattr(i, "text")]
                ).strip()
            except Exception:
                text = ""

        if text:
            print(f"{message.name}: {text}")

        if message.name == "InformativeAgent" and text:
            self.last_agent_response = text
            return

        if message.name == "MCPAgent":
            self.request_handled = True
            return

    async def handle_request(self, user_input: str) -> str:
        try:
            self.current_input = user_input.strip()
            self.last_agent_response = ""
            self.request_handled = False
            self.notification_sent = False
            self.task_plugin.last_action_message = ""
            self.task_plugin.last_action_type = ""

            if not self.current_input:
                return "Please enter a valid query."

            if "list" in self.current_input.lower():
                try:
                    words = self.current_input.lower().split()
                    if "for" in words:
                        name = words[words.index("for") + 1]
                        return await self.task_plugin.list_tasks(name)
                    else:
                        return await self.task_plugin.list_tasks()
                except Exception as e:
                    return f"Error listing tasks: {str(e)}"

            result = await self.orchestration.invoke(
                task=self.current_input,
                runtime=self.runtime,
            )

            await result.get()

            print("DEBUG AFTER RESULT")
            print("last_action_message =", self.task_plugin.last_action_message)
            print("last_action_type =", self.task_plugin.last_action_type)
            print("notification_sent =", self.notification_sent)

            if self.task_plugin.last_action_message and not self.notification_sent:
                send_result = self.notification_service.send_notification(
                    self.task_plugin.last_action_message
                )
                print("DEBUG ABOUT TO SEND:", self.task_plugin.last_action_message)
                print("MCP PROGRAMMATIC SEND:", send_result)
                self.notification_sent = send_result.get("success", False)

            if self.request_handled:
                return self.last_agent_response or self.task_plugin.last_action_message or "Task completed."

            if self.last_agent_response:
                return self.last_agent_response

            if self.task_plugin.last_action_message:
                return self.task_plugin.last_action_message

            return "Task completed successfully."

        except Exception as e:
            print("ERROR:", e)
            return "Something went wrong."

    async def shutdown(self):
        try:
            if self.mcp_plugin:
                await self.mcp_plugin.close()
        except Exception:
            pass

        await self.runtime.stop_when_idle()