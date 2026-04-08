import asyncio
import logging

from semantic_kernel import Kernel
from semantic_kernel.utils.logging import setup_logging
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

async def main():
    kernel = Kernel()

    chat_completion = OllamaChatCompletion(
        service_id="chat",
        ai_model_id="llama3",
        host="http://localhost:11434",
    )

    kernel.add_service(chat_completion)

    setup_logging()
    logging.getLogger("kernel").setLevel(logging.DEBUG)

    execution_settings = AzureChatPromptExecutionSettings()
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    history = ChatHistory()

    while True:
        userInput = input("User > ")

        if userInput == "exit":
            break

        history.add_user_message(userInput)

        result = await chat_completion.get_chat_message_content(
            chat_history=history,
            settings=execution_settings,
            kernel=kernel,
        )

        print("Assistant >", result)

        history.add_message(result)

if __name__ == "__main__":
    asyncio.run(main())