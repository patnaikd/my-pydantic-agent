from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from pydantic_ai.messages import ToolReturnPart

from agent_with_tools import get_agent

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "DEBUG").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger("pydantic_ai_cli")


def tool_events_from_messages(messages: list[object]) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    for message in messages:
        parts = getattr(message, "parts", None)
        if not parts:
            continue
        for part in parts:
            if isinstance(part, ToolReturnPart):
                content = part.model_response_str()
                events.append({"tool": part.tool_name, "result": content})
    return events


def print_banner() -> None:
    print("Pydantic AI CLI Chatbot")
    print("Type /exit or /quit to leave. Type /reset to clear history.")
    print()


def prompt_loop() -> None:
    messages: list[dict[str, str]] = []
    model_messages: list[object] = []

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Note: Set ANTHROPIC_API_KEY to use the Anthropic model.")
        print()

    agent = get_agent()

    while True:
        try:
            prompt = input("You> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\nInterrupted. Type /exit to quit.")
            continue

        if not prompt:
            continue

        lowered = prompt.lower()
        if lowered in {"/exit", "/quit", "exit", "quit"}:
            break
        if lowered == "/reset":
            messages.clear()
            model_messages.clear()
            print("History cleared.")
            continue

        logger.info("user_prompt chars=%s", len(prompt))
        messages.append({"role": "user", "content": prompt})

        try:
            result = agent.run_sync(prompt, message_history=model_messages)
        except Exception as exc:
            logger.exception("agent_error")
            error_text = f"Agent error: {exc}"
            print(f"Assistant> {error_text}")
            messages.append({"role": "assistant", "content": error_text})
            continue

        tool_events = tool_events_from_messages(result.new_messages())
        for event in tool_events:
            print(f"[tool:{event['tool']}]")
            print(event["result"])
            messages.append(
                {
                    "role": "assistant",
                    "content": event["result"],
                    "kind": "tool",
                    "tool": event["tool"],
                }
            )

        response_text = result.output if isinstance(result.output, str) else str(result.output)
        print(f"Assistant> {response_text}")
        messages.append({"role": "assistant", "content": response_text})
        model_messages = result.all_messages()


if __name__ == "__main__":
    print_banner()
    prompt_loop()
