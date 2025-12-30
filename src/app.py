from __future__ import annotations

import json
import logging
import os

import streamlit as st
from dotenv import load_dotenv
from pydantic_ai.messages import ModelMessagesTypeAdapter, ToolReturnPart

from agent_with_tools import get_agent

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "DEBUG").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger("pydantic_ai_streamlit")


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("model_messages", [])


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


if __name__ == "__main__":

    init_state()

    st.title("Pydantic AI Streamlit Chatbot")

    if not os.getenv("ANTHROPIC_API_KEY"):
        st.info("Set ANTHROPIC_API_KEY to use the Anthropic model.")

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if message.get("kind") == "tool":
                st.markdown(f"Tool output: `{message['tool']}`")
                st.code(message["content"])
            else:
                st.markdown(message["content"])

    if prompt := st.chat_input("Ask anything"):
        logger.info("user_prompt chars=%s", len(prompt))
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            with st.spinner("Thinking..."):
                agent = get_agent()
                result = agent.run_sync(
                    prompt,
                    message_history=st.session_state["model_messages"],
                )
        except Exception as exc:
            logger.exception("agent_error")
            error_text = f"Agent error: {exc}"
            with st.chat_message("assistant"):
                st.markdown(error_text)
            st.session_state["messages"].append({"role": "assistant", "content": error_text})
            st.stop()

        tool_events = tool_events_from_messages(result.new_messages())
        for event in tool_events:
            with st.chat_message("assistant"):
                st.markdown(f"Tool output: `{event['tool']}`")
                st.code(event["result"])
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": event["result"],
                    "kind": "tool",
                    "tool": event["tool"],
                }
            )

        response_text = result.output if isinstance(result.output, str) else str(result.output)
        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.session_state["messages"].append({"role": "assistant", "content": response_text})
        st.session_state["model_messages"] = result.all_messages()

    with st.expander("messages (raw json)", expanded=False):
        st.code(
            json.dumps(st.session_state["messages"], indent=2, default=str),
            language="json",
        )

    with st.expander("model_messages (raw json)", expanded=False):
        model_messages_json = ModelMessagesTypeAdapter.dump_json(
            st.session_state["model_messages"],
            indent=2,
        ).decode("utf-8")
        st.code(model_messages_json, language="json")
