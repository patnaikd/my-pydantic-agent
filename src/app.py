from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("pydantic_ai_streamlit")

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use tools when they help, and keep responses concise."
)
MODEL_NAME = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")


def read_text_file(path: str) -> str:
    file_path = Path(path).expanduser()
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.exception("read_file failed", extra={"path": str(file_path)})
        return f"read_file error: {exc}"

    logger.info("read_file ok", extra={"path": str(file_path), "size": len(content)})
    return content


def write_text_file(path: str, content: str) -> str:
    file_path = Path(path).expanduser()
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        message = f"Wrote {len(content)} chars to {file_path}"
    except Exception as exc:
        logger.exception("write_file failed", extra={"path": str(file_path)})
        message = f"write_file error: {exc}"

    logger.info("write_file ok", extra={"path": str(file_path), "size": len(content)})
    return message


def run_bash(command: str) -> str:
    try:
        completed = subprocess.run(
            command,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
        )
        output = (
            f"$ {command}\n"
            f"exit_code: {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    except Exception as exc:
        logger.exception("run_bash failed", extra={"command": command})
        output = f"run_bash error: {exc}"

    logger.info("run_bash ok", extra={"command": command})
    return output


def run_python(script_path: str, args: list[str] | None = None) -> str:
    script = Path(script_path).expanduser()
    cmd = [sys.executable, str(script)] + (args or [])
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        output = (
            f"$ {' '.join(cmd)}\n"
            f"exit_code: {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    except Exception as exc:
        logger.exception("run_python failed", extra={"command": " ".join(cmd)})
        output = f"run_python error: {exc}"

    logger.info("run_python ok", extra={"command": " ".join(cmd)})
    return output


@st.cache_resource
def get_agent() -> Agent:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    provider = AnthropicProvider(api_key=api_key) if api_key else AnthropicProvider()
    model = AnthropicModel(MODEL_NAME, provider=provider)
    my_coding_agent = Agent(model, system_prompt=SYSTEM_PROMPT)

    @my_coding_agent.tool_plain
    def read_file(path: str) -> str:
        return read_text_file(path)

    @my_coding_agent.tool_plain
    def write_file(path: str, content: str) -> str:
        return write_text_file(path, content)

    @my_coding_agent.tool_plain
    def execute_bash(command: str) -> str:
        return run_bash(command)

    @my_coding_agent.tool_plain
    def execute_python(script_path: str, args: list[str] | None = None) -> str:
        return run_python(script_path, args)

    return my_coding_agent

def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("model_messages", [])
    st.session_state.setdefault("tool_log", [])


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


init_state()

st.title("Pydantic AI Streamlit Chatbot")

if not os.getenv("ANTHROPIC_API_KEY"):
    st.info("Set ANTHROPIC_API_KEY to use the Anthropic model.")

with st.expander("Tool events (session)", expanded=False):
    if st.session_state["tool_log"]:
        for event in st.session_state["tool_log"]:
            st.markdown(f"Tool output: `{event['tool']}`")
            st.code(event["result"])
    else:
        st.caption("No tool events yet.")

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        if message.get("kind") == "tool":
            st.markdown(f"Tool output: `{message['tool']}`")
            st.code(message["content"])
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("Ask anything"):
    logger.info("user_prompt", extra={"prompt_chars": len(prompt)})
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
    st.session_state["tool_log"].extend(tool_events)
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
