from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

logger = logging.getLogger("pydantic_ai_streamlit")

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use tools when they help, and keep responses concise."
)
DEFAULT_MODEL_NAME = "claude-haiku-4-5"


def read_text_file(path: str) -> str:
    file_path = Path(path).expanduser()
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.exception("read_file failed path=%s", file_path)
        return f"read_file error: {exc}"

    logger.info("read_file ok path=%s size=%s", file_path, len(content))
    logger.debug("read_file output=%s", json.dumps(content, default=str))
    return content


def write_text_file(path: str, content: str) -> str:
    file_path = Path(path).expanduser()
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        message = f"Wrote {len(content)} chars to {file_path}"
    except Exception as exc:
        logger.exception("write_file failed path=%s", file_path)
        message = f"write_file error: {exc}"

    logger.info("write_file ok path=%s size=%s", file_path, len(content))
    logger.debug("write_file output=%s", json.dumps(message, default=str))
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
        logger.exception("run_bash failed command=%s", command)
        output = f"run_bash error: {exc}"

    logger.info("run_bash ok command=%s", command)
    logger.debug("run_bash output=%s", json.dumps(output, default=str))
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
        logger.exception("run_python failed command=%s", " ".join(cmd))
        output = f"run_python error: {exc}"

    logger.info("run_python ok command=%s", " ".join(cmd))
    logger.debug("run_python output=%s", json.dumps(output, default=str))
    return output


@st.cache_resource
def get_agent() -> Agent:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    provider = AnthropicProvider(api_key=api_key) if api_key else AnthropicProvider()
    model_name = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL_NAME)
    model = AnthropicModel(model_name, provider=provider)
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
