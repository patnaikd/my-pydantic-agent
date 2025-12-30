from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from pydantic_ai._run_context import RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelName
from pydantic_ai.profiles import ModelProfileSpec
from pydantic_ai.providers import Provider
from pydantic_ai.providers.anthropic import AsyncAnthropicClient
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

logger = logging.getLogger("custom_logging_model")


class LoggingAnthropicModel(AnthropicModel):
    """Log all Model-level method calls while delegating to AnthropicModel."""

    def __init__(
        self,
        model_name: AnthropicModelName,
        *,
        provider: str | Provider[AsyncAnthropicClient] = "anthropic",
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ) -> None:
        super().__init__(
            model_name,
            provider=provider,
            profile=profile,
            settings=settings,
        )

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        logger.info(
            "model.request start model=%s messages=%s output_mode=%s tools=%s",
            super().model_name,
            len(messages),
            model_request_parameters.output_mode,
            len(model_request_parameters.function_tools) + len(model_request_parameters.output_tools),
        )
        try:
            response = await super().request(messages, model_settings, model_request_parameters)
        except Exception:
            logger.exception("model.request error model=%s", super().model_name)
            raise
        logger.info("model.request done model=%s", super().model_name)
        return response

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        logger.debug(
            "model.count_tokens start model=%s messages=%s",
            super().model_name,
            len(messages),
        )
        try:
            usage = await super().count_tokens(messages, model_settings, model_request_parameters)
        except Exception:
            logger.exception("model.count_tokens error model=%s", super().model_name)
            raise
        logger.debug("model.count_tokens done model=%s", super().model_name)
        return usage

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        logger.info(
            "model.request_stream start model=%s messages=%s output_mode=%s tools=%s",
            super().model_name,
            len(messages),
            model_request_parameters.output_mode,
            len(model_request_parameters.function_tools) + len(model_request_parameters.output_tools),
        )
        try:
            async with super().request_stream(
                messages,
                model_settings,
                model_request_parameters,
                run_context,
            ) as response_stream:
                logger.info("model.request_stream ready model=%s", super().model_name)
                yield response_stream
        except Exception:
            logger.exception("model.request_stream error model=%s", super().model_name)
            raise
        finally:
            logger.info("model.request_stream done model=%s", super().model_name)

    def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
        logger.debug("model.customize_request_parameters model=%s", super().model_name)
        return super().customize_request_parameters(model_request_parameters)

    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        logger.debug("model.prepare_request model=%s", super().model_name)
        return super().prepare_request(model_settings, model_request_parameters)

    @classmethod
    def supported_builtin_tools(cls):
        logger.debug("model.supported_builtin_tools")
        return super().supported_builtin_tools()

    @property
    def model_name(self) -> str:
        logger.debug("model.model_name")
        return super().model_name

    @property
    def label(self) -> str:
        logger.debug("model.label")
        return super().label

    @property
    def system(self) -> str:
        logger.debug("model.system")
        return super().system

    @property
    def base_url(self) -> str | None:
        logger.debug("model.base_url")
        return super().base_url

    @property
    def profile(self):  # type: ignore[override]
        logger.debug("model.profile")
        return super().profile

    @property
    def settings(self) -> ModelSettings | None:
        logger.debug("model.settings")
        return super().settings

    @staticmethod
    def _get_instructions(
        messages: Sequence[ModelMessage],
        model_request_parameters: ModelRequestParameters | None = None,
    ) -> str | None:
        logger.debug("model._get_instructions")
        return AnthropicModel._get_instructions(messages, model_request_parameters)
