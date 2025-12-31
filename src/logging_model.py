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


def _safe_repr(value: object, limit: int = 800) -> str:
    try:
        text = repr(value)
    except Exception as exc:
        return f"<repr_error {type(value).__name__}: {exc}>"
    if len(text) > limit:
        return f"{text[:limit]}...(truncated)"
    return text


def _summarize(value: object) -> str:
    if isinstance(value, ModelRequestParameters):
        return (
            "ModelRequestParameters("
            f"output_mode={value.output_mode}, "
            f"function_tools={len(value.function_tools)}, "
            f"output_tools={len(value.output_tools)}, "
            f"builtin_tools={len(value.builtin_tools)}, "
            f"allow_text_output={value.allow_text_output}, "
            f"allow_image_output={value.allow_image_output}"
            ")"
        )
    if isinstance(value, ModelResponse):
        return (
            "ModelResponse("
            f"parts={len(value.parts)}, "
            f"usage={_safe_repr(value.usage)}, "
            f"model_name={value.model_name!r}, "
            f"finish_reason={value.finish_reason!r}"
            ")"
        )
    if isinstance(value, RequestUsage):
        return f"RequestUsage({_safe_repr(value)})"
    if isinstance(value, str):
        return f"str(len={len(value)}) {_safe_repr(value)}"
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return f"{type(value).__name__}(len={len(value)})"
    if isinstance(value, dict):
        return f"dict(len={len(value)})"
    if value is None:
        return "None"
    return _safe_repr(value)


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
        logger.info(
            "model.__init__ input model_name=%s provider=%s profile=%s settings=%s",
            _summarize(model_name),
            _summarize(provider),
            _summarize(profile),
            _summarize(settings),
        )
        super().__init__(
            model_name,
            provider=provider,
            profile=profile,
            settings=settings,
        )
        logger.info(
            "model.__init__ output instance=%s model_name=%s base_url=%s",
            f"{self.__class__.__name__}@{id(self)}",
            _summarize(super().model_name),
            _summarize(super().base_url),
        )

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        logger.info(
            "model.request input model=%s messages=%s model_settings=%s model_request_parameters=%s",
            _summarize(super().model_name),
            _summarize(messages),
            _summarize(model_settings),
            _summarize(model_request_parameters),
        )
        try:
            response = await super().request(messages, model_settings, model_request_parameters)
        except Exception:
            logger.exception("model.request error model=%s", super().model_name)
            raise
        logger.info(
            "model.request output model=%s response=%s",
            _summarize(super().model_name),
            _summarize(response),
        )
        return response

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        logger.info(
            "model.count_tokens input model=%s messages=%s model_settings=%s model_request_parameters=%s",
            _summarize(super().model_name),
            _summarize(messages),
            _summarize(model_settings),
            _summarize(model_request_parameters),
        )
        try:
            usage = await super().count_tokens(messages, model_settings, model_request_parameters)
        except Exception:
            logger.exception("model.count_tokens error model=%s", super().model_name)
            raise
        logger.info(
            "model.count_tokens output model=%s usage=%s",
            _summarize(super().model_name),
            _summarize(usage),
        )
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
            "model.request_stream input model=%s messages=%s model_settings=%s model_request_parameters=%s run_context=%s",
            _summarize(super().model_name),
            _summarize(messages),
            _summarize(model_settings),
            _summarize(model_request_parameters),
            _summarize(run_context),
        )
        try:
            async with super().request_stream(
                messages,
                model_settings,
                model_request_parameters,
                run_context,
            ) as response_stream:
                logger.info(
                    "model.request_stream output model=%s response_stream=%s",
                    _summarize(super().model_name),
                    _summarize(response_stream),
                )
                yield response_stream
        except Exception:
            logger.exception("model.request_stream error model=%s", super().model_name)
            raise
        finally:
            logger.info("model.request_stream done model=%s", _summarize(super().model_name))

    def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
        logger.info(
            "model.customize_request_parameters input model=%s model_request_parameters=%s",
            _summarize(super().model_name),
            _summarize(model_request_parameters),
        )
        params = super().customize_request_parameters(model_request_parameters)
        logger.info(
            "model.customize_request_parameters output model=%s model_request_parameters=%s",
            _summarize(super().model_name),
            _summarize(params),
        )
        return params

    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        logger.info(
            "model.prepare_request input model=%s model_settings=%s model_request_parameters=%s",
            _summarize(super().model_name),
            _summarize(model_settings),
            _summarize(model_request_parameters),
        )
        settings_out, params_out = super().prepare_request(model_settings, model_request_parameters)
        logger.info(
            "model.prepare_request output model=%s model_settings=%s model_request_parameters=%s",
            _summarize(super().model_name),
            _summarize(settings_out),
            _summarize(params_out),
        )
        return settings_out, params_out

    @classmethod
    def supported_builtin_tools(cls):
        logger.info("model.supported_builtin_tools input cls=%s", cls.__name__)
        tools = super().supported_builtin_tools()
        logger.info("model.supported_builtin_tools output cls=%s tools=%s", cls.__name__, _summarize(tools))
        return tools

    @property
    def model_name(self) -> str:
        logger.info("model.model_name input instance=%s", f"{self.__class__.__name__}@{id(self)}")
        value = super().model_name
        logger.info("model.model_name output value=%s", _summarize(value))
        return value

    @property
    def label(self) -> str:
        logger.info("model.label input instance=%s", f"{self.__class__.__name__}@{id(self)}")
        value = super().label
        logger.info("model.label output value=%s", _summarize(value))
        return value

    @property
    def system(self) -> str:
        logger.info("model.system input instance=%s", f"{self.__class__.__name__}@{id(self)}")
        value = super().system
        logger.info("model.system output value=%s", _summarize(value))
        return value

    @property
    def base_url(self) -> str | None:
        logger.info("model.base_url input instance=%s", f"{self.__class__.__name__}@{id(self)}")
        value = super().base_url
        logger.info("model.base_url output value=%s", _summarize(value))
        return value

    @property
    def profile(self):  # type: ignore[override]
        logger.info("model.profile input instance=%s", f"{self.__class__.__name__}@{id(self)}")
        value = super().profile
        logger.info("model.profile output value=%s", _summarize(value))
        return value

    @property
    def settings(self) -> ModelSettings | None:
        logger.info("model.settings input instance=%s", f"{self.__class__.__name__}@{id(self)}")
        value = super().settings
        logger.info("model.settings output value=%s", _summarize(value))
        return value

    @staticmethod
    def _get_instructions(
        messages: Sequence[ModelMessage],
        model_request_parameters: ModelRequestParameters | None = None,
    ) -> str | None:
        logger.info(
            "model._get_instructions input messages=%s model_request_parameters=%s",
            _summarize(messages),
            _summarize(model_request_parameters),
        )
        instructions = AnthropicModel._get_instructions(messages, model_request_parameters)
        logger.info("model._get_instructions output instructions=%s", _summarize(instructions))
        return instructions
