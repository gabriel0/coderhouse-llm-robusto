import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TypeVar

import anthropic
import openai

from schemas import ChatMessage, ModelParams, ModelResponse

T = TypeVar("T")

MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 1.0

_RETRYABLE: tuple[type[BaseException], ...] = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.InternalServerError,
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
    anthropic.OverloadedError,
)

_AUTH_ERRORS: tuple[type[BaseException], ...] = (
    openai.AuthenticationError,
    anthropic.AuthenticationError,
)

_RATE_LIMIT_ERRORS: tuple[type[BaseException], ...] = (
    openai.RateLimitError,
    anthropic.RateLimitError,
)

_NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    openai.APIConnectionError,
    anthropic.APIConnectionError,
)


def format_llm_error(error: Exception) -> str:
    if isinstance(error, _AUTH_ERRORS):
        return "API key inválida o ausente."
    if isinstance(error, _RATE_LIMIT_ERRORS):
        return "Límite de tasa (rate limit) alcanzado. Intenta de nuevo más tarde."
    if isinstance(error, _NETWORK_ERRORS):
        return "Error de red al contactar al proveedor."
    return f"Error del proveedor: {error}"


def is_retryable(error: Exception) -> bool:
    if isinstance(error, _AUTH_ERRORS):
        return False
    return isinstance(error, _RETRYABLE)


class BaseLLMClient(ABC):
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    @property
    @abstractmethod
    def provider(self) -> str:
        raise NotImplementedError

    def resolve_model(self, params: ModelParams) -> str:
        return params.model or self._model

    def error_response(self, error: Exception, model: str) -> ModelResponse:
        return ModelResponse(
            content="",
            provider=self.provider,
            model=model,
            success=False,
            error=format_llm_error(error),
        )

    async def _with_retry(self, operation: Callable[[], Awaitable[T]]) -> T:
        last_error: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return await operation()
            except Exception as error:
                last_error = error
                if not is_retryable(error) or attempt == MAX_ATTEMPTS:
                    raise
                await asyncio.sleep(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
        assert last_error is not None
        raise last_error

    async def generate(
        self,
        messages: list[ChatMessage],
        params: ModelParams | None = None,
    ) -> ModelResponse:
        params = params or ModelParams()
        validated = [ChatMessage.model_validate(message) for message in messages]
        model = self.resolve_model(params)
        try:
            return await self._with_retry(
                lambda: self._generate(validated, params, model)
            )
        except Exception as error:
            return self.error_response(error, model)

    async def generate_stream(
        self,
        messages: list[ChatMessage],
        params: ModelParams | None = None,
    ) -> AsyncIterator[str]:
        params = params or ModelParams()
        validated = [ChatMessage.model_validate(message) for message in messages]
        model = self.resolve_model(params)

        for attempt in range(1, MAX_ATTEMPTS + 1):
            yielded = False
            try:
                async for token in self._generate_stream(validated, params, model):
                    yielded = True
                    yield token
                return
            except Exception as error:
                can_retry = (
                    not yielded
                    and is_retryable(error)
                    and attempt < MAX_ATTEMPTS
                )
                if can_retry:
                    await asyncio.sleep(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
                    continue
                yield f"[ERROR] {format_llm_error(error)}"
                return

    @abstractmethod
    async def _generate(
        self,
        messages: list[ChatMessage],
        params: ModelParams,
        model: str,
    ) -> ModelResponse:
        raise NotImplementedError

    @abstractmethod
    def _generate_stream(
        self,
        messages: list[ChatMessage],
        params: ModelParams,
        model: str,
    ) -> AsyncIterator[str]:
        raise NotImplementedError
