import os
from collections.abc import AsyncIterator
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from anthropic_client import AnthropicClient
from base import BaseLLMClient
from google_client import GoogleClient
from openai_client import OpenAIClient
from schemas import ChatMessage, LLMConfig, ModelParams, ModelResponse, Provider


class AsyncLLMManager:
    """Carga OpenAI, Anthropic o Google según configuración y expone una interfaz unificada."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = self._build_client(config)

    @classmethod
    def from_env(cls) -> "AsyncLLMManager":
        load_dotenv(Path(__file__).with_name(".env"), override=True)
        raw_provider = os.getenv("LLM_PROVIDER", Provider.OPENAI.value).strip().lower()
        try:
            provider = Provider(raw_provider)
        except ValueError as error:
            raise ValueError(
                f"Proveedor no soportado: '{raw_provider}'. "
                "Usa 'openai', 'anthropic' o 'google'."
            ) from error

        try:
            config = LLMConfig(
                provider=provider,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
                google_model=os.getenv("GOOGLE_MODEL", "gemini-3.6-flash"),
            )
        except ValidationError as error:
            first = error.errors()[0]
            message = str(first.get("msg", "Configuración inválida"))
            raise ValueError(message.removeprefix("Value error, ")) from error
        return cls(config)

    @property
    def provider(self) -> Provider:
        return self._config.provider

    @property
    def client(self) -> BaseLLMClient:
        return self._client

    def _build_client(self, config: LLMConfig) -> BaseLLMClient:
        if config.provider is Provider.OPENAI:
            assert config.openai_api_key is not None
            return OpenAIClient(
                api_key=config.openai_api_key.get_secret_value(),
                model=config.openai_model,
            )
        if config.provider is Provider.ANTHROPIC:
            assert config.anthropic_api_key is not None
            return AnthropicClient(
                api_key=config.anthropic_api_key.get_secret_value(),
                model=config.anthropic_model,
            )
        assert config.google_api_key is not None
        return GoogleClient(
            api_key=config.google_api_key.get_secret_value(),
            model=config.google_model,
        )

    async def generate(
        self,
        messages: list[ChatMessage],
        params: ModelParams | None = None,
    ) -> ModelResponse:
        return await self._client.generate(messages, params)

    async def generate_stream(
        self,
        messages: list[ChatMessage],
        params: ModelParams | None = None,
    ) -> AsyncIterator[str]:
        async for token in self._client.generate_stream(messages, params):
            yield token
