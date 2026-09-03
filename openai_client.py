from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from base import BaseLLMClient
from schemas import ChatMessage, ModelParams, ModelResponse

# Clase para manejar el cliente de OpenAI
class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str) -> None:
        super().__init__(api_key, model)
        self._client = AsyncOpenAI(api_key=api_key)

    # Método para obtener el proveedor del cliente
    @property
    def provider(self) -> str:
        return "openai"

    # Método para convertir los mensajes a los formatos de OpenAI
    def _to_openai_messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": message.role, "content": message.content} for message in messages]

    # Método para generar una respuesta
    async def _generate(
        self,
        messages: list[ChatMessage],
        params: ModelParams,
        model: str,
    ) -> ModelResponse:
        response = await self._client.chat.completions.create(
            model=model,
            messages=self._to_openai_messages(messages),
            temperature=params.temperature,
            max_tokens=params.max_tokens,
        )
        choice = response.choices[0]
        return ModelResponse(
            content=choice.message.content or "",
            provider=self.provider,
            model=model,
            success=True,
            finish_reason=choice.finish_reason,
        )

    # Método para generar una respuesta en streaming
    async def _generate_stream(
        self,
        messages: list[ChatMessage],
        params: ModelParams,
        model: str,
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=model,
            messages=self._to_openai_messages(messages),
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            stream=True,
        )
        async for chunk in stream: # Genera la respuesta en streaming
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content # Obtiene el token generado
            if delta:
                yield delta # Retorna el token generado
