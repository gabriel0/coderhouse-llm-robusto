from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from base import BaseLLMClient
from schemas import ChatMessage, ModelParams, ModelResponse


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str) -> None:
        super().__init__(api_key, model)
        self._client = AsyncOpenAI(api_key=api_key)

    @property
    def provider(self) -> str:
        return "openai"

    def _to_openai_messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": message.role, "content": message.content} for message in messages]

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
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
