from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from base import BaseLLMClient
from schemas import ChatMessage, ModelParams, ModelResponse


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str) -> None:
        super().__init__(api_key, model)
        self._client = AsyncAnthropic(api_key=api_key)

    @property
    def provider(self) -> str:
        return "anthropic"

    def _split_system(
        self, messages: list[ChatMessage]
    ) -> tuple[str | None, list[dict[str, str]]]:
        system_parts = [
            message.content for message in messages if message.role == "system"
        ]
        conversation = [
            {"role": message.role, "content": message.content}
            for message in messages
            if message.role != "system"
        ]
        system_prompt = "\n\n".join(system_parts) if system_parts else None
        return system_prompt, conversation

    def _text_from_content(self, content: list[object]) -> str:
        texts: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                texts.append(text)
        return "".join(texts)

    async def _generate(
        self,
        messages: list[ChatMessage],
        params: ModelParams,
        model: str,
    ) -> ModelResponse:
        system_prompt, conversation = self._split_system(messages)
        kwargs: dict[str, object] = {
            "model": model,
            "max_tokens": params.max_tokens,
            "messages": conversation,
            "extra_body": {"temperature": min(params.temperature, 1.0)},
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await self._client.messages.create(**kwargs)
        return ModelResponse(
            content=self._text_from_content(list(response.content)),
            provider=self.provider,
            model=model,
            success=True,
            finish_reason=response.stop_reason,
        )

    async def _generate_stream(
        self,
        messages: list[ChatMessage],
        params: ModelParams,
        model: str,
    ) -> AsyncIterator[str]:
        system_prompt, conversation = self._split_system(messages)
        kwargs: dict[str, object] = {
            "model": model,
            "max_tokens": params.max_tokens,
            "messages": conversation,
            "extra_body": {"temperature": min(params.temperature, 1.0)},
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
