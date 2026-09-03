from collections.abc import AsyncIterator

from google.genai import Client as GoogleGenAIClient
from google.genai import types

from base import BaseLLMClient
from schemas import ChatMessage, ModelParams, ModelResponse


class GoogleClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str) -> None:
        super().__init__(api_key, model)
        self._client = GoogleGenAIClient(api_key=api_key)

    @property
    def provider(self) -> str:
        return "google"

    def _to_gemini_input(
        self, messages: list[ChatMessage]
    ) -> tuple[str | None, list[types.Content]]:
        system_parts = [
            message.content for message in messages if message.role == "system"
        ]
        contents: list[types.Content] = []
        for message in messages:
            if message.role == "system":
                continue
            role = "model" if message.role == "assistant" else "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=message.content)],
                )
            )
        system_prompt = "\n\n".join(system_parts) if system_parts else None
        return system_prompt, contents

    def _build_config(
        self, params: ModelParams, system_prompt: str | None
    ) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=params.temperature,
            max_output_tokens=params.max_tokens,
            system_instruction=system_prompt,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

    def _finish_reason(self, response: types.GenerateContentResponse) -> str | None:
        if not response.candidates:
            return None
        reason = response.candidates[0].finish_reason
        return str(reason) if reason is not None else None

    async def _generate(
        self,
        messages: list[ChatMessage],
        params: ModelParams,
        model: str,
    ) -> ModelResponse:
        system_prompt, contents = self._to_gemini_input(messages)
        response = await self._client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=self._build_config(params, system_prompt),
        )
        return ModelResponse(
            content=response.text or "",
            provider=self.provider,
            model=model,
            success=True,
            finish_reason=self._finish_reason(response),
        )

    async def _generate_stream(
        self,
        messages: list[ChatMessage],
        params: ModelParams,
        model: str,
    ) -> AsyncIterator[str]:
        system_prompt, contents = self._to_gemini_input(messages)
        stream = await self._client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=self._build_config(params, system_prompt),
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text
