from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ModelParams(BaseModel):
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    model: str | None = None


class ModelResponse(BaseModel):
    content: str = ""
    provider: str
    model: str
    success: bool = True
    error: str | None = None
    finish_reason: str | None = None


class LLMConfig(BaseModel):
    provider: Provider
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-5-sonnet-latest"
    google_model: str = "gemini-2.5-flash"

    @model_validator(mode="after")
    def validate_api_key(self) -> "LLMConfig":
        api_keys = {
            Provider.OPENAI: self.openai_api_key,
            Provider.ANTHROPIC: self.anthropic_api_key,
            Provider.GOOGLE: self.google_api_key,
        }
        api_key = api_keys[self.provider]
        if api_key is None or not api_key.get_secret_value().strip():
            raise ValueError(
                f"Falta la API key para el proveedor '{self.provider.value}'"
            )
        return self
