from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class LLMSelection(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    provider: Literal["ollama", "openai"]
    model: str = Field(min_length=1, max_length=200)


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)
    llm: LLMSelection | None = None


class SessionResetRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str = Field(min_length=1, max_length=128)
