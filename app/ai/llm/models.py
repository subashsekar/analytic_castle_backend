"""Typed request/response models for the LLM infrastructure layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.ai.llm.usage import LLMUsage

LLMMessageRole = Literal["system", "user", "assistant"]


class LLMMessage(BaseModel):
    role: LLMMessageRole
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message content must not be empty")
        return stripped


class StructuredOutputConfig(BaseModel):
    """Provider structured-output configuration for LLM requests."""

    response_format: dict[str, Any] = Field(
        default_factory=lambda: {"type": "json_object"}
    )


class LLMRequest(BaseModel):
    model: str = Field(min_length=1)
    messages: list[LLMMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)
    structured_output: StructuredOutputConfig | None = None


@dataclass(frozen=True)
class LLMModelConfig:
    """Provider-agnostic model configuration for future agents."""

    model: str
    temperature: float
    max_tokens: int
    timeout: float
    retries: int
    retry_base_backoff: float = 0.5
    retry_max_backoff: float = 30.0


@dataclass(frozen=True)
class LLMResponse:
    """Normalized LLM completion result."""

    content: str
    model: str
    finish_reason: str | None = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    request_id: str | None = None
