"""Typed models for prompt definitions, versions, and rendered output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ai.llm.models import StructuredOutputConfig

PromptMessageRole = Literal["system", "user", "assistant"]


class PromptVersion(BaseModel):
    """Semantic version label for a prompt definition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = Field(min_length=1, max_length=64)

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("prompt version must not be empty")
        if not stripped.replace("_", "").replace("-", "").isalnum():
            raise ValueError("prompt version contains invalid characters")
        return stripped

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class SystemPrompt:
    """Versioned system prompt content."""

    prompt_id: str
    version: PromptVersion
    content: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.prompt_id.strip():
            raise ValueError("prompt_id must not be empty")
        if not self.content.strip():
            raise ValueError("system prompt content must not be empty")


@dataclass(frozen=True)
class PromptTemplate:
    """Versioned user/assistant prompt template with named placeholders."""

    prompt_id: str
    version: PromptVersion
    template: str
    role: PromptMessageRole = "user"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.prompt_id.strip():
            raise ValueError("prompt_id must not be empty")
        if not self.template.strip():
            raise ValueError("prompt template must not be empty")


@dataclass(frozen=True)
class RenderedPrompt:
    """Safe rendered prompt result. Never logs full content by default."""

    prompt_id: str
    version: PromptVersion
    role: PromptMessageRole
    content: str

    @property
    def char_count(self) -> int:
        return len(self.content)

    def __repr__(self) -> str:
        return (
            "RenderedPrompt("
            f"prompt_id={self.prompt_id!r}, "
            f"version={self.version!r}, "
            f"role={self.role!r}, "
            f"char_count={self.char_count})"
        )


@dataclass(frozen=True)
class PromptBundle:
    """Rendered prompts ready to convert into LLM messages."""

    version: PromptVersion
    system: RenderedPrompt | None = None
    user: RenderedPrompt | None = None
    assistant: RenderedPrompt | None = None
    structured_output: StructuredOutputConfig | None = None

    def __post_init__(self) -> None:
        if self.system is None and self.user is None and self.assistant is None:
            raise ValueError("prompt bundle must contain at least one message")
