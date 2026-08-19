from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel

from app.ai.types import TokenUsage

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str


@dataclass(frozen=True)
class LLMGeneration:
    content: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str | None = None


@dataclass(frozen=True)
class StructuredGeneration(Generic[T]):
    result: T
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str | None = None


class LLMProvider(Protocol):
    """Async LLM interface. Application code depends on this, not a vendor SDK."""

    @property
    def name(self) -> str: ...

    async def generate(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMGeneration: ...

    async def generate_structured(
        self,
        messages: Sequence[LLMMessage],
        schema: type[T],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> StructuredGeneration[T]: ...
