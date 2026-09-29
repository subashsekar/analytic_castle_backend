"""Token usage normalization for LLM provider responses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMUsage:
    """Normalized token usage reported by an LLM provider, when available."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    model: str | None = None
    request_id: str | None = None

    @property
    def available(self) -> bool:
        return any(
            value is not None
            for value in (self.prompt_tokens, self.completion_tokens, self.total_tokens)
        )


def _optional_non_negative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def normalize_usage(
    raw: object,
    *,
    model: str | None = None,
    request_id: str | None = None,
) -> LLMUsage:
    """Normalize provider usage payloads into the internal model."""
    if not isinstance(raw, dict):
        return LLMUsage(model=model, request_id=request_id)
    return LLMUsage(
        prompt_tokens=_optional_non_negative_int(raw.get("prompt_tokens")),
        completion_tokens=_optional_non_negative_int(raw.get("completion_tokens")),
        total_tokens=_optional_non_negative_int(raw.get("total_tokens")),
        model=model,
        request_id=request_id,
    )
