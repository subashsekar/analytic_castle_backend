from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.ai.intent_types import AIIntent, AIRequestPlan
from app.ai.metadata_types import ResolvedMetadataContext

AI_CAPABILITY_CHAT = "chat"
AI_CAPABILITY_METADATA_LOOKUP = "metadata_lookup"
DEFAULT_AI_CAPABILITIES: frozenset[str] = frozenset(
    {AI_CAPABILITY_CHAT, AI_CAPABILITY_METADATA_LOOKUP}
)

_MAX_INTENT_CHARS = 64
_MAX_METADATA_ITEMS = 20
_MAX_WARNING_ITEMS = 20
_MAX_ANALYSIS_ANSWER_CHARS = 32_000


@dataclass(frozen=True)
class TokenUsage:
    """Token usage reported by an LLM provider, when available."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    @property
    def available(self) -> bool:
        return any(
            value is not None
            for value in (self.input_tokens, self.output_tokens, self.total_tokens)
        )


@dataclass(frozen=True)
class MetadataSnippet:
    """A compact metadata match. Never includes sample rows or credentials."""

    schema_name: str
    table_name: str | None = None
    column_name: str | None = None
    matched_name: str = ""

    def display_name(self) -> str:
        parts = [self.schema_name]
        if self.table_name:
            parts.append(self.table_name)
        if self.column_name:
            parts.append(self.column_name)
        return ".".join(parts)


@dataclass(frozen=True)
class AIContext:
    """Authorized analyst context for a single request.

    Does not load a full database catalog. Metadata is retrieved later through
    the metadata context provider when a user question needs it.
    """

    user_id: UUID
    workspace_id: UUID
    organization_id: UUID
    data_source_id: UUID
    data_source_name: str
    data_source_type: str
    workspace_name: str
    workspace_role: str | None
    allowed_capabilities: frozenset[str] = DEFAULT_AI_CAPABILITIES
    metadata: tuple[MetadataSnippet, ...] = ()
    conversation_id: UUID | None = None


@dataclass(frozen=True)
class AIRequest:
    message: str
    data_source_id: UUID
    request_id: str | None = None
    conversation_id: UUID | None = None


@dataclass(frozen=True)
class AIResponse:
    response: str
    model: str
    request_id: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    analysis: AIAnalysisResult | None = None
    intent: AIIntent | None = None
    plan: AIRequestPlan | None = None
    metadata_context: ResolvedMetadataContext | None = None


class AIAnalysisResult(BaseModel):
    """Normalized structured analyst output. Not a raw provider payload."""

    model_config = ConfigDict(extra="ignore")

    answer: str = Field(min_length=1, max_length=_MAX_ANALYSIS_ANSWER_CHARS)
    intent: str = Field(default="unknown", max_length=_MAX_INTENT_CHARS)
    requires_data_access: bool = False
    metadata_context: list[str] = Field(
        default_factory=list,
        max_length=_MAX_METADATA_ITEMS,
    )
    warnings: list[str] = Field(default_factory=list, max_length=_MAX_WARNING_ITEMS)
