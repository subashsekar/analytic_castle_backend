"""Question profiling for chat analysis: which analysis the question needs.

Combines the validated intent/plan with generic language cues so SQL
generation receives a structured spec and Phase 8 runs only relevant agents.
Nothing here names business metrics, tables, or columns.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from datetime import date

from app.ai.intent_types import AIIntent, AIIntentType, AIPlanOperation, AIRequestPlan
from app.ai.metadata_types import ResolvedMetadataContext
from app.ai.state.models import ConversationContextData


class QuestionKind(str, enum.Enum):
    LOOKUP = "LOOKUP"
    AGGREGATE = "AGGREGATE"
    BREAKDOWN = "BREAKDOWN"
    RANKING = "RANKING"
    COMPARISON = "COMPARISON"
    TREND = "TREND"
    ANOMALY = "ANOMALY"
    DIAGNOSTIC = "DIAGNOSTIC"
    INSIGHT = "INSIGHT"
    RECOMMENDATION = "RECOMMENDATION"


def _cue(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(patterns), re.IGNORECASE)


_CUES: dict[QuestionKind, re.Pattern[str]] = {
    QuestionKind.DIAGNOSTIC: _cue(
        r"\bwhy\b",
        r"\bcaus(e|ed|es|ing)\b",
        r"\breasons?\b",
        r"\bdriv(e|er|ers|ing|en)\b",
        r"\bexplain\b",
        r"\bcontribut",
        r"\bwhat happened\b",
        r"\broot cause\b",
    ),
    QuestionKind.ANOMALY: _cue(
        r"\banomal",
        r"\boutliers?\b",
        r"\bunusual\b",
        r"\bspikes?\b",
        r"\babnormal",
        r"\bunexpected\b",
        r"\birregular",
    ),
    QuestionKind.TREND: _cue(
        r"\btrend",
        r"\bover time\b",
        r"\b(month|week|quarter|year|day)[- ]over[- ](month|week|quarter|year|day)\b",
        r"\b(mom|yoy|qoq|wow)\b",
        r"\bgrow(th|ing|n)?\b",
        r"\bdeclin",
        r"\btrajectory\b",
        r"\b(daily|weekly|monthly|quarterly|yearly|annually)\b",
        r"\b(by|per|each) (day|week|month|quarter|year)\b",
    ),
    QuestionKind.COMPARISON: _cue(
        r"\bvs\.?\b",
        r"\bversus\b",
        r"\bcompar",
        r"\bdifference between\b",
        r"\brelative to\b",
        r"\bagainst\b",
    ),
    QuestionKind.RANKING: _cue(
        r"\btop\b",
        r"\bbottom\b",
        r"\bbest\b",
        r"\bworst\b",
        r"\bhighest\b",
        r"\blowest\b",
        r"\brank",
        r"\bmost\b",
        r"\bleast\b",
        r"\blargest\b",
        r"\bsmallest\b",
    ),
    QuestionKind.BREAKDOWN: _cue(
        r"\bby\s+(?!day\b|week\b|month\b|quarter\b|year\b)\w+",
        r"\bbreak\s*down\b",
        r"\bbreakdown\b",
        r"\bsplit\b",
        r"\bdistribution\b",
        r"\bshare\b",
        r"\bper\s+\w+",
        r"\bmix\b",
        r"\bsegment",
    ),
    QuestionKind.INSIGHT: _cue(
        r"\binsight",
        r"\bsummar",
        r"\boverview\b",
        r"\bperform",
        r"\bhow (are|is|did|was|were) .* (doing|going)\b",
        r"\btakeaways?\b",
        r"\bhighlights?\b",
    ),
    QuestionKind.RECOMMENDATION: _cue(
        r"\brecommend",
        r"\bshould (we|i)\b",
        r"\bwhat (can|could|should) (we|i) do\b",
        r"\bhow (can|could|do|should) (we|i) (improve|increase|reduce|fix|grow|boost|lower)\b",
        r"\bsuggest",
        r"\bnext steps?\b",
        r"\baction(s|able)?\b",
        r"\bimprove\b",
    ),
}

_INTENT_KINDS: dict[AIIntentType, QuestionKind] = {
    AIIntentType.DATA_LOOKUP: QuestionKind.LOOKUP,
    AIIntentType.AGGREGATION: QuestionKind.AGGREGATE,
    AIIntentType.COMPARISON: QuestionKind.COMPARISON,
    AIIntentType.TREND_ANALYSIS: QuestionKind.TREND,
    AIIntentType.RANKING: QuestionKind.RANKING,
    AIIntentType.SUMMARY: QuestionKind.INSIGHT,
}

_OPERATION_KINDS: dict[AIPlanOperation, QuestionKind] = {
    AIPlanOperation.TREND: QuestionKind.TREND,
    AIPlanOperation.RANK: QuestionKind.RANKING,
    AIPlanOperation.COMPARE: QuestionKind.COMPARISON,
    AIPlanOperation.GROUP_BY: QuestionKind.BREAKDOWN,
    AIPlanOperation.SUMMARY: QuestionKind.INSIGHT,
}


@dataclass(frozen=True)
class AnalysisProfile:
    """Question kinds plus the Phase 8 agents they justify."""

    kinds: frozenset[QuestionKind]
    run_data_analyst: bool = False
    run_trend: bool = False
    run_anomaly: bool = False
    run_root_cause: bool = False
    run_insight: bool = False
    run_recommendation: bool = False

    @property
    def uses_llm_agents(self) -> bool:
        return any(
            (
                self.run_data_analyst,
                self.run_trend,
                self.run_anomaly,
                self.run_root_cause,
                self.run_insight,
                self.run_recommendation,
            )
        )

    @property
    def is_diagnostic(self) -> bool:
        return QuestionKind.DIAGNOSTIC in self.kinds

    def labels(self) -> list[str]:
        return sorted(kind.value for kind in self.kinds)


def build_analysis_profile(
    message: str,
    *,
    intent: AIIntent | None = None,
    plan: AIRequestPlan | None = None,
) -> AnalysisProfile:
    kinds: set[QuestionKind] = {
        kind for kind, pattern in _CUES.items() if pattern.search(message or "")
    }
    if intent is not None and intent.intent in _INTENT_KINDS:
        kinds.add(_INTENT_KINDS[intent.intent])
    if intent is not None and intent.dimensions:
        kinds.add(QuestionKind.BREAKDOWN)
    if plan is not None:
        for operation in plan.operations:
            if operation in _OPERATION_KINDS:
                kinds.add(_OPERATION_KINDS[operation])
    if not kinds:
        kinds.add(QuestionKind.AGGREGATE)

    diagnostic = QuestionKind.DIAGNOSTIC in kinds
    anomaly = QuestionKind.ANOMALY in kinds
    trend = QuestionKind.TREND in kinds
    recommendation = QuestionKind.RECOMMENDATION in kinds
    insight = QuestionKind.INSIGHT in kinds
    explanatory = kinds & {
        QuestionKind.COMPARISON,
        QuestionKind.BREAKDOWN,
        QuestionKind.RANKING,
    }
    return AnalysisProfile(
        kinds=frozenset(kinds),
        run_data_analyst=bool(explanatory or insight) and not diagnostic,
        run_trend=trend or anomaly or diagnostic,
        run_anomaly=anomaly or diagnostic or trend,
        run_root_cause=diagnostic,
        run_insight=insight or diagnostic or recommendation,
        run_recommendation=recommendation,
    )


def build_analysis_spec(
    *,
    intent: AIIntent | None,
    plan: AIRequestPlan | None,
    metadata: ResolvedMetadataContext,
    profile: AnalysisProfile,
    today: date | None = None,
) -> str:
    """Structured, schema-grounded spec for SQL generation/correction prompts."""
    lines: list[str] = []
    if plan is not None and plan.operations:
        lines.append("Operations: " + ", ".join(op.value for op in plan.operations[:8]))
    lines.append("Question type: " + ", ".join(profile.labels()))
    lines.append(f"Today: {(today or date.today()).isoformat()}")
    if intent is not None:
        if intent.metrics:
            lines.append(
                "Metrics: "
                + "; ".join(
                    f"{m.name} ({m.aggregation.value})" if m.aggregation.value != "NONE" else m.name
                    for m in intent.metrics[:10]
                )
            )
        if intent.dimensions:
            lines.append("Dimensions: " + "; ".join(d.name for d in intent.dimensions[:10]))
        if intent.filters:
            rendered = []
            for item in intent.filters[:10]:
                if item.values:
                    value = "[" + ", ".join(str(v) for v in item.values[:10]) + "]"
                elif item.start is not None or item.end is not None:
                    value = f"{item.start}..{item.end}"
                else:
                    value = str(item.value)
                rendered.append(f"{item.field} {item.operator.value} {value}")
            lines.append("Filters: " + "; ".join(rendered))
        if intent.time_range is not None:
            tr = intent.time_range
            if tr.start_date and tr.end_date:
                lines.append(
                    f"Time range: {tr.start_date.isoformat()} to {tr.end_date.isoformat()} (inclusive)"
                )
            else:
                lines.append(f"Time range: {tr.preset.value} (relative to Today)")
        if intent.sort is not None:
            lines.append(f"Sort: {intent.sort.field} {intent.sort.direction.value}")
        limit = intent.safe_limit or intent.requested_limit
        if limit:
            lines.append(f"Row limit: {limit}")

    mappings = _concept_mappings(metadata)
    if mappings:
        lines.append("Catalog matches for requested concepts: " + "; ".join(mappings))
    time_columns = [
        f"{c.schema_name}.{c.table_name}.{c.column_name}"
        for c in metadata.resolved_time_columns[:5]
    ]
    if time_columns:
        lines.append("Date/time columns: " + ", ".join(time_columns))
    if metadata.unresolved_concepts:
        lines.append(
            "Concepts with no catalog match (do not substitute other columns): "
            + ", ".join(metadata.unresolved_concepts[:10])
        )
    if profile.is_diagnostic:
        lines.append(
            "Diagnostic question: include the comparison period and a breakdown by "
            "relevant listed categorical columns so drivers can be measured."
        )
    return "\n".join(lines)


def _concept_mappings(metadata: ResolvedMetadataContext) -> list[str]:
    out: list[str] = []
    for label, group in (
        ("metric", metadata.resolved_metrics),
        ("dimension", metadata.resolved_dimensions),
        ("filter", metadata.resolved_filters),
    ):
        for item in group[:10]:
            if not item.candidates:
                continue
            targets = ", ".join(
                f"{c.schema_name}.{c.table_name}.{c.column_name}" for c in item.candidates[:3]
            )
            flag = " (ambiguous)" if item.ambiguous else ""
            out.append(f"{label} '{item.requested}' -> {targets}{flag}")
    return out


def build_conversation_context(
    conversation: ConversationContextData | None,
    *,
    current_message: str,
    max_messages: int = 6,
    max_chars_per_message: int = 400,
) -> str | None:
    """Compact prior turns so SQL generation can resolve follow-up references."""
    if conversation is None or not conversation.messages:
        return None
    prior = list(conversation.messages)
    stripped = (current_message or "").strip()
    if prior and prior[-1].content.strip() == stripped:
        prior = prior[:-1]
    if not prior:
        return None
    lines: list[str] = []
    for message in prior[-max_messages:]:
        role = getattr(message.role, "value", str(message.role))
        text = " ".join(message.content.split())
        if len(text) > max_chars_per_message:
            text = text[: max_chars_per_message - 3] + "..."
        lines.append(f"{role}: {text}")
    return "\n".join(lines)
