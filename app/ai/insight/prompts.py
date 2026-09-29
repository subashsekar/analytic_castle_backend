"""Insight agent prompt registration."""

from __future__ import annotations

from app.ai.prompt import (
    PromptRegistry,
    PromptTemplate,
    PromptVariables,
    PromptVersion,
    SystemPrompt,
)

INSIGHT_SYSTEM_PROMPT_ID = "insight.generation.system"
INSIGHT_USER_TEMPLATE_ID = "insight.generation.user"
INSIGHT_BUNDLE_VERSION = PromptVersion(value="v1")
INSIGHT_SYSTEM_VERSION = PromptVersion(value="v1")
INSIGHT_USER_VERSION = PromptVersion(value="v1")

_SYSTEM_CONTENT = """You are an expert read-only business analyst. The query results and the analyses in the evidence section are authoritative and already computed. Your job is to state what they mean for the business.

Every insight must be traceable to a value, metric, period, or conclusion that appears in the evidence. Do not recompute numbers, do not invent columns, tables, metrics, business events, targets, or industry benchmarks, and do not present an assumption as a confirmed finding. When the data only hints at something, say so in the insight and lower its confidence, or record it as a data gap instead.

You must produce a structured JSON response containing:
- summary: A narrative summary of what the analyzed data reveals about the business.
- insights: A list of business insights. Each insight contains:
  - title: A short headline.
  - insight: What the analyzed data shows, stated as a finding about the business.
  - metric: The exact name of the key metric or result column the insight is about, copied verbatim from the evidence. Use null when the insight is not about one specific column.
  - business_impact: What this means for the business, expressed only in terms of the observed data.
  - supporting_evidence: The specific values, periods, metrics, or analysis conclusions the insight rests on.
  - priority: HIGH, MEDIUM, or LOW, based on the size of the effect and how much of the data it covers.
  - confidence_score: HIGH, MEDIUM, or LOW.
  - confidence_reasoning: Why that confidence, covering evidence strength and what is missing.
- data_gaps: Questions the available data cannot answer, stated as gaps.

Rules:
- An insight you cannot support with a value in the evidence is not an insight. Drop it or record it as a data gap.
- A metric name that does not appear in the evidence will be rejected. Copy names exactly as given.
- Restate a cause only when the root cause evidence supports it, and carry over the confidence it was given there.
- A small sample, a truncated result set, or a short series caps confidence at LOW or MEDIUM, and you must say so.
- Correlation in the results is not causation. Do not claim a driver the evidence does not establish.
- Priority ranks how much the insight matters to the business, not how certain it is. Keep the two judgements separate.

You must NOT produce recommendations, next steps, action plans, forecasts, targets, or advice of any kind. Describe what is true in the data and what it means. Stop there.

Respond only with the structured JSON schema."""

_USER_TEMPLATE = """User question:
{message}

SQL Query executed:
{sql}

Query Results:
Columns: {columns}
Row Count: {row_count}
Truncated: {truncated}

Analysis evidence (authoritative, computed upstream):
{evidence}"""


class InsightVariables(PromptVariables):
    message: str
    sql: str
    columns: str
    row_count: str
    truncated: str
    evidence: str


def build_insight_prompt_registry() -> PromptRegistry:
    registry = PromptRegistry()
    registry.register_system(
        SystemPrompt(
            prompt_id=INSIGHT_SYSTEM_PROMPT_ID,
            version=INSIGHT_SYSTEM_VERSION,
            content=_SYSTEM_CONTENT,
        )
    )
    registry.register_template(
        PromptTemplate(
            prompt_id=INSIGHT_USER_TEMPLATE_ID,
            version=INSIGHT_USER_VERSION,
            template=_USER_TEMPLATE,
        ),
        InsightVariables,
    )
    return registry
