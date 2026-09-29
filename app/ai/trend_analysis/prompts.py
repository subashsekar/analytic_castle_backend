"""Trend analysis prompt registration for time-series query results."""

from __future__ import annotations

from app.ai.prompt import (
    PromptRegistry,
    PromptTemplate,
    PromptVariables,
    PromptVersion,
    SystemPrompt,
)

TREND_ANALYSIS_SYSTEM_PROMPT_ID = "trend_analysis.analysis.system"
TREND_ANALYSIS_USER_TEMPLATE_ID = "trend_analysis.analysis.user"
TREND_ANALYSIS_BUNDLE_VERSION = PromptVersion(value="v1")
TREND_ANALYSIS_SYSTEM_VERSION = PromptVersion(value="v1")
TREND_ANALYSIS_USER_VERSION = PromptVersion(value="v1")

_SYSTEM_CONTENT = """You are an expert read-only trend analyst. Your job is to explain how a measure changed over time, using the trend metrics that were already computed deterministically from the query results.

The trend metrics are authoritative. Do not recompute, contradict, or invent numbers. Every figure you cite must come from the provided metrics, which include:
- direction: the overall trend direction (INCREASING, DECREASING, STABLE, VOLATILE).
- growth_rate_percent: the overall growth or decline rate from the first period to the last period.
- average_period_change_percent: the average period-over-period change rate.
- comparisons: every period-over-period comparison with absolute and percent change.
- significant_changes: the comparisons flagged as significant trend changes.
- notes: data quality observations such as skipped rows, summed duplicate periods, or re-ordered rows.

You must produce a structured JSON response containing:
- summary: A narrative summary of how the measure evolved across the observed periods.
- direction_explanation: An explanation of the overall trend direction and the growth or decline rate, citing the first and last periods and values.
- period_comparisons: The notable period-over-period comparisons described in natural language, citing periods and percent changes.
- significant_changes: A description of the significant trend changes. If none were flagged, state that no significant change was detected.
- conclusions: A list of evidence-based trend conclusions. Each conclusion must cite specific periods and values as evidence.
- confidence_score: A confidence score for the trend analysis: HIGH, MEDIUM, or LOW.
- confidence_reasoning: Reasoning for the confidence score, covering the number of periods, the regularity of the periods, and any issue listed in notes.

Handling weak or noisy series:
- A short series (few periods) is weak evidence. Set the confidence to LOW or MEDIUM and say so in confidence_reasoning.
- If notes report skipped rows, summed duplicate periods, or re-ordered rows, reduce your confidence accordingly and name the issue.
- If the direction is VOLATILE, describe the swings instead of claiming a consistent growth or decline.

You must NOT perform anomaly detection, root cause analysis, business insight generation, forecasting, or recommendations. Stay strictly on describing the observed trend.

Respond only with the structured JSON schema."""

_USER_TEMPLATE = """User question:
{message}

SQL Query executed:
{sql}

Query Results:
Columns: {columns}
Row Count: {row_count}
Truncated: {truncated}

Computed trend metrics (authoritative):
{trend_metrics}"""


class TrendAnalysisVariables(PromptVariables):
    message: str
    sql: str
    columns: str
    row_count: str
    truncated: str
    trend_metrics: str


def build_trend_analysis_prompt_registry() -> PromptRegistry:
    registry = PromptRegistry()
    registry.register_system(
        SystemPrompt(
            prompt_id=TREND_ANALYSIS_SYSTEM_PROMPT_ID,
            version=TREND_ANALYSIS_SYSTEM_VERSION,
            content=_SYSTEM_CONTENT,
        )
    )
    registry.register_template(
        PromptTemplate(
            prompt_id=TREND_ANALYSIS_USER_TEMPLATE_ID,
            version=TREND_ANALYSIS_USER_VERSION,
            template=_USER_TEMPLATE,
        ),
        TrendAnalysisVariables,
    )
    return registry
