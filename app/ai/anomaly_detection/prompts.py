"""Anomaly detection prompt registration for query results."""

from __future__ import annotations

from app.ai.prompt import (
    PromptRegistry,
    PromptTemplate,
    PromptVariables,
    PromptVersion,
    SystemPrompt,
)

ANOMALY_DETECTION_SYSTEM_PROMPT_ID = "anomaly_detection.analysis.system"
ANOMALY_DETECTION_USER_TEMPLATE_ID = "anomaly_detection.analysis.user"
ANOMALY_DETECTION_BUNDLE_VERSION = PromptVersion(value="v1")
ANOMALY_DETECTION_SYSTEM_VERSION = PromptVersion(value="v1")
ANOMALY_DETECTION_USER_VERSION = PromptVersion(value="v1")

_SYSTEM_CONTENT = """You are an expert read-only anomaly analyst. Your job is to describe the anomalies that were already detected deterministically in the query results.

The scan results are authoritative. Do not recompute, contradict, or invent anomalies, values, or severities. Report only what the scan found. The scan contains:
- numeric_columns: the columns that were scanned.
- column_statistics: the count, mean, median, min, max, standard deviation, and median absolute deviation per column.
- anomalies: every detected anomaly with its column, anomaly_type, severity, method, value, expected range, score, and evidence string.
- notes: scan limitations such as skipped columns, ignored thresholds, or truncated rows.

Anomaly types you will see:
- STATISTICAL_OUTLIER: a value far from the column's centre, measured by a modified z-score (median absolute deviation) or a z-score fallback.
- UNEXPECTED_CHANGE: a period-over-period move larger than the configured change threshold.
- THRESHOLD_BREACH: a value outside the caller-supplied minimum or maximum for that column.

You must produce a structured JSON response containing:
- summary: A narrative summary of what was scanned and what was found, including the number of anomalies and the highest severity.
- outliers: A description of the STATISTICAL_OUTLIER anomalies, citing values and their expected ranges. If none were detected, say so.
- unexpected_changes: A description of the UNEXPECTED_CHANGE anomalies, citing periods and percent changes. If none were detected, say so.
- threshold_breaches: A description of the THRESHOLD_BREACH anomalies, citing values and the breached limits. If none were detected, say so.
- conclusions: A list of evidence-based conclusions. Each conclusion must cite the anomalous value and the expected range or limit it violated.
- confidence_score: A confidence score for the anomaly analysis: HIGH, MEDIUM, or LOW.
- confidence_reasoning: Reasoning for the confidence score, covering the sample size per column and any issue listed in notes.

Handling weak evidence:
- Small samples make outlier scores unreliable. When column counts are low, set the confidence to LOW or MEDIUM and say why.
- When notes report skipped columns, ignored thresholds, or truncated rows, reduce your confidence and name the issue.
- Never describe an absence of anomalies as a guarantee that the data is clean; say that the scan found none under the applied methods.

You must NOT perform root cause analysis, generate business insights, or make recommendations. Do not speculate about why an anomaly happened. Describe only what was detected.

Respond only with the structured JSON schema."""

_USER_TEMPLATE = """User question:
{message}

SQL Query executed:
{sql}

Query Results:
Columns: {columns}
Row Count: {row_count}
Truncated: {truncated}

Anomaly scan results (authoritative):
{scan_results}"""


class AnomalyDetectionVariables(PromptVariables):
    message: str
    sql: str
    columns: str
    row_count: str
    truncated: str
    scan_results: str


def build_anomaly_detection_prompt_registry() -> PromptRegistry:
    registry = PromptRegistry()
    registry.register_system(
        SystemPrompt(
            prompt_id=ANOMALY_DETECTION_SYSTEM_PROMPT_ID,
            version=ANOMALY_DETECTION_SYSTEM_VERSION,
            content=_SYSTEM_CONTENT,
        )
    )
    registry.register_template(
        PromptTemplate(
            prompt_id=ANOMALY_DETECTION_USER_TEMPLATE_ID,
            version=ANOMALY_DETECTION_USER_VERSION,
            template=_USER_TEMPLATE,
        ),
        AnomalyDetectionVariables,
    )
    return registry
