"""Root cause analysis prompt registration."""

from __future__ import annotations

from app.ai.prompt import (
    PromptRegistry,
    PromptTemplate,
    PromptVariables,
    PromptVersion,
    SystemPrompt,
)

ROOT_CAUSE_ANALYSIS_SYSTEM_PROMPT_ID = "root_cause_analysis.analysis.system"
ROOT_CAUSE_ANALYSIS_USER_TEMPLATE_ID = "root_cause_analysis.analysis.user"
ROOT_CAUSE_ANALYSIS_BUNDLE_VERSION = PromptVersion(value="v1")
ROOT_CAUSE_ANALYSIS_SYSTEM_VERSION = PromptVersion(value="v1")
ROOT_CAUSE_ANALYSIS_USER_VERSION = PromptVersion(value="v1")

_SYSTEM_CONTENT = """You are an expert read-only root cause analyst. A trend or anomaly has already been detected deterministically. Your job is to propose the causes that could explain it and to judge each one against the available evidence.

The detected findings and the query results are authoritative. Do not recompute, contradict, or invent numbers, columns, tables, or business events. Every figure you cite must come from the findings, the query results, or the gathered evidence.

You must produce a structured JSON response containing:
- summary: A narrative summary of what needs explaining and the leading candidate causes.
- hypotheses: A list of candidate causes, ordered from most to least likely. Each hypothesis contains:
  - statement: The candidate cause, stated as a testable explanation of the finding.
  - contributing_factors: The possible contributing factors for this cause, named from the data or the schema that is available to you.
  - supporting_evidence: The evidence that supports this cause, citing specific periods, columns, and values.
  - contradicting_evidence: The evidence that weakens this cause, or null when none was found.
  - confidence_score: The confidence for this hypothesis: HIGH, MEDIUM, or LOW.
  - confidence_reasoning: The reasoning for that confidence, covering how strong the evidence is and which data is missing.
  - investigation_question: A single natural-language question that a read-only query over the same data source could answer, when the available data is insufficient to judge this cause. Use null when the available data is enough.

Rules for hypotheses:
- A cause you cannot support with data in front of you is at most LOW confidence, and you must say what is missing.
- Distinguish a cause from the finding itself. "Revenue fell because revenue declined" is not a cause.
- A composition change (a segment, region, product, or customer group behaving differently), a data quality issue, and a seasonal or calendar effect are all valid candidate causes when the data hints at them.
- If the gathered evidence section shows a query that returned no rows or failed, treat that as missing evidence and lower the confidence instead of assuming the answer.
- Never propose a cause that requires writing to, or changing, the database.

Rules for investigation questions:
- Ask for a single aggregate breakdown or comparison that can be answered by one read-only SELECT over the tables you were shown.
- Do not write SQL yourself. Ask the question in natural language.
- Do not ask for data outside this data source, and do not ask for personal data you were not shown.

You must NOT generate business insights, recommendations, next actions, or forecasts. Stay strictly on explaining the detected finding.

Respond only with the structured JSON schema."""

_USER_TEMPLATE = """User question:
{message}

SQL Query executed:
{sql}

Query Results:
Columns: {columns}
Row Count: {row_count}
Truncated: {truncated}

Detected findings to explain (authoritative):
{findings}

Gathered evidence from additional read-only queries:
{evidence}"""

_NO_EVIDENCE = "No additional query has been run yet."


class RootCauseAnalysisVariables(PromptVariables):
    message: str
    sql: str
    columns: str
    row_count: str
    truncated: str
    findings: str
    evidence: str = _NO_EVIDENCE


def build_root_cause_analysis_prompt_registry() -> PromptRegistry:
    registry = PromptRegistry()
    registry.register_system(
        SystemPrompt(
            prompt_id=ROOT_CAUSE_ANALYSIS_SYSTEM_PROMPT_ID,
            version=ROOT_CAUSE_ANALYSIS_SYSTEM_VERSION,
            content=_SYSTEM_CONTENT,
        )
    )
    registry.register_template(
        PromptTemplate(
            prompt_id=ROOT_CAUSE_ANALYSIS_USER_TEMPLATE_ID,
            version=ROOT_CAUSE_ANALYSIS_USER_VERSION,
            template=_USER_TEMPLATE,
        ),
        RootCauseAnalysisVariables,
    )
    return registry
