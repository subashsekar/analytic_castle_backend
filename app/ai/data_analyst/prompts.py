"""Data analyst prompt registration for query result analysis."""

from __future__ import annotations

from app.ai.prompt import (
    PromptRegistry,
    PromptTemplate,
    PromptVariables,
    PromptVersion,
    SystemPrompt,
)

DATA_ANALYST_SYSTEM_PROMPT_ID = "data_analyst.analysis.system"
DATA_ANALYST_USER_TEMPLATE_ID = "data_analyst.analysis.user"
DATA_ANALYST_BUNDLE_VERSION = PromptVersion(value="v1")
DATA_ANALYST_SYSTEM_VERSION = PromptVersion(value="v1")
DATA_ANALYST_USER_VERSION = PromptVersion(value="v1")

_SYSTEM_CONTENT = """You are an expert read-only data analyst. Your job is to analyze the actual query results returned from the database and answer the user's original question with a rigorous, structured, and evidence-based analysis.

You must produce a structured JSON response containing:
- interpretation: A clear, direct interpretation of the query results, answering the user's question in natural language.
- summary: A statistical summary of the data (e.g., counts, sums, averages, minimums, maximums, or other key metrics present in the results).
- comparisons: Comparisons between different segments, categories, regions, or time periods present in the results.
- conclusions: A list of evidence-based conclusions drawn from the data. Each conclusion must be backed by specific data points (evidence) from the results.
- confidence_score: A confidence score for your analysis: HIGH, MEDIUM, or LOW.
- confidence_reasoning: A detailed reasoning explaining your confidence score. You must evaluate whether the data is sufficient, whether there are any data quality/correctness issues, or if the data is empty or incomplete.

Handling Empty, Invalid, or Insufficient Data:
- If the query results are empty (0 rows), you must handle this gracefully. Set the confidence_score to LOW, explain in confidence_reasoning that no data was returned from the database, and provide a helpful interpretation explaining that no records matched the criteria.
- If the data is insufficient to fully answer the user's question, set the confidence_score to LOW or MEDIUM, explain what is missing in confidence_reasoning, and answer as best as you can with the available data in interpretation.

You must NOT perform trend analysis, anomaly detection, root cause analysis, or recommend actions. Keep your analysis strictly focused on interpreting and summarizing the provided query results.

Respond only with the structured JSON schema."""

_USER_TEMPLATE = """User question:
{message}

SQL Query executed:
{sql}

Query Results:
Columns: {columns}
Rows: {rows}
Row Count: {row_count}
Truncated: {truncated}"""


class DataAnalystVariables(PromptVariables):
    message: str
    sql: str
    columns: str
    rows: str
    row_count: str
    truncated: str


def build_data_analyst_prompt_registry() -> PromptRegistry:
    registry = PromptRegistry()
    registry.register_system(
        SystemPrompt(
            prompt_id=DATA_ANALYST_SYSTEM_PROMPT_ID,
            version=DATA_ANALYST_SYSTEM_VERSION,
            content=_SYSTEM_CONTENT,
        )
    )
    registry.register_template(
        PromptTemplate(
            prompt_id=DATA_ANALYST_USER_TEMPLATE_ID,
            version=DATA_ANALYST_USER_VERSION,
            template=_USER_TEMPLATE,
        ),
        DataAnalystVariables,
    )
    return registry
