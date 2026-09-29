"""Planner prompt registration for analysis planning."""

from __future__ import annotations

from app.ai.prompt import (
    PromptRegistry,
    PromptTemplate,
    PromptVariables,
    PromptVersion,
    SystemPrompt,
)

PLANNER_SYSTEM_PROMPT_ID = "planner.analysis.system"
PLANNER_USER_TEMPLATE_ID = "planner.analysis.user"
PLANNER_BUNDLE_VERSION = PromptVersion(value="v1")
PLANNER_SYSTEM_VERSION = PromptVersion(value="v1")
PLANNER_USER_VERSION = PromptVersion(value="v1")

_SYSTEM_CONTENT = """You are an analytical planning agent for a read-only data analyst.

Given the user's message and detected intent, produce a structured analysis plan:
- Confirm or refine the analytical intent
- List required data concepts (tables, columns, metrics, dimensions, filters, time ranges)
- Define an ordered sequence of planning actions (not execution)
- Identify required capabilities (METADATA, DATABASE, SAMPLE_DATA, AGGREGATION, TIME_FILTER, RELATIONSHIPS)
- List operations needed (SELECT, FILTER, AGGREGATE, GROUP_BY, SORT, RANK, COMPARE, TREND, SUMMARY, etc.)
- Identify missing information that blocks planning
- Set requires_clarification=true when critical details are missing
- Set unsupported=true only for requests outside read-only analytical scope

For each action step, optionally name a descriptive tool reference:
- metadata.lookup
- postgres.query
- sample_data.read

You must NOT generate SQL, execute queries, or return data. Planning only.

Respond only with the structured JSON schema."""

_USER_TEMPLATE = """User message:
{message}

Detected intent: {intent}
Intent operation: {operation}
Intent subject: {subject}
Requires data access: {requires_data_access}
Requires metadata: {requires_metadata}
Data source available: {has_data_source}
Data source name: {data_source_name}
Metrics: {metrics}
Dimensions: {dimensions}
Filters: {filters}
Time range: {time_range}"""


class PlannerAnalysisVariables(PromptVariables):
    message: str
    intent: str
    operation: str
    subject: str
    requires_data_access: str
    requires_metadata: str
    has_data_source: str
    data_source_name: str
    metrics: str
    dimensions: str
    filters: str
    time_range: str


def build_planner_prompt_registry() -> PromptRegistry:
    registry = PromptRegistry()
    registry.register_system(
        SystemPrompt(
            prompt_id=PLANNER_SYSTEM_PROMPT_ID,
            version=PLANNER_SYSTEM_VERSION,
            content=_SYSTEM_CONTENT,
        )
    )
    registry.register_template(
        PromptTemplate(
            prompt_id=PLANNER_USER_TEMPLATE_ID,
            version=PLANNER_USER_VERSION,
            template=_USER_TEMPLATE,
        ),
        PlannerAnalysisVariables,
    )
    return registry
