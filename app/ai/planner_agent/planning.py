"""LLM-backed analysis planning for the planner agent."""

from __future__ import annotations

import logging

from app.ai.intent_types import AIIntent
from app.ai.llm import (
    AsyncLLMClient,
    LLMError,
    LLMRequest,
    LLMResponseValidationError,
)
from app.ai.planner_agent.errors import PlannerLLMError, PlannerValidationError
from app.ai.planner_agent.logging_helpers import plan_log_context
from app.ai.planner_agent.models import AnalysisPlan, LLMPlannerOutput
from app.ai.planner_agent.prompts import (
    PLANNER_BUNDLE_VERSION,
    PLANNER_SYSTEM_PROMPT_ID,
    PLANNER_USER_TEMPLATE_ID,
    PlannerAnalysisVariables,
)
from app.ai.planner_agent.validation import normalize_plan, parse_llm_plan
from app.ai.prompt import (
    PromptRegistry,
    bundle_to_llm_messages,
    structured_output_from_model,
)

logger = logging.getLogger(__name__)


async def create_analysis_plan(
    *,
    client: AsyncLLMClient,
    registry: PromptRegistry,
    intent: AIIntent,
    message: str,
    has_data_source: bool,
    data_source_name: str | None,
) -> AnalysisPlan:
    bundle = registry.build_bundle(
        bundle_version=PLANNER_BUNDLE_VERSION,
        system_prompt_id=PLANNER_SYSTEM_PROMPT_ID,
        user_template_id=PLANNER_USER_TEMPLATE_ID,
        user_variables=PlannerAnalysisVariables(
            message=message,
            intent=intent.intent.value,
            operation=intent.operation.value if intent.operation else "none",
            subject=intent.subject or "none",
            requires_data_access="true" if intent.requires_data_access else "false",
            requires_metadata="true" if intent.requires_metadata else "false",
            has_data_source="true" if has_data_source else "false",
            data_source_name=data_source_name or "none",
            metrics=", ".join(metric.name for metric in intent.metrics) or "none",
            dimensions=", ".join(dim.name for dim in intent.dimensions) or "none",
            filters=", ".join(item.field for item in intent.filters) or "none",
            time_range=intent.time_range.preset.value if intent.time_range else "none",
        ),
        structured_output=structured_output_from_model(LLMPlannerOutput),
    )
    request = LLMRequest(
        model=client.config.model,
        messages=bundle_to_llm_messages(bundle),
        structured_output=bundle.structured_output,
    )
    try:
        response = await client.complete(request)
    except LLMError as exc:
        raise PlannerLLMError(str(exc)) from exc

    if not response.content:
        raise PlannerLLMError("Planner response was empty")

    try:
        llm_result = parse_llm_plan(response.content)
        plan = normalize_plan(llm_result=llm_result, intent=intent)
    except (LLMResponseValidationError, PlannerValidationError) as exc:
        raise PlannerLLMError(str(exc)) from exc

    logger.info("planner analysis plan created", extra=plan_log_context(plan))
    return plan
