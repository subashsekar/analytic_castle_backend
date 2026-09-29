"""LLM-backed request classification for the supervisor."""

from __future__ import annotations

import logging

from app.ai.llm import (
    AsyncLLMClient,
    LLMError,
    LLMRequest,
    LLMResponseValidationError,
)
from app.ai.prompt import (
    PromptRegistry,
    bundle_to_llm_messages,
    structured_output_from_model,
)
from app.ai.supervisor.errors import SupervisorClassificationError, SupervisorLLMError
from app.ai.supervisor.logging_helpers import classification_log_context
from app.ai.supervisor.models import LLMRequestClassification, RequestClassification
from app.ai.supervisor.prompts import (
    SUPERVISOR_BUNDLE_VERSION,
    SUPERVISOR_SYSTEM_PROMPT_ID,
    SUPERVISOR_USER_TEMPLATE_ID,
    SupervisorClassificationVariables,
)
from app.ai.supervisor.validation import (
    normalize_classification,
    parse_llm_classification,
)
from app.enums import AgentPhase

logger = logging.getLogger(__name__)


async def classify_request(
    *,
    client: AsyncLLMClient,
    registry: PromptRegistry,
    message: str,
    current_phase: AgentPhase,
    has_data_source: bool,
) -> RequestClassification:
    bundle = registry.build_bundle(
        bundle_version=SUPERVISOR_BUNDLE_VERSION,
        system_prompt_id=SUPERVISOR_SYSTEM_PROMPT_ID,
        user_template_id=SUPERVISOR_USER_TEMPLATE_ID,
        user_variables=SupervisorClassificationVariables(
            current_phase=current_phase.value,
            has_data_source="true" if has_data_source else "false",
            message=message,
        ),
        structured_output=structured_output_from_model(LLMRequestClassification),
    )
    request = LLMRequest(
        model=client.config.model,
        messages=bundle_to_llm_messages(bundle),
        structured_output=bundle.structured_output,
    )
    try:
        response = await client.complete(request)
    except LLMError as exc:
        raise SupervisorLLMError(str(exc)) from exc

    if not response.content:
        raise SupervisorLLMError("Classification response was empty")

    try:
        llm_result = parse_llm_classification(response.content)
        classification = normalize_classification(llm_result)
    except (LLMResponseValidationError, SupervisorClassificationError) as exc:
        raise SupervisorLLMError(str(exc)) from exc
    logger.info(
        "supervisor request classified",
        extra=classification_log_context(classification),
    )
    return classification
