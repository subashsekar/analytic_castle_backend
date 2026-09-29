"""Supervisor agent for request classification and workflow routing."""

from app.ai.supervisor.classification import classify_request
from app.ai.supervisor.errors import (
    InvalidRoutingError,
    SupervisorAuthorizationError,
    SupervisorClassificationError,
    SupervisorConfigurationError,
    SupervisorError,
    SupervisorErrorCode,
    SupervisorLLMError,
)
from app.ai.supervisor.logging_helpers import (
    classification_log_context,
    decision_log_context,
    routing_log_context,
)
from app.ai.supervisor.models import (
    ClassificationConfidence,
    LLMRequestClassification,
    RequestCategory,
    RequestClassification,
    SuperviseAdvanceParams,
    SuperviseMessageParams,
    SupervisorAction,
    SupervisorDecision,
    SupervisorResult,
)
from app.ai.supervisor.prompts import (
    SUPERVISOR_BUNDLE_VERSION,
    SUPERVISOR_SYSTEM_PROMPT_ID,
    SUPERVISOR_USER_TEMPLATE_ID,
    SupervisorClassificationVariables,
    build_supervisor_prompt_registry,
)
from app.ai.supervisor.routing import resolve_advance_routing, resolve_routing
from app.ai.supervisor.service import SupervisorAgent
from app.ai.supervisor.validation import (
    build_decision,
    classification_from_policy,
    ensure_routing_allowed,
    normalize_classification,
    parse_llm_classification,
)
from app.ai.supervisor.workflow import advance_until_complete

__all__ = [
    "SUPERVISOR_BUNDLE_VERSION",
    "SUPERVISOR_SYSTEM_PROMPT_ID",
    "SUPERVISOR_USER_TEMPLATE_ID",
    "ClassificationConfidence",
    "InvalidRoutingError",
    "LLMRequestClassification",
    "RequestCategory",
    "RequestClassification",
    "SuperviseAdvanceParams",
    "SuperviseMessageParams",
    "SupervisorAction",
    "SupervisorAgent",
    "SupervisorAuthorizationError",
    "SupervisorClassificationError",
    "SupervisorClassificationVariables",
    "SupervisorConfigurationError",
    "SupervisorDecision",
    "SupervisorError",
    "SupervisorErrorCode",
    "SupervisorLLMError",
    "SupervisorResult",
    "advance_until_complete",
    "build_decision",
    "build_supervisor_prompt_registry",
    "classification_from_policy",
    "classification_log_context",
    "classify_request",
    "decision_log_context",
    "ensure_routing_allowed",
    "normalize_classification",
    "parse_llm_classification",
    "resolve_advance_routing",
    "resolve_routing",
    "routing_log_context",
]
