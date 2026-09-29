"""Shared planner test helpers and fixtures."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.ai.intent_types import (
    AIConfidence,
    AIIntentType,
    AIMetric,
    AIOperationType,
    LLMIntentDetection,
)
from app.ai.llm import AsyncLLMClient
from app.ai.planner_agent import PlannerAgent
from app.ai.state.models import AgentStatePayload
from app.ai.state.serialization import deserialize_payload
from tests.ai_fakes import FakeLLMProvider
from tests.test_supervisor import _client_config


def analytical_intent_detection() -> LLMIntentDetection:
    return LLMIntentDetection(
        intent=AIIntentType.ANALYTICAL_QUERY,
        operation=AIOperationType.AGGREGATE,
        subject="orders",
        metrics=[AIMetric(name="revenue")],
        confidence=AIConfidence.HIGH,
        requires_data_access=True,
        requires_metadata=True,
        requires_clarification=False,
    )


def configure_analytical_provider(provider: FakeLLMProvider) -> None:
    provider.intent = analytical_intent_detection()


def persisted_agent_payload(payload: dict[str, Any]) -> AgentStatePayload:
    return deserialize_payload(payload)


def plan_body(**overrides: object) -> dict[str, Any]:
    body: dict[str, Any] = {
        "intent": "ANALYTICAL_QUERY",
        "operations": ["SELECT", "AGGREGATE"],
        "required_capabilities": ["METADATA", "DATABASE", "AGGREGATION"],
        "required_data": [
            {"name": "orders", "kind": "TABLE"},
            {"name": "revenue", "kind": "METRIC"},
        ],
        "action_steps": [
            {
                "step_order": 1,
                "action": "RESOLVE_METADATA",
                "description": "Resolve order table metadata",
                "tool": "metadata.lookup",
            },
            {
                "step_order": 2,
                "action": "AGGREGATE",
                "description": "Aggregate revenue by region",
                "tool": "postgres.query",
            },
        ],
        "missing_information": [],
        "requires_clarification": False,
        "unsupported": False,
    }
    body.update(overrides)
    return body


def success_response(content: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "gen-planner-1",
        "model": "openai/gpt-4o-mini",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(content),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def mock_planner_client() -> AsyncLLMClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=success_response(plan_body()))

    return AsyncLLMClient(_client_config(), transport=httpx.MockTransport(handler))


def mock_planner_llm_client(handler: Any) -> AsyncLLMClient:
    transport = httpx.MockTransport(handler)
    return AsyncLLMClient(_client_config(), transport=transport)


@pytest.fixture
def mock_planner_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent supervised orchestrator tests from calling a live planner LLM."""

    async def _create_analysis_plan(**_kwargs: Any) -> Any:
        from app.ai.intent_types import AIPlanCapability
        from app.ai.planner_agent.models import LLMPlannerOutput
        from app.ai.planner_agent.validation import normalize_plan

        intent = _kwargs["intent"]
        plan = normalize_plan(
            llm_result=LLMPlannerOutput.model_validate(
                plan_body(intent=intent.intent.value)
            ),
            intent=intent,
        )
        assert AIPlanCapability.DATABASE in plan.request_plan.required_capabilities
        return plan

    monkeypatch.setattr(
        "app.ai.planner_agent.service.create_analysis_plan",
        _create_analysis_plan,
    )


def planner_agent_for_session(db_session: Any) -> PlannerAgent:
    return PlannerAgent(db_session, llm_client=mock_planner_client())


def advance_session_to_planning(
    service: Any,
    snapshot: Any,
    *,
    workspace_id: Any,
    user_id: Any,
) -> Any:
    from app.ai.state import SetPhaseTransition
    from app.enums import AgentPhase

    snapshot = service.update_agent_state(
        snapshot.session_id,
        SetPhaseTransition(phase=AgentPhase.INTENT),
        workspace_id=workspace_id,
        user_id=user_id,
        expected_version=snapshot.agent_state.version,
    )
    return service.update_agent_state(
        snapshot.session_id,
        SetPhaseTransition(phase=AgentPhase.PLANNING),
        workspace_id=workspace_id,
        user_id=user_id,
        expected_version=snapshot.agent_state.version,
    )
