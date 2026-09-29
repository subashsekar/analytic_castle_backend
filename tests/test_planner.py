from __future__ import annotations

import json
import logging

import httpx
import pytest
from sqlalchemy.orm import Session

from app.ai.intent_types import (
    AIConfidence,
    AIIntent,
    AIIntentType,
    AIOperationType,
    AIPlanCapability,
    AIPlanOperation,
)
from app.ai.llm import LLMResponseValidationError
from app.ai.planner_agent import (
    AnalysisPlan,
    LLMPlannerOutput,
    PlanCreateParams,
    PlannerActionKind,
    PlannerAgent,
    PlannerAuthorizationError,
    PlannerLLMError,
    PlannerToolRef,
    PlannerValidationError,
    RequiredDataKind,
    apply_planner_intent,
    build_planner_prompt_registry,
    create_analysis_plan,
    map_planner_error,
    normalize_plan,
    parse_action_steps,
    parse_llm_plan,
    parse_required_data,
    plan_custom_entries,
    plan_from_intent,
    plan_log_context,
)
from app.ai.state import (
    AnalysisSessionNotFoundError,
    SetPhaseTransition,
)
from app.core.logging import RedactingFilter, redact_secret
from app.enums import AgentPhase
from tests.conftest import run_async
from tests.planner_helpers import (
    advance_session_to_planning,
    mock_planner_llm_client,
    plan_body,
    success_response,
)
from tests.test_supervisor import (
    _create_session,
    _seed_data_source,
    _seed_workspace,
)

FAKE_API_KEY = "sk-fake-planner-key-not-real"


def _analytical_intent() -> AIIntent:
    return AIIntent(
        intent=AIIntentType.ANALYTICAL_QUERY,
        operation=AIOperationType.AGGREGATE,
        subject="orders",
        confidence=AIConfidence.HIGH,
        requires_data_access=True,
        requires_metadata=True,
    )


def test_plan_custom_entries_round_trip() -> None:
    llm_result = LLMPlannerOutput.model_validate(plan_body())
    plan = normalize_plan(llm_result=llm_result, intent=_analytical_intent())
    entries = plan_custom_entries(plan)
    restored_required = parse_required_data(entries["plan_required_data"])
    restored_steps = parse_action_steps(entries["plan_action_steps"])
    assert restored_required[0].name == "orders"
    assert restored_steps[0].action is PlannerActionKind.RESOLVE_METADATA


def test_apply_planner_intent_preserves_classified_intent() -> None:
    llm_result = LLMPlannerOutput.model_validate(plan_body(intent="AGGREGATION"))
    plan = normalize_plan(llm_result=llm_result, intent=_analytical_intent())
    resolved = apply_planner_intent(_analytical_intent(), plan)
    assert resolved.intent is AIIntentType.ANALYTICAL_QUERY
    assert plan.detected_intent is AIIntentType.ANALYTICAL_QUERY


def test_normalize_plan_handles_llm_unsupported_with_reason() -> None:
    llm_result = LLMPlannerOutput.model_validate(
        plan_body(unsupported=True, unsupported_reason="write_operation")
    )
    plan = normalize_plan(llm_result=llm_result, intent=_analytical_intent())
    assert plan.request_plan.unsupported is True
    assert plan.detected_intent is AIIntentType.UNSUPPORTED


def test_normalize_plan_handles_llm_ambiguous_clarification() -> None:
    llm_result = LLMPlannerOutput.model_validate(
        plan_body(
            requires_clarification=True,
            clarification_question="Which metric should be used?",
            missing_information=["metric"],
        )
    )
    plan = normalize_plan(llm_result=llm_result, intent=_analytical_intent())
    assert plan.request_plan.requires_clarification is True
    assert plan.request_plan.clarification_question == "Which metric should be used?"


# --- Parsing and normalization ---


def test_parse_llm_plan_accepts_valid_payload() -> None:
    result = parse_llm_plan(json.dumps(plan_body()))
    assert result.intent is AIIntentType.ANALYTICAL_QUERY
    assert len(result.action_steps) == 2


def test_parse_llm_plan_rejects_invalid_json() -> None:
    with pytest.raises(LLMResponseValidationError, match="valid JSON"):
        parse_llm_plan("not-json")


def test_parse_llm_plan_rejects_malformed_schema() -> None:
    with pytest.raises(LLMResponseValidationError, match="schema validation"):
        parse_llm_plan(json.dumps({"intent": "NOT_A_REAL_INTENT"}))


def test_normalize_plan_builds_analysis_plan_with_required_data() -> None:
    llm_result = LLMPlannerOutput.model_validate(plan_body())
    plan = normalize_plan(llm_result=llm_result, intent=_analytical_intent())
    assert plan.detected_intent is AIIntentType.ANALYTICAL_QUERY
    assert plan.required_data[0].name == "orders"
    assert plan.required_data[0].kind is RequiredDataKind.TABLE
    assert plan.action_steps[0].action is PlannerActionKind.RESOLVE_METADATA
    assert plan.action_steps[1].tool is PlannerToolRef.POSTGRES_QUERY
    assert AIPlanOperation.SELECT in plan.request_plan.operations


def test_normalize_plan_maps_missing_information_to_clarification() -> None:
    llm_result = LLMPlannerOutput.model_validate(
        plan_body(
            missing_information=["Which time period should be used?"],
            requires_clarification=False,
        )
    )
    plan = normalize_plan(llm_result=llm_result, intent=_analytical_intent())
    assert plan.request_plan.requires_clarification is True
    assert plan.missing_information == ["Which time period should be used?"]


def test_normalize_plan_requires_reason_for_unsupported() -> None:
    llm_result = LLMPlannerOutput.model_validate(
        plan_body(unsupported=True, unsupported_reason=None)
    )
    with pytest.raises(PlannerValidationError, match="reason"):
        normalize_plan(llm_result=llm_result, intent=_analytical_intent())


def test_normalize_plan_requires_clarification_question() -> None:
    llm_result = LLMPlannerOutput.model_validate(
        plan_body(
            requires_clarification=True,
            clarification_question=None,
            missing_information=[],
        )
    )
    with pytest.raises(PlannerValidationError, match="question"):
        normalize_plan(
            llm_result=llm_result,
            intent=AIIntent(
                intent=AIIntentType.UNKNOWN,
                confidence=AIConfidence.LOW,
                requires_clarification=True,
            ),
        )


def test_plan_from_intent_handles_unsupported_without_llm() -> None:
    intent = AIIntent(
        intent=AIIntentType.UNSUPPORTED,
        confidence=AIConfidence.HIGH,
        unsupported_reason="write_operation",
    )
    plan = plan_from_intent(intent)
    assert plan.request_plan.unsupported is True


def test_plan_from_intent_handles_clarification_without_llm() -> None:
    intent = AIIntent(
        intent=AIIntentType.UNKNOWN,
        confidence=AIConfidence.MEDIUM,
        requires_clarification=True,
        clarification_question="Which metric?",
    )
    plan = plan_from_intent(intent)
    assert plan.request_plan.requires_clarification is True


# --- Intent detection via planning output ---


def test_create_analysis_plan_detects_intent_from_llm() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=success_response(plan_body()))

    plan = run_async(
        create_analysis_plan(
            client=mock_planner_llm_client(handler),
            registry=build_planner_prompt_registry(),
            intent=_analytical_intent(),
            message="Show revenue by region",
            has_data_source=True,
            data_source_name="Analytics DB",
        )
    )
    assert plan.detected_intent is AIIntentType.ANALYTICAL_QUERY
    assert AIPlanCapability.METADATA in plan.request_plan.required_capabilities


# --- Tool/action identification ---


def test_normalize_plan_identifies_action_sequence() -> None:
    llm_result = LLMPlannerOutput.model_validate(plan_body())
    plan = normalize_plan(llm_result=llm_result, intent=_analytical_intent())
    assert [step.step_order for step in plan.action_steps] == [1, 2]
    assert plan.action_steps[0].tool is PlannerToolRef.METADATA_LOOKUP


# --- Logging ---


def test_plan_log_context_omits_message_content() -> None:
    llm_result = LLMPlannerOutput.model_validate(plan_body())
    plan = normalize_plan(llm_result=llm_result, intent=_analytical_intent())
    context = plan_log_context(plan)
    assert "orders" not in str(context)
    assert context["required_data_count"] == 2


def test_map_planner_error_maps_timeout_and_validation() -> None:
    from app.ai.exceptions import AIProviderTimeoutError, AIResponseValidationError
    from app.ai.llm.errors import LLMResponseValidationError, LLMTimeoutError

    try:
        raise LLMTimeoutError("timed out")
    except LLMTimeoutError as exc:
        planner_error = PlannerLLMError("timed out")
        planner_error.__cause__ = exc
        timeout_mapped = map_planner_error(planner_error)
    assert isinstance(timeout_mapped, AIProviderTimeoutError)

    try:
        raise LLMResponseValidationError("invalid schema")
    except LLMResponseValidationError as exc:
        planner_error = PlannerLLMError("invalid schema")
        planner_error.__cause__ = exc
        validation_mapped = map_planner_error(planner_error)
    assert isinstance(validation_mapped, AIResponseValidationError)


def test_redacting_filter_masks_secrets_in_planner_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "sk-test-planner-secret"
    logger = logging.getLogger("tests.planner.logging")
    caplog.set_level(logging.INFO)
    logger.info("planner event", extra={"token": redact_secret(secret)})
    record = caplog.records[-1]
    rendered = RedactingFilter().filter(record)
    assert rendered is True
    assert secret not in record.getMessage()


# --- Service ---


def test_planner_creates_plan_in_planning_phase(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    service, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
    )
    snapshot = advance_session_to_planning(
        service,
        snapshot,
        workspace_id=workspace.id,
        user_id=user.id,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=success_response(plan_body()))

    planner = PlannerAgent(db_session, llm_client=mock_planner_llm_client(handler))
    result = run_async(
        planner.create_plan(
            PlanCreateParams(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                expected_agent_version=snapshot.agent_state.version,
                intent=_analytical_intent(),
                message="Show revenue by region",
                data_source_name="Analytics DB",
            )
        )
    )

    assert isinstance(result.plan, AnalysisPlan)
    assert result.session.agent_state is not None
    assert result.session.agent_state.phase is AgentPhase.PLANNING
    assert result.session.agent_state.payload.plan_version == "v1"
    assert "TABLE:orders" in result.session.agent_state.payload.metadata_refs
    assert result.session.agent_state.payload.custom["planner_action"] == "PLAN_CREATED"
    assert "plan_required_data" in result.session.agent_state.payload.custom
    assert "plan_action_steps" in result.session.agent_state.payload.custom


def test_planner_service_handles_llm_unsupported_response(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    service, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
    )
    snapshot = advance_session_to_planning(
        service,
        snapshot,
        workspace_id=workspace.id,
        user_id=user.id,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=success_response(
                plan_body(unsupported=True, unsupported_reason="write_operation")
            ),
        )

    planner = PlannerAgent(db_session, llm_client=mock_planner_llm_client(handler))
    result = run_async(
        planner.create_plan(
            PlanCreateParams(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                expected_agent_version=snapshot.agent_state.version,
                intent=_analytical_intent(),
                message="Delete all records",
            )
        )
    )
    assert result.plan.request_plan.unsupported is True
    assert result.session.agent_state.payload.custom.get("unsupported") == "true"


def test_planner_service_handles_llm_ambiguous_response(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    service, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
    )
    snapshot = advance_session_to_planning(
        service,
        snapshot,
        workspace_id=workspace.id,
        user_id=user.id,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=success_response(
                plan_body(
                    requires_clarification=True,
                    clarification_question="Which region?",
                    missing_information=["region"],
                )
            ),
        )

    planner = PlannerAgent(db_session, llm_client=mock_planner_llm_client(handler))
    result = run_async(
        planner.create_plan(
            PlanCreateParams(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                expected_agent_version=snapshot.agent_state.version,
                intent=_analytical_intent(),
                message="Show revenue",
            )
        )
    )
    assert result.plan.request_plan.requires_clarification is True
    assert (
        result.session.agent_state.payload.custom.get("requires_clarification")
        == "true"
    )


def test_planner_skips_llm_for_unsupported_intent(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    service, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
    )
    snapshot = advance_session_to_planning(
        service,
        snapshot,
        workspace_id=workspace.id,
        user_id=user.id,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("LLM should not be called for unsupported intents")

    planner = PlannerAgent(db_session, llm_client=mock_planner_llm_client(handler))
    result = run_async(
        planner.create_plan(
            PlanCreateParams(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                expected_agent_version=snapshot.agent_state.version,
                intent=AIIntent(
                    intent=AIIntentType.UNSUPPORTED,
                    confidence=AIConfidence.HIGH,
                    unsupported_reason="write_operation",
                ),
                message="Delete all records",
            )
        )
    )
    assert result.plan.request_plan.unsupported is True


def test_planner_rejects_non_planning_phase(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    service, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
    )
    snapshot = service.update_agent_state(
        snapshot.session_id,
        SetPhaseTransition(phase=AgentPhase.INTENT),
        workspace_id=workspace.id,
        user_id=user.id,
        expected_version=snapshot.agent_state.version,
    )

    planner = PlannerAgent(
        db_session,
        llm_client=mock_planner_llm_client(
            lambda _: httpx.Response(200, json=success_response(plan_body()))
        ),
    )
    with pytest.raises(PlannerAuthorizationError, match="PLANNING"):
        run_async(
            planner.create_plan(
                PlanCreateParams(
                    session_id=snapshot.session_id,
                    workspace_id=workspace.id,
                    user_id=user.id,
                    expected_agent_version=snapshot.agent_state.version,
                    intent=_analytical_intent(),
                    message="Show revenue",
                )
            )
        )


def test_planner_llm_failure_surfaces_as_planner_error(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    service, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
    )
    snapshot = advance_session_to_planning(
        service,
        snapshot,
        workspace_id=workspace.id,
        user_id=user.id,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "provider down"}})

    planner = PlannerAgent(db_session, llm_client=mock_planner_llm_client(handler))
    with pytest.raises(PlannerLLMError):
        run_async(
            planner.create_plan(
                PlanCreateParams(
                    session_id=snapshot.session_id,
                    workspace_id=workspace.id,
                    user_id=user.id,
                    expected_agent_version=snapshot.agent_state.version,
                    intent=_analytical_intent(),
                    message="Show revenue",
                )
            )
        )


def test_planner_malformed_llm_output_raises_validation_error(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    service, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
    )
    snapshot = advance_session_to_planning(
        service,
        snapshot,
        workspace_id=workspace.id,
        user_id=user.id,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=success_response({"unexpected": "payload"}),
        )

    planner = PlannerAgent(db_session, llm_client=mock_planner_llm_client(handler))
    with pytest.raises(PlannerLLMError):
        run_async(
            planner.create_plan(
                PlanCreateParams(
                    session_id=snapshot.session_id,
                    workspace_id=workspace.id,
                    user_id=user.id,
                    expected_agent_version=snapshot.agent_state.version,
                    intent=_analytical_intent(),
                    message="Show revenue",
                )
            )
        )


def test_planner_llm_timeout_surfaces_as_planner_error(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    service, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
    )
    snapshot = advance_session_to_planning(
        service,
        snapshot,
        workspace_id=workspace.id,
        user_id=user.id,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    planner = PlannerAgent(db_session, llm_client=mock_planner_llm_client(handler))
    with pytest.raises(PlannerLLMError):
        run_async(
            planner.create_plan(
                PlanCreateParams(
                    session_id=snapshot.session_id,
                    workspace_id=workspace.id,
                    user_id=user.id,
                    expected_agent_version=snapshot.agent_state.version,
                    intent=_analytical_intent(),
                    message="Show revenue",
                )
            )
        )


def test_planner_cross_workspace_access_is_denied(db_session: Session) -> None:
    user_a, workspace_a, organization_a = _seed_workspace(db_session)
    user_b, workspace_b, _ = _seed_workspace(db_session)
    _, snapshot = _create_session(
        db_session,
        user=user_a,
        workspace=workspace_a,
        organization=organization_a,
    )

    planner = PlannerAgent(
        db_session,
        llm_client=mock_planner_llm_client(
            lambda _: httpx.Response(200, json=success_response(plan_body()))
        ),
    )
    with pytest.raises(AnalysisSessionNotFoundError):
        run_async(
            planner.create_plan(
                PlanCreateParams(
                    session_id=snapshot.session_id,
                    workspace_id=workspace_b.id,
                    user_id=user_b.id,
                    expected_agent_version=1,
                    intent=_analytical_intent(),
                    message="Show revenue",
                )
            )
        )


def test_planner_rejects_foreign_data_source(db_session: Session) -> None:
    user_a, workspace_a, organization_a = _seed_workspace(db_session)
    user_b, workspace_b, _ = _seed_workspace(db_session)
    foreign_source = _seed_data_source(db_session, workspace=workspace_b, user=user_b)
    _, snapshot = _create_session(
        db_session,
        user=user_a,
        workspace=workspace_a,
        organization=organization_a,
    )
    from app.db.models import AnalysisSession

    row = db_session.get(AnalysisSession, snapshot.session_id)
    assert row is not None
    row.data_source_id = foreign_source.id
    db_session.flush()

    planner = PlannerAgent(
        db_session,
        llm_client=mock_planner_llm_client(
            lambda _: httpx.Response(200, json=success_response(plan_body()))
        ),
    )
    with pytest.raises(PlannerAuthorizationError, match="not accessible"):
        run_async(
            planner.create_plan(
                PlanCreateParams(
                    session_id=snapshot.session_id,
                    workspace_id=workspace_a.id,
                    user_id=user_a.id,
                    expected_agent_version=snapshot.agent_state.version,
                    intent=_analytical_intent(),
                    message="Show revenue",
                )
            )
        )
