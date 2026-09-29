from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import httpx
import pytest
from sqlalchemy.orm import Session

from app.ai.llm import AsyncLLMClient, LLMClientConfig, LLMResponseValidationError
from app.ai.state import (
    AgentStateService,
    AnalysisSessionNotFoundError,
    CreateSessionParams,
    SetPhaseTransition,
)
from app.ai.supervisor import (
    InvalidRoutingError,
    RequestCategory,
    RequestClassification,
    SuperviseAdvanceParams,
    SuperviseMessageParams,
    SupervisorAction,
    SupervisorAgent,
    SupervisorAuthorizationError,
    SupervisorClassificationError,
    SupervisorLLMError,
    build_decision,
    classification_from_policy,
    classification_log_context,
    decision_log_context,
    ensure_routing_allowed,
    normalize_classification,
    parse_llm_classification,
    resolve_advance_routing,
    resolve_routing,
)
from app.ai.supervisor.models import (
    ClassificationConfidence,
    LLMRequestClassification,
)
from app.core.logging import RedactingFilter, redact_secret
from app.db.models import (
    DataSource,
    DataSourceType,
    Organization,
    User,
    UserRole,
    Workspace,
)
from app.enums import AgentPhase, AnalysisSessionStatus
from tests.conftest import run_async

FAKE_API_KEY = "sk-fake-supervisor-key-not-real"


def _client_config(**overrides: object) -> LLMClientConfig:
    payload: dict[str, object] = {
        "api_key": FAKE_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4o-mini",
        "timeout": 5.0,
        "temperature": 0.2,
        "max_tokens": 256,
        "max_retries": 0,
        "retry_base_backoff": 0.1,
        "retry_max_backoff": 1.0,
    }
    payload.update(overrides)
    return LLMClientConfig(**payload)  # type: ignore[arg-type]


def _classification_body(**overrides: object) -> dict[str, Any]:
    body: dict[str, Any] = {
        "category": "ANALYTICAL_QUERY",
        "confidence": "HIGH",
        "requires_data_access": True,
        "requires_clarification": False,
    }
    body.update(overrides)
    return body


def _success_response(content: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "gen-supervisor-1",
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


def _mock_client(handler: Any) -> AsyncLLMClient:
    transport = httpx.MockTransport(handler)
    return AsyncLLMClient(_client_config(), transport=transport)


def _user() -> User:
    return User(
        first_name="Alan",
        last_name="Turing",
        email=f"alan-{uuid.uuid4().hex[:8]}@example.com",
        password_hash="hashed-password",
        role=UserRole.USER,
        is_verified=True,
        is_active=True,
    )


def _seed_workspace(db_session: Session) -> tuple[User, Workspace, Organization]:
    user = _user()
    organization = Organization(
        name="Supervisor Org",
        slug=f"supervisor-{uuid.uuid4().hex[:8]}",
    )
    db_session.add_all([user, organization])
    db_session.flush()
    workspace = Workspace(
        organization_id=organization.id,
        name="Supervisor Workspace",
        slug=f"supervisor-ws-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(workspace)
    db_session.flush()
    return user, workspace, organization


def _seed_data_source(
    db_session: Session, *, workspace: Workspace, user: User
) -> DataSource:
    data_source = DataSource(
        workspace_id=workspace.id,
        name="Analytics DB",
        type=DataSourceType.POSTGRESQL,
        created_by=user.id,
    )
    db_session.add(data_source)
    db_session.flush()
    return data_source


def _create_session(
    db_session: Session,
    *,
    user: User,
    workspace: Workspace,
    organization: Organization,
    data_source: DataSource | None = None,
) -> tuple[AgentStateService, Any]:
    service = AgentStateService(db_session)
    snapshot = service.create_session(
        CreateSessionParams(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
            data_source_id=data_source.id if data_source else None,
        )
    )
    return service, snapshot


# --- Classification parsing ---


def test_parse_llm_classification_accepts_valid_payload() -> None:
    result = parse_llm_classification(json.dumps(_classification_body()))
    assert result.category is RequestCategory.ANALYTICAL_QUERY


def test_parse_llm_classification_rejects_invalid_json() -> None:
    with pytest.raises(LLMResponseValidationError, match="valid JSON"):
        parse_llm_classification("not-json")


def test_parse_llm_classification_rejects_malformed_schema() -> None:
    with pytest.raises(LLMResponseValidationError, match="schema validation"):
        parse_llm_classification(json.dumps({"category": "NOT_A_REAL_CATEGORY"}))


def test_normalize_classification_maps_low_confidence_unknown_to_ambiguous() -> None:
    llm_result = LLMRequestClassification(
        category=RequestCategory.UNKNOWN,
        confidence=ClassificationConfidence.LOW,
    )
    normalized = normalize_classification(llm_result)
    assert normalized.category is RequestCategory.AMBIGUOUS


def test_normalize_classification_requires_reason_for_unsupported() -> None:
    llm_result = LLMRequestClassification(
        category=RequestCategory.UNSUPPORTED,
        confidence=ClassificationConfidence.HIGH,
    )
    with pytest.raises(SupervisorClassificationError, match="reason"):
        normalize_classification(llm_result)


def test_classification_from_policy_marks_unsupported() -> None:
    classification = classification_from_policy("write_operation")
    assert classification.category is RequestCategory.UNSUPPORTED
    assert classification.source == "policy"


# --- Routing ---


def test_resolve_routing_analytical_query_routes_to_intent_agent() -> None:
    classification = RequestClassification(
        category=RequestCategory.ANALYTICAL_QUERY,
        confidence=ClassificationConfidence.HIGH,
        requires_data_access=True,
    )
    action, next_phase, message = resolve_routing(
        phase=AgentPhase.INITIAL,
        classification=classification,
        has_data_source=True,
    )
    assert action is SupervisorAction.RUN_INTENT_AGENT
    assert next_phase is AgentPhase.INTENT
    assert message is None


def test_resolve_routing_blocks_data_access_without_data_source() -> None:
    classification = RequestClassification(
        category=RequestCategory.ANALYTICAL_QUERY,
        confidence=ClassificationConfidence.HIGH,
        requires_data_access=True,
    )
    action, next_phase, message = resolve_routing(
        phase=AgentPhase.INITIAL,
        classification=classification,
        has_data_source=False,
    )
    assert action is SupervisorAction.RESPOND_UNSUPPORTED
    assert next_phase is None
    assert "data source" in (message or "").lower()


def test_resolve_routing_unknown_category_requests_clarification() -> None:
    classification = RequestClassification(
        category=RequestCategory.UNKNOWN,
        confidence=ClassificationConfidence.HIGH,
        requires_clarification=False,
    )
    action, next_phase, _ = resolve_routing(
        phase=AgentPhase.INITIAL,
        classification=classification,
        has_data_source=True,
    )
    assert action is SupervisorAction.REQUEST_CLARIFICATION
    assert next_phase is AgentPhase.AWAITING_CLARIFICATION


def test_resolve_routing_invalid_terminal_phase_raises() -> None:
    classification = RequestClassification(
        category=RequestCategory.GENERAL,
        confidence=ClassificationConfidence.HIGH,
    )
    with pytest.raises(InvalidRoutingError, match="terminal"):
        resolve_routing(
            phase=AgentPhase.COMPLETED,
            classification=classification,
            has_data_source=True,
        )


def test_resolve_advance_routing_from_intent_to_planner() -> None:
    action, next_phase, message = resolve_advance_routing(AgentPhase.INTENT)
    assert action is SupervisorAction.RUN_PLANNER
    assert next_phase is AgentPhase.PLANNING
    assert message is None


def test_ensure_routing_allowed_rejects_invalid_phase_transition() -> None:
    with pytest.raises(InvalidRoutingError, match="Cannot route"):
        ensure_routing_allowed(
            action=SupervisorAction.RUN_INTENT_AGENT,
            current_phase=AgentPhase.INITIAL,
            next_phase=AgentPhase.PLANNING,
        )


def test_build_decision_rejects_non_initial_phase() -> None:
    classification = RequestClassification(
        category=RequestCategory.ANALYTICAL_QUERY,
        confidence=ClassificationConfidence.HIGH,
    )
    with pytest.raises(InvalidRoutingError, match="INITIAL phase"):
        build_decision(
            classification=classification,
            current_phase=AgentPhase.INTENT,
            has_data_source=True,
        )


def test_custom_payload_merge_preserves_existing_keys() -> None:
    from app.ai.state.models import AgentStatePayload, AgentStateSnapshot
    from app.ai.state.serialization import deserialize_payload

    current = AgentStateSnapshot(
        phase=AgentPhase.INITIAL,
        payload=AgentStatePayload(custom={"existing_key": "keep-me"}),
        version=1,
    )
    payload_custom = dict(current.payload.custom)
    payload_custom["supervisor_action"] = SupervisorAction.RUN_INTENT_AGENT.value
    payload_custom["request_category"] = RequestCategory.ANALYTICAL_QUERY.value
    merged = current.payload.model_dump()
    merged.update({"custom": payload_custom})
    result = deserialize_payload(merged)
    assert result.custom["existing_key"] == "keep-me"
    assert result.custom["supervisor_action"] == "RUN_INTENT_AGENT"


def test_map_supervisor_error_maps_timeout_and_validation() -> None:
    from app.ai.exceptions import AIProviderTimeoutError, AIResponseValidationError
    from app.ai.llm.errors import LLMResponseValidationError, LLMTimeoutError
    from app.ai.supervisor.errors import SupervisorLLMError
    from app.ai.supervisor.errors_mapping import map_supervisor_error

    try:
        raise LLMTimeoutError("timed out")
    except LLMTimeoutError as exc:
        supervisor_error = SupervisorLLMError("timed out")
        supervisor_error.__cause__ = exc
        timeout_mapped = map_supervisor_error(supervisor_error)
    assert isinstance(timeout_mapped, AIProviderTimeoutError)

    try:
        raise LLMResponseValidationError("invalid schema")
    except LLMResponseValidationError as exc:
        supervisor_error = SupervisorLLMError("invalid schema")
        supervisor_error.__cause__ = exc
        validation_mapped = map_supervisor_error(supervisor_error)
    assert isinstance(validation_mapped, AIResponseValidationError)

    from app.ai.supervisor.errors import SupervisorClassificationError

    try:
        raise SupervisorClassificationError("unsupported without reason")
    except SupervisorClassificationError as exc:
        supervisor_error = SupervisorLLMError("classification failed")
        supervisor_error.__cause__ = exc
        classification_mapped = map_supervisor_error(supervisor_error)
    assert isinstance(classification_mapped, AIResponseValidationError)


# --- Logging ---


def test_classification_log_context_omits_message_content() -> None:
    context = classification_log_context(
        RequestClassification(
            category=RequestCategory.ANALYTICAL_QUERY,
            confidence=ClassificationConfidence.HIGH,
            reason="secret user text",
        )
    )
    assert "secret user text" not in str(context)
    assert context["has_reason"] is True


def test_decision_log_context_is_safe_for_logging() -> None:
    decision = build_decision(
        classification=RequestClassification(
            category=RequestCategory.AMBIGUOUS,
            confidence=ClassificationConfidence.MEDIUM,
            requires_clarification=True,
            clarification_question="What metric do you need?",
        ),
        current_phase=AgentPhase.INITIAL,
        has_data_source=True,
    )
    context = decision_log_context(decision)
    assert "What metric do you need?" not in str(context)
    assert context["supervisor_action"] == SupervisorAction.REQUEST_CLARIFICATION.value


def test_redacting_filter_masks_secrets_in_supervisor_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "sk-test-supervisor-secret"
    logger = logging.getLogger("tests.supervisor.logging")
    caplog.set_level(logging.INFO)
    logger.info("supervisor event", extra={"token": redact_secret(secret)})
    record = caplog.records[-1]
    rendered = RedactingFilter().filter(record)
    assert rendered is True
    assert secret not in record.getMessage()


# --- Service: classification and routing ---


def test_supervisor_routes_analytical_query_to_intent_agent(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_response(_classification_body()))

    supervisor = SupervisorAgent(db_session, llm_client=_mock_client(handler))
    result = run_async(
        supervisor.handle_message(
            SuperviseMessageParams(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="How many orders last month?",
                expected_agent_version=snapshot.agent_state.version,
                expected_context_version=snapshot.conversation_version or 1,
            )
        )
    )

    assert result.decision.action is SupervisorAction.RUN_INTENT_AGENT
    assert result.decision.next_phase is AgentPhase.INTENT
    assert result.session.agent_state is not None
    assert result.session.agent_state.phase is AgentPhase.INTENT
    assert result.session.conversation is not None
    assert result.session.conversation.message_count == 1


def test_supervisor_rejects_unsupported_write_request_without_llm(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("LLM should not be called for policy-blocked requests")

    supervisor = SupervisorAgent(db_session, llm_client=_mock_client(handler))
    result = run_async(
        supervisor.handle_message(
            SuperviseMessageParams(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="Delete all customer records",
                expected_agent_version=snapshot.agent_state.version,
                expected_context_version=snapshot.conversation_version or 1,
            )
        )
    )

    assert result.decision.action is SupervisorAction.RESPOND_UNSUPPORTED
    assert result.decision.classification.source == "policy"
    assert result.session.status == AnalysisSessionStatus.COMPLETED.value


def test_supervisor_requests_clarification_for_ambiguous_message(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_response(
                {
                    "category": "AMBIGUOUS",
                    "confidence": "MEDIUM",
                    "requires_clarification": True,
                    "clarification_question": "Which table should I inspect?",
                }
            ),
        )

    supervisor = SupervisorAgent(db_session, llm_client=_mock_client(handler))
    result = run_async(
        supervisor.handle_message(
            SuperviseMessageParams(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="Show me something",
                expected_agent_version=snapshot.agent_state.version,
                expected_context_version=snapshot.conversation_version or 1,
            )
        )
    )

    assert result.decision.action is SupervisorAction.REQUEST_CLARIFICATION
    assert result.session.agent_state is not None
    assert result.session.agent_state.phase is AgentPhase.AWAITING_CLARIFICATION
    assert result.session.status == AnalysisSessionStatus.ACTIVE.value


def test_supervisor_llm_failure_surfaces_as_supervisor_error(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "provider down"}})

    supervisor = SupervisorAgent(db_session, llm_client=_mock_client(handler))
    with pytest.raises(SupervisorLLMError):
        run_async(
            supervisor.handle_message(
                SuperviseMessageParams(
                    session_id=snapshot.session_id,
                    workspace_id=workspace.id,
                    user_id=user.id,
                    message="How many orders?",
                    expected_agent_version=snapshot.agent_state.version,
                    expected_context_version=snapshot.conversation_version or 1,
                )
            )
        )


def test_supervisor_malformed_llm_output_raises_validation_error(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_response({"unexpected": "payload"}),
        )

    supervisor = SupervisorAgent(db_session, llm_client=_mock_client(handler))
    with pytest.raises(SupervisorLLMError):
        run_async(
            supervisor.handle_message(
                SuperviseMessageParams(
                    session_id=snapshot.session_id,
                    workspace_id=workspace.id,
                    user_id=user.id,
                    message="How many orders?",
                    expected_agent_version=snapshot.agent_state.version,
                    expected_context_version=snapshot.conversation_version or 1,
                )
            )
        )


def test_supervisor_advance_workflow_transitions_intent_to_planning(
    db_session: Session,
) -> None:
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

    supervisor = SupervisorAgent(
        db_session,
        llm_client=_mock_client(lambda _: httpx.Response(500)),
    )
    result = run_async(
        supervisor.advance_workflow(
            SuperviseAdvanceParams(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                expected_agent_version=snapshot.agent_state.version,
            )
        )
    )

    assert result.decision.action is SupervisorAction.RUN_PLANNER
    assert result.session.agent_state is not None
    assert result.session.agent_state.phase is AgentPhase.PLANNING


def test_supervisor_rejects_message_when_not_initial_phase(
    db_session: Session,
) -> None:
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

    supervisor = SupervisorAgent(
        db_session,
        llm_client=_mock_client(
            lambda _: httpx.Response(
                200, json=_success_response(_classification_body())
            )
        ),
    )
    with pytest.raises(InvalidRoutingError, match="cannot accept new user messages"):
        run_async(
            supervisor.handle_message(
                SuperviseMessageParams(
                    session_id=snapshot.session_id,
                    workspace_id=workspace.id,
                    user_id=user.id,
                    message="Another message",
                    expected_agent_version=snapshot.agent_state.version,
                    expected_context_version=snapshot.conversation_version or 1,
                )
            )
        )


def test_supervisor_llm_timeout_surfaces_as_supervisor_error(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    supervisor = SupervisorAgent(db_session, llm_client=_mock_client(handler))
    with pytest.raises(SupervisorLLMError):
        run_async(
            supervisor.handle_message(
                SuperviseMessageParams(
                    session_id=snapshot.session_id,
                    workspace_id=workspace.id,
                    user_id=user.id,
                    message="How many orders?",
                    expected_agent_version=snapshot.agent_state.version,
                    expected_context_version=snapshot.conversation_version or 1,
                )
            )
        )


def test_supervisor_cross_workspace_access_is_denied(db_session: Session) -> None:
    user_a, workspace_a, organization_a = _seed_workspace(db_session)
    user_b, workspace_b, _organization_b = _seed_workspace(db_session)
    _, snapshot = _create_session(
        db_session,
        user=user_a,
        workspace=workspace_a,
        organization=organization_a,
    )

    supervisor = SupervisorAgent(
        db_session,
        llm_client=_mock_client(
            lambda _: httpx.Response(200, json=_success_response({}))
        ),
    )
    with pytest.raises(AnalysisSessionNotFoundError):
        run_async(
            supervisor.handle_message(
                SuperviseMessageParams(
                    session_id=snapshot.session_id,
                    workspace_id=workspace_b.id,
                    user_id=user_b.id,
                    message="How many orders?",
                    expected_agent_version=1,
                    expected_context_version=1,
                )
            )
        )


def test_supervisor_rejects_foreign_data_source(db_session: Session) -> None:
    user_a, workspace_a, organization_a = _seed_workspace(db_session)
    user_b, workspace_b, _ = _seed_workspace(db_session)
    foreign_source = _seed_data_source(db_session, workspace=workspace_b, user=user_b)
    _service, snapshot = _create_session(
        db_session,
        user=user_a,
        workspace=workspace_a,
        organization=organization_a,
    )
    # Simulate a poisoned session reference by mutating ORM row directly.
    from app.db.models import AnalysisSession

    row = db_session.get(AnalysisSession, snapshot.session_id)
    assert row is not None
    row.data_source_id = foreign_source.id
    db_session.flush()

    supervisor = SupervisorAgent(
        db_session,
        llm_client=_mock_client(
            lambda _: httpx.Response(200, json=_success_response({}))
        ),
    )
    with pytest.raises(SupervisorAuthorizationError, match="not accessible"):
        run_async(
            supervisor.handle_message(
                SuperviseMessageParams(
                    session_id=snapshot.session_id,
                    workspace_id=workspace_a.id,
                    user_id=user_a.id,
                    message="How many orders?",
                    expected_agent_version=snapshot.agent_state.version,
                    expected_context_version=snapshot.conversation_version or 1,
                )
            )
        )
