from __future__ import annotations

import logging
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.ai.exceptions import (
    AIProviderAuthenticationError,
    AIProviderError,
    AIProviderTimeoutError,
)
from app.ai.intent_types import (
    AggregationType,
    AIConfidence,
    AIIntentType,
    AIMetric,
    AIOperationType,
    LLMIntentDetection,
)
from app.api.routes.ai import require_llm_provider
from app.core.logging import RedactingFilter
from app.core.security import create_access_token, hash_password
from app.db.models import (
    DataSource,
    DataSourceConnection,
    Organization,
    User,
    UserRole,
    Workspace,
    WorkspaceMember,
    WorkspaceRole,
)
from app.enums import DataSourceType
from app.main import app
from app.services.credentials import encrypt_secret
from tests.ai_fakes import FakeLLMProvider
from tests.test_ai_metadata import _seed_sales_catalog

pytest_plugins = ["tests.planner_helpers"]

PREFIX = "/api/v1/ai"
VALID_PASSWORD = "SecurePassword123!"
CUSTOMER_PASSWORD = "CustomerDbPassword!@# 42"
LLM_SECRET = "sk-test-secret-llm-key-do-not-log"


def _auth_header(user: User) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user.id)}"}


def _create_user(db_session: Session, *, role: UserRole = UserRole.USER) -> User:
    user = User(
        first_name="Test",
        last_name="User",
        email=f"user-{uuid.uuid4().hex[:8]}@example.com",
        password_hash=hash_password(VALID_PASSWORD),
        role=role,
        is_verified=True,
        is_active=True,
    )
    db_session.add(user)
    db_session.flush()
    return user


def _owned_source(db_session: Session, owner: User, *, name: str = "Analytics") -> DataSource:
    """Org + workspace (owner membership) + connected data source, written directly via ORM."""
    suffix = uuid.uuid4().hex[:8]
    org = Organization(name=f"Org {name}", slug=f"org-{suffix}")
    db_session.add(org)
    db_session.flush()
    workspace = Workspace(organization_id=org.id, name=name, slug=f"ws-{suffix}")
    db_session.add(workspace)
    db_session.flush()
    db_session.add(
        WorkspaceMember(workspace_id=workspace.id, user_id=owner.id, role=WorkspaceRole.OWNER)
    )
    source = DataSource(
        workspace_id=workspace.id,
        name=name,
        type=DataSourceType.POSTGRESQL,
        created_by=owner.id,
    )
    db_session.add(source)
    db_session.flush()
    db_session.add(
        DataSourceConnection(
            data_source_id=source.id,
            host="db.example.com",
            port=5432,
            database_name="analytics",
            username="analytics_user",
            encrypted_password=encrypt_secret(CUSTOMER_PASSWORD),
            ssl_mode="require",
        )
    )
    db_session.flush()
    return source


def _chat(client: TestClient, user: User, source_id: object, message: str = "Hello"):
    return client.post(
        f"{PREFIX}/chat",
        headers=_auth_header(user),
        json={"message": message, "data_source_id": str(source_id)},
    )


@pytest.fixture
def fake_provider() -> FakeLLMProvider:
    return FakeLLMProvider()


@pytest.fixture
def source(db_session: Session, test_user: User) -> DataSource:
    return _owned_source(db_session, test_user)


@pytest.fixture(autouse=True)
def mock_planner_for_supervised_chat(mock_planner_llm: None) -> None:
    """Prevent supervised chat API tests from calling a live planner LLM."""


@pytest.fixture(autouse=True)
def mock_supervisor_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.ai.supervisor.models import (
        ClassificationConfidence,
        RequestCategory,
        RequestClassification,
    )

    async def _classify(**_kwargs: object) -> RequestClassification:
        return RequestClassification(
            category=RequestCategory.ANALYTICAL_QUERY,
            confidence=ClassificationConfidence.HIGH,
            requires_data_access=True,
        )

    monkeypatch.setattr(
        "app.ai.supervisor.service.classify_request",
        _classify,
    )


@pytest.fixture
def ai_client(
    client: TestClient,
    fake_provider: FakeLLMProvider,
) -> TestClient:
    app.dependency_overrides[require_llm_provider] = lambda: fake_provider
    return client


def test_chat_requires_authentication(ai_client: TestClient) -> None:
    response = ai_client.post(
        f"{PREFIX}/chat",
        json={"message": "Hello", "data_source_id": str(uuid.uuid4())},
    )
    assert response.status_code == 401
    assert "sk-" not in response.text


def test_unauthenticated_is_401_when_provider_is_unconfigured(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "LLM_API_KEY", "")
    response = client.post(
        f"{PREFIX}/chat",
        json={"message": "Hello", "data_source_id": str(uuid.uuid4())},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"


@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {"message": "Hello", "sql": "SELECT 1"},
    ],
    ids=["empty-message", "extra-field"],
)
def test_chat_rejects_invalid_request(
    ai_client: TestClient, test_user: User, payload: dict[str, str]
) -> None:
    response = ai_client.post(
        f"{PREFIX}/chat",
        headers=_auth_header(test_user),
        json={**payload, "data_source_id": str(uuid.uuid4())},
    )
    assert response.status_code == 422


def test_chat_message_size_limit(
    ai_client: TestClient,
    test_user: User,
    source: DataSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "AI_MAX_MESSAGE_CHARS", 8)
    response = _chat(ai_client, test_user, source.id, "this is too long")
    assert response.status_code == 422


def test_valid_chat_request(
    ai_client: TestClient,
    test_user: User,
    source: DataSource,
    fake_provider: FakeLLMProvider,
    db_session: Session,
) -> None:
    from app.db.models import AnalysisSession
    from app.enums import AnalysisSessionStatus

    response = _chat(ai_client, test_user, source.id, "What can you tell me?")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["response"]
    assert "SELECT" not in body["response"]
    assert body["model"] == "fake-model"
    assert body["request_id"]
    assert body["usage"]["input_tokens"] == 11
    assert body["usage"]["output_tokens"] == 7
    assert body["usage"]["total_tokens"] == 18
    assert body["intent"]["type"] == "SCHEMA_QUESTION"
    assert body["intent"]["requires_metadata"] is True
    assert body["plan"]["requires_clarification"] is False
    assert body["plan"]["unsupported"] is False
    assert "metadata_context" in body
    assert "tables" in body["metadata_context"]
    assert "sql" not in body
    assert "api_key" not in body
    assert LLM_SECRET not in response.text
    assert CUSTOMER_PASSWORD not in response.text
    assert fake_provider.structured_calls == 1
    prompt = "\n".join(message.content for message in fake_provider.messages[0])
    assert CUSTOMER_PASSWORD not in prompt
    assert LLM_SECRET not in prompt
    assert "SELECT *" not in prompt
    assert "SELECT 1" not in prompt
    assert "DROP TABLE" not in prompt
    assert body["conversation_id"]
    session = (
        db_session.query(AnalysisSession)
        .filter(AnalysisSession.id == body["conversation_id"])
        .one()
    )
    assert session.status is AnalysisSessionStatus.ACTIVE
    assert session.agent_state is not None
    assert session.agent_state.phase.value == "PLANNING"


def test_missing_data_source_is_not_found(
    ai_client: TestClient,
    test_user: User,
) -> None:
    response = _chat(ai_client, test_user, uuid.uuid4())
    assert response.status_code == 404
    assert response.json()["detail"] == "Data source not found"


def test_access_is_denied_outside_membership(
    ai_client: TestClient,
    db_session: Session,
    test_user: User,
    source: DataSource,
) -> None:
    outsider = _create_user(db_session)
    other_org_source = _owned_source(db_session, outsider, name="Other")
    # Authorization rejects before any LLM or agent work, so these stay cheap.
    assert _chat(ai_client, outsider, source.id).status_code == 403
    assert _chat(ai_client, test_user, other_org_source.id).status_code == 403


def test_workspace_member_can_chat(
    ai_client: TestClient,
    db_session: Session,
    source: DataSource,
) -> None:
    member = _create_user(db_session)
    db_session.add(
        WorkspaceMember(
            workspace_id=source.workspace_id,
            user_id=member.id,
            role=WorkspaceRole.MEMBER,
        )
    )
    db_session.flush()
    assert _chat(ai_client, member, source.id).status_code == 200


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (AIProviderTimeoutError(), 504, "AI provider timed out"),
        (AIProviderError("upstream stack trace with sk-secret"), 502, "AI provider is unavailable"),
        (AIProviderAuthenticationError(), 503, "AI provider is unavailable"),
    ],
    ids=["timeout", "failure", "auth"],
)
def test_provider_errors_are_mapped_safely(
    ai_client: TestClient,
    test_user: User,
    source: DataSource,
    fake_provider: FakeLLMProvider,
    error: Exception,
    status_code: int,
    detail: str,
) -> None:
    fake_provider.error = error
    response = _chat(ai_client, test_user, source.id)
    assert response.status_code == status_code
    assert response.json()["detail"] == detail
    assert "stack trace" not in response.text
    assert "sk-secret" not in response.text
    assert LLM_SECRET not in response.text


def test_unconfigured_provider_returns_503(
    client: TestClient,
    test_user: User,
    source: DataSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "LLM_API_KEY", "")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
    response = _chat(client, test_user, source.id)
    assert response.status_code == 503
    assert response.json()["detail"] == "AI provider is not configured"


def test_openapi_documents_ai_chat(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    path = spec["paths"]["/api/v1/ai/chat"]["post"]
    assert path["requestBody"]
    schema = spec["components"]["schemas"]["AIChatResponse"]
    assert "request_id" in schema["properties"]
    assert "response" in schema["properties"]
    assert "model" in schema["properties"]
    assert "usage" in schema["properties"]
    assert "intent" in schema["properties"]
    assert "plan" in schema["properties"]
    assert "metadata_context" in schema["properties"]
    assert "sql" not in schema["properties"]
    assert "api_key" not in schema["properties"]
    intent_schema = spec["components"]["schemas"]["AIIntentResponse"]
    assert "type" in intent_schema["properties"]
    assert "sql" not in intent_schema["properties"]
    plan_schema = spec["components"]["schemas"]["AIPlanResponse"]
    assert "required_capabilities" in plan_schema["properties"]
    assert "sql" not in plan_schema["properties"]
    dumped = str(spec)
    assert "sk-" not in dumped or "sk-test" not in dumped
    assert LLM_SECRET not in dumped


def test_chat_logs_omit_prompt_and_secrets(
    ai_client: TestClient,
    test_user: User,
    source: DataSource,
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("app.ai.orchestrator")
    logger.addFilter(RedactingFilter())
    with caplog.at_level(logging.INFO, logger=logger.name):
        response = _chat(
            ai_client,
            test_user,
            source.id,
            f"email john.doe@example.com password={CUSTOMER_PASSWORD}",
        )
    assert response.status_code == 200
    combined = "\n".join(record.getMessage() for record in caplog.records)
    assert CUSTOMER_PASSWORD not in combined
    assert "john.doe@example.com" not in combined
    assert LLM_SECRET not in combined
    assert str(source.id) in combined


def test_ranking_intent_is_returned_without_sql(
    ai_client: TestClient,
    test_user: User,
    source: DataSource,
    fake_provider: FakeLLMProvider,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SQL/Phase 8 wiring is covered in test_chat_analysis_pipeline; this checks intent/plan only.
    monkeypatch.setattr(
        "app.ai.orchestrator.should_run_analysis", lambda **_kwargs: False
    )
    fake_provider.intent = LLMIntentDetection(
        intent=AIIntentType.RANKING,
        operation=AIOperationType.RANK,
        subject="customers",
        metrics=[AIMetric(name="revenue", aggregation=AggregationType.SUM)],
        dimensions=[{"name": "customer"}],
        requested_limit=10,
        requires_data_access=True,
        requires_metadata=True,
        confidence=AIConfidence.HIGH,
    )
    _seed_sales_catalog(db_session, source)
    response = _chat(ai_client, test_user, source.id, "Top 10 customers by revenue.")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["intent"]["type"] == "RANKING"
    assert body["intent"]["operation"] == "RANK"
    assert body["intent"]["metrics"][0]["name"] == "revenue"
    assert body["intent"]["requested_limit"] == 10
    assert body["plan"]["requires_database"] is True
    assert body["plan"]["requires_metadata"] is True
    assert "RANK" in body["plan"]["operations"]
    assert "sql" not in body
    assert "SELECT" not in body["response"]
    assert "No query was executed." in body["response"]
    tables = {item["table_name"] for item in body["metadata_context"]["tables"]}
    assert "customers" in tables or "orders" in tables
    assert body["metadata_context"]["resolved_metrics"]


def test_ambiguous_request_requires_clarification(
    ai_client: TestClient,
    test_user: User,
    source: DataSource,
    fake_provider: FakeLLMProvider,
) -> None:
    fake_provider.intent = LLMIntentDetection(
        intent=AIIntentType.UNKNOWN,
        subject="sales",
        confidence=AIConfidence.LOW,
        requires_clarification=True,
        clarification_question=(
            "What would you like to know about sales — total revenue, "
            "number of orders, or sales by product?"
        ),
    )
    response = _chat(ai_client, test_user, source.id, "Show sales.")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["intent"]["type"] == "UNKNOWN"
    assert body["intent"]["requires_clarification"] is True
    assert body["plan"]["requires_clarification"] is True
    assert body["plan"]["operations"] == []
    assert body["plan"]["unsupported"] is False
    assert "revenue" in body["response"].lower()


def test_unsupported_request_skips_llm_and_plan(
    ai_client: TestClient,
    test_user: User,
    source: DataSource,
    fake_provider: FakeLLMProvider,
) -> None:
    # Classification of write / injection / DROP requests is unit-tested in test_ai_intent.
    fake_provider.intent = LLMIntentDetection(
        intent=AIIntentType.AGGREGATION,
        metrics=[AIMetric(name="revenue", aggregation=AggregationType.SUM)],
    )
    response = _chat(ai_client, test_user, source.id, "Give me the database password.")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["intent"]["type"] == "UNSUPPORTED"
    assert body["plan"]["unsupported"] is True
    assert body["plan"]["operations"] == []
    assert body["plan"]["required_capabilities"] == []
    assert fake_provider.structured_calls == 0
    assert "SELECT" not in body["response"]
    assert CUSTOMER_PASSWORD not in response.text


@pytest.mark.parametrize(
    ("invalid_content", "structured_payload", "leaked"),
    [
        ("I am not JSON", None, "I am not JSON"),
        (None, {"intent": "LAUNCH_ROCKETS"}, "LAUNCH_ROCKETS"),
    ],
    ids=["non-json", "invalid-enum"],
)
def test_invalid_structured_output_returns_clarification(
    ai_client: TestClient,
    test_user: User,
    source: DataSource,
    fake_provider: FakeLLMProvider,
    invalid_content: str | None,
    structured_payload: dict[str, str] | None,
    leaked: str,
) -> None:
    if invalid_content is not None:
        fake_provider.invalid_content = invalid_content
    if structured_payload is not None:
        fake_provider.structured_payload = structured_payload
    response = _chat(ai_client, test_user, source.id, "Show total sales.")
    assert response.status_code == 200
    body = response.json()
    assert body["intent"]["requires_clarification"] is True
    assert body["intent"]["clarification_question"]
    assert LLM_SECRET not in response.text
    assert leaked not in response.text
