"""Unit and security tests for Phase 7.4 SQL correction."""

from __future__ import annotations

import ast
import json
import logging
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import httpx
import pytest
from sqlalchemy.orm import Session

from app.ai.exceptions import (
    AIContextError,
    AIProviderError,
    AIProviderTimeoutError,
    AIRequestValidationError,
    AIResponseValidationError,
)
from app.ai.intent_types import (
    AIConfidence,
    AIIntent,
    AIIntentType,
    AIOperationType,
)
from app.ai.llm import AsyncLLMClient, LLMClientConfig
from app.ai.llm.errors import (
    LLMRateLimitError,
    LLMResponseValidationError,
    LLMTimeoutError,
)
from app.ai.metadata_types import ResolvedMetadataContext, empty_resolved_context
from app.ai.prompt import (
    PromptVariables,
    PromptVersionError,
    extract_placeholders,
)
from app.ai.sql_correction import (
    CORRECTABLE_VIOLATION_CODES,
    NON_CORRECTABLE_VIOLATION_CODES,
    SQL_CORRECTION_BUNDLE_VERSION,
    SQL_CORRECTION_USER_TEMPLATE_ID,
    SQLCorrectionAuthorizationError,
    SQLCorrectionConfigurationError,
    SQLCorrectionError,
    SQLCorrectionErrorCode,
    SQLCorrectionLLMError,
    SQLCorrectionOutcome,
    SQLCorrectionSchemaError,
    SQLCorrectionService,
    SQLCorrectionServiceResult,
    SQLCorrectionStatus,
    SQLCorrectionValidationError,
    SQLCorrectionVariables,
    SQLCorrectParams,
    SQLErrorAnalysis,
    SQLErrorCorrectability,
    SQLErrorSource,
    analysis_log_context,
    analyze_sql_failure,
    build_correction_feedback,
    build_sql_correction_prompt_registry,
    correct_failed_sql,
    correct_sql,
    correction_log_context,
    execute_corrected_sql,
    has_correction_failure,
    map_sql_correction_error,
    normalize_correction_output,
    parse_llm_correction,
    require_correction_failure,
    require_validated_correction,
    resolve_max_attempts,
    serialize_correction_error,
    serialize_correction_outcome,
    serialize_correction_service_result,
    sql_is_unchanged,
)
from app.ai.sql_correction.prompts import SQL_CORRECTION_SYSTEM_PROMPT_ID
from app.ai.sql_execution import SQLExecutionStatus
from app.ai.sql_generation import SchemaPromptContext, build_schema_prompt_context
from app.ai.sql_generation.models import LLMSQLOutput
from app.ai.sql_validation.models import (
    SQLValidationViolation,
    SQLValidationViolationCode,
)
from app.ai.state import AgentStateService
from app.connectors.types import QueryResult
from app.core.config import settings
from app.core.logging import RedactingFilter, redact_secret
from app.db.models import DataSource, Organization, User, Workspace, WorkspaceMember
from app.enums import UserRole, WorkspacePermission, WorkspaceRole
from app.mcp import POSTGRES_QUERY_TOOL_NAME, build_postgres_mcp
from tests.conftest import _create_user, run_async
from tests.test_sql_execution import _attach_connection
from tests.test_sql_generation import (
    _fake_metadata,
    _FixedSnapshotStateService,
    _seed_orders_catalog,
    _StaticContentClient,
)
from tests.test_supervisor import _create_session, _seed_data_source, _seed_workspace

FAKE_API_KEY = "sk-fake-sql-correction-key-not-real"
SECRET_SNIPPET = "CustomerDbPassword!@# 42"
_SQL_CORRECTION_DIR = (
    Path(__file__).resolve().parents[1] / "app" / "ai" / "sql_correction"
)
FAILED_SQL = "SELECT region, revenue FROM public.orders"
VALID_SQL = "SELECT region, amount FROM public.orders"


def _client_config(**overrides: object) -> LLMClientConfig:
    payload: dict[str, object] = {
        "api_key": FAKE_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4o-mini",
        "timeout": 5.0,
        "temperature": 0.2,
        "max_tokens": 512,
        "max_retries": 0,
        "retry_base_backoff": 0.1,
        "retry_max_backoff": 1.0,
    }
    payload.update(overrides)
    return LLMClientConfig(**payload)  # type: ignore[arg-type]


def _success_response(content: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "corr-sql-1",
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
    return AsyncLLMClient(_client_config(), transport=httpx.MockTransport(handler))


def _sql_body(**overrides: object) -> dict[str, Any]:
    body: dict[str, Any] = {
        "sql": VALID_SQL,
        "dialect": "postgresql",
        "referenced_tables": ["public.orders"],
        "referenced_columns": ["public.orders.region", "public.orders.amount"],
        "explanation": "Replace unknown revenue with amount",
        "confidence": "HIGH",
        "requires_clarification": False,
        "assumptions": ["amount is the revenue metric"],
        "suggested_limit": 100,
    }
    body.update(overrides)
    return body


def _analytical_intent() -> AIIntent:
    return AIIntent(
        intent=AIIntentType.ANALYTICAL_QUERY,
        operation=AIOperationType.AGGREGATE,
        subject="orders",
        confidence=AIConfidence.HIGH,
        requires_data_access=True,
        requires_metadata=True,
    )


def _violation(
    code: SQLValidationViolationCode,
    message: str = "failed",
    identifier: str | None = "revenue",
) -> SQLValidationViolation:
    return SQLValidationViolation(code=code, message=message, identifier=identifier)


def _unknown_column_violation() -> SQLValidationViolation:
    return _violation(
        SQLValidationViolationCode.UNKNOWN_COLUMN,
        "Unknown column revenue",
        "revenue",
    )


def _seed_member(
    db_session: Session,
    *,
    workspace: Workspace,
    user: User,
    role: WorkspaceRole = WorkspaceRole.OWNER,
) -> WorkspaceMember:
    member = WorkspaceMember(
        workspace_id=workspace.id,
        user_id=user.id,
        role=role,
    )
    db_session.add(member)
    db_session.flush()
    return member


def _authorized_params(
    *,
    user: User,
    workspace: Workspace,
    organization: Organization,
    data_source: DataSource,
    metadata: ResolvedMetadataContext,
    sql: str = FAILED_SQL,
    session_id: UUID | None = None,
    violations: tuple[SQLValidationViolation, ...] = (),
    execution_error_code: str | None = None,
    execution_error_message: str | None = None,
    max_attempts: int | None = None,
    message: str = "Show revenue by region",
) -> SQLCorrectParams:
    if not violations and not execution_error_code and not execution_error_message:
        violations = (_unknown_column_violation(),)
    return SQLCorrectParams(
        workspace_id=workspace.id,
        organization_id=organization.id,
        user_id=user.id,
        data_source_id=data_source.id,
        sql=sql,
        metadata=metadata,
        message=message,
        intent=_analytical_intent(),
        data_source_name=data_source.name,
        session_id=session_id,
        violations=violations,
        execution_error_code=execution_error_code,
        execution_error_message=execution_error_message,
        max_attempts=max_attempts,
    )


def test_analyze_correctable_validation_errors() -> None:
    analysis = analyze_sql_failure(violations=(_unknown_column_violation(),))
    assert analysis.correctability is SQLErrorCorrectability.CORRECTABLE
    assert analysis.source is SQLErrorSource.VALIDATION
    assert "UNKNOWN_COLUMN" in analysis.codes


def test_analyze_correctable_execution_database_error() -> None:
    analysis = analyze_sql_failure(
        execution_error_code="SQL_EXECUTION_DATABASE_ERROR",
        execution_error_message="column revenue does not exist",
    )
    assert analysis.correctability is SQLErrorCorrectability.CORRECTABLE
    assert analysis.source is SQLErrorSource.EXECUTION


def test_analyze_non_correctable_dangerous_sql() -> None:
    analysis = analyze_sql_failure(
        violations=(
            _violation(
                SQLValidationViolationCode.DANGEROUS_STATEMENT,
                "DELETE is not allowed",
                None,
            ),
        )
    )
    assert analysis.correctability is SQLErrorCorrectability.NON_CORRECTABLE


def test_analyze_non_correctable_execution_timeout() -> None:
    analysis = analyze_sql_failure(
        execution_error_code="SQL_EXECUTION_TIMEOUT_ERROR",
        execution_error_message="The query timed out",
    )
    assert analysis.correctability is SQLErrorCorrectability.NON_CORRECTABLE


def test_analyze_non_correctable_authorization() -> None:
    analysis = analyze_sql_failure(
        execution_error_code="SQL_EXECUTION_AUTHORIZATION_ERROR",
        execution_error_message="Data source is not accessible",
    )
    assert analysis.correctability is SQLErrorCorrectability.NON_CORRECTABLE


def test_correction_requires_an_actual_failure() -> None:
    with pytest.raises(SQLCorrectionValidationError, match="failure"):
        require_correction_failure()
    with pytest.raises(SQLCorrectionValidationError, match="failure"):
        analyze_sql_failure()


def test_parse_and_normalize_correction_output() -> None:
    parsed = parse_llm_correction(json.dumps(_sql_body()))
    outcome = normalize_correction_output(llm_result=parsed)
    assert outcome.generated is not None
    assert outcome.generated.sql == VALID_SQL
    assert outcome.generated.generation_version == "v1"


def test_parse_llm_correction_rejects_invalid_json() -> None:
    with pytest.raises(LLMResponseValidationError):
        parse_llm_correction("not-json")


def test_resolve_max_attempts_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "AI_SQL_CORRECTION_MAX_ATTEMPTS", 2)
    assert resolve_max_attempts(None) == 2
    assert resolve_max_attempts(1) == 1
    assert resolve_max_attempts(9) == 2
    with pytest.raises(SQLCorrectionValidationError):
        resolve_max_attempts(0)


def test_feedback_redacts_secrets() -> None:
    text = build_correction_feedback(
        violations=(
            _violation(
                SQLValidationViolationCode.UNKNOWN_COLUMN,
                f"password={SECRET_SNIPPET}",
                "revenue",
            ),
        )
    )
    assert SECRET_SNIPPET not in text
    assert "UNKNOWN_COLUMN" in text


def test_feedback_redacts_literals_and_identifiers() -> None:
    text = build_correction_feedback(
        execution_error_code="UNDEFINED_COLUMN",
        execution_error_message=(
            "column public.orders.ssn does not exist for value 'alice@example.com'"
        ),
    )
    assert "alice@example.com" not in text
    assert "public.orders" not in text
    assert "[REDACTED]" in text or "REDACTED_IDENTIFIER" in text


def test_successful_correction_of_unknown_column() -> None:
    outcome = run_async(
        correct_failed_sql(
            client=_mock_client(
                lambda _: httpx.Response(200, json=_success_response(_sql_body()))
            ),
            registry=build_sql_correction_prompt_registry(),
            params=SQLCorrectParams(
                workspace_id=uuid.uuid4(),
                organization_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                data_source_id=uuid.uuid4(),
                sql=FAILED_SQL,
                metadata=_fake_metadata(),
                violations=(_unknown_column_violation(),),
            ),
        )
    )
    assert outcome.status is SQLCorrectionStatus.CORRECTED
    assert outcome.validated is not None
    assert outcome.validated.sql.lower().startswith("select")
    assert outcome.attempt_count == 1
    assert outcome.analysis is not None
    assert outcome.analysis.correctability is SQLErrorCorrectability.CORRECTABLE


def test_non_correctable_error_does_not_call_llm() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls["count"] += 1
        return httpx.Response(200, json=_success_response(_sql_body()))

    outcome = run_async(
        correct_failed_sql(
            client=_mock_client(handler),
            registry=build_sql_correction_prompt_registry(),
            params=SQLCorrectParams(
                workspace_id=uuid.uuid4(),
                organization_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                data_source_id=uuid.uuid4(),
                sql="DELETE FROM public.orders",
                metadata=_fake_metadata(),
                violations=(
                    _violation(
                        SQLValidationViolationCode.DANGEROUS_STATEMENT,
                        "DELETE is not allowed",
                        None,
                    ),
                ),
            ),
        )
    )
    assert outcome.status is SQLCorrectionStatus.UNCORRECTED
    assert outcome.validated is None
    assert outcome.attempt_count == 0
    assert calls["count"] == 0


def test_correction_that_remains_invalid() -> None:
    outcome = run_async(
        correct_failed_sql(
            client=_mock_client(
                lambda _: httpx.Response(
                    200,
                    json=_success_response(
                        _sql_body(sql="SELECT region, missing FROM public.orders")
                    ),
                )
            ),
            registry=build_sql_correction_prompt_registry(),
            params=SQLCorrectParams(
                workspace_id=uuid.uuid4(),
                organization_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                data_source_id=uuid.uuid4(),
                sql=FAILED_SQL,
                metadata=_fake_metadata(),
                violations=(_unknown_column_violation(),),
                max_attempts=1,
            ),
        )
    )
    assert outcome.status is SQLCorrectionStatus.UNCORRECTED
    assert outcome.validated is None
    assert outcome.generated is not None
    assert any(
        item.code is SQLValidationViolationCode.UNKNOWN_COLUMN
        for item in outcome.violations
    )


def test_correction_that_attempts_dangerous_sql() -> None:
    outcome = run_async(
        correct_failed_sql(
            client=_mock_client(
                lambda _: httpx.Response(
                    200,
                    json=_success_response(_sql_body(sql="DELETE FROM public.orders")),
                )
            ),
            registry=build_sql_correction_prompt_registry(),
            params=SQLCorrectParams(
                workspace_id=uuid.uuid4(),
                organization_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                data_source_id=uuid.uuid4(),
                sql=FAILED_SQL,
                metadata=_fake_metadata(),
                violations=(_unknown_column_violation(),),
            ),
        )
    )
    assert outcome.status is SQLCorrectionStatus.UNCORRECTED
    assert outcome.validated is None
    assert outcome.analysis is not None
    assert outcome.analysis.correctability is SQLErrorCorrectability.NON_CORRECTABLE
    assert any(
        item.code
        in {
            SQLValidationViolationCode.DANGEROUS_STATEMENT,
            SQLValidationViolationCode.NOT_READONLY,
        }
        for item in outcome.violations
    )
    with pytest.raises(SQLCorrectionValidationError, match="validation"):
        require_validated_correction(outcome)


def test_correction_that_references_unknown_tables() -> None:
    outcome = run_async(
        correct_failed_sql(
            client=_mock_client(
                lambda _: httpx.Response(
                    200,
                    json=_success_response(
                        _sql_body(sql="SELECT region FROM public.ghost_table")
                    ),
                )
            ),
            registry=build_sql_correction_prompt_registry(),
            params=SQLCorrectParams(
                workspace_id=uuid.uuid4(),
                organization_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                data_source_id=uuid.uuid4(),
                sql=FAILED_SQL,
                metadata=_fake_metadata(),
                violations=(_unknown_column_violation(),),
                max_attempts=1,
            ),
        )
    )
    assert outcome.status is SQLCorrectionStatus.UNCORRECTED
    assert any(
        item.code is SQLValidationViolationCode.UNKNOWN_TABLE
        for item in outcome.violations
    )


def test_multiple_correction_attempts_then_success() -> None:
    payloads = [
        _sql_body(sql="SELECT region, missing FROM public.orders"),
        _sql_body(sql=VALID_SQL),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json=_success_response(payloads.pop(0)))

    outcome = run_async(
        correct_failed_sql(
            client=_mock_client(handler),
            registry=build_sql_correction_prompt_registry(),
            params=SQLCorrectParams(
                workspace_id=uuid.uuid4(),
                organization_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                data_source_id=uuid.uuid4(),
                sql=FAILED_SQL,
                metadata=_fake_metadata(),
                violations=(_unknown_column_violation(),),
                max_attempts=2,
            ),
        )
    )
    assert outcome.status is SQLCorrectionStatus.CORRECTED
    assert outcome.attempt_count == 2
    assert outcome.validated is not None


def test_maximum_retry_enforcement() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls["count"] += 1
        return httpx.Response(
            200,
            json=_success_response(
                _sql_body(
                    sql=f"SELECT region, missing{calls['count']} FROM public.orders"
                )
            ),
        )

    outcome = run_async(
        correct_failed_sql(
            client=_mock_client(handler),
            registry=build_sql_correction_prompt_registry(),
            params=SQLCorrectParams(
                workspace_id=uuid.uuid4(),
                organization_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                data_source_id=uuid.uuid4(),
                sql=FAILED_SQL,
                metadata=_fake_metadata(),
                violations=(_unknown_column_violation(),),
                max_attempts=2,
            ),
        )
    )
    assert outcome.status is SQLCorrectionStatus.UNCORRECTED
    assert outcome.attempt_count == 2
    assert outcome.max_attempts == 2
    assert calls["count"] == 2


def test_correct_sql_llm_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    with pytest.raises(SQLCorrectionLLMError):
        run_async(
            correct_sql(
                client=_mock_client(handler),
                registry=build_sql_correction_prompt_registry(),
                previous_sql=FAILED_SQL,
                metadata=_fake_metadata(),
                feedback="- UNKNOWN_COLUMN: revenue",
                attempt=1,
            )
        )


def test_correct_sql_llm_failure() -> None:
    with pytest.raises(SQLCorrectionLLMError):
        run_async(
            correct_sql(
                client=_mock_client(
                    lambda _: httpx.Response(
                        500, json={"error": {"message": "provider down"}}
                    )
                ),
                registry=build_sql_correction_prompt_registry(),
                previous_sql=FAILED_SQL,
                metadata=_fake_metadata(),
                feedback="- UNKNOWN_COLUMN: revenue",
                attempt=1,
            )
        )


def test_correct_sql_malformed_output() -> None:
    with pytest.raises(SQLCorrectionLLMError):
        run_async(
            correct_sql(
                client=_mock_client(
                    lambda _: httpx.Response(
                        200, json=_success_response({"not": "a sql payload"})
                    )
                ),
                registry=build_sql_correction_prompt_registry(),
                previous_sql=FAILED_SQL,
                metadata=_fake_metadata(),
                feedback="- UNKNOWN_COLUMN: revenue",
                attempt=1,
            )
        )


def test_correct_sql_rejects_missing_schema() -> None:
    with pytest.raises(SQLCorrectionSchemaError):
        run_async(
            correct_sql(
                client=_mock_client(
                    lambda _: httpx.Response(200, json=_success_response(_sql_body()))
                ),
                registry=build_sql_correction_prompt_registry(),
                previous_sql=FAILED_SQL,
                metadata=_fake_metadata(with_tables=False),
                feedback="- UNKNOWN_COLUMN: revenue",
                attempt=1,
            )
        )


def test_map_sql_correction_error_maps_timeout_and_validation() -> None:
    try:
        raise LLMTimeoutError("timed out")
    except LLMTimeoutError as exc:
        mapped_timeout = SQLCorrectionLLMError("timed out")
        mapped_timeout.__cause__ = exc
        assert isinstance(
            map_sql_correction_error(mapped_timeout), AIProviderTimeoutError
        )

    try:
        raise LLMResponseValidationError("invalid schema")
    except LLMResponseValidationError as exc:
        mapped_validation = SQLCorrectionLLMError("invalid schema")
        mapped_validation.__cause__ = exc
        assert isinstance(
            map_sql_correction_error(mapped_validation), AIResponseValidationError
        )

    assert isinstance(
        map_sql_correction_error(SQLCorrectionAuthorizationError("denied")),
        AIContextError,
    )
    assert isinstance(
        map_sql_correction_error(SQLCorrectionSchemaError("missing")),
        AIRequestValidationError,
    )
    assert isinstance(
        map_sql_correction_error(SQLCorrectionValidationError("bad")),
        AIResponseValidationError,
    )
    assert isinstance(
        map_sql_correction_error(SQLCorrectionLLMError("provider")),
        AIProviderError,
    )


def test_sensitive_prompt_variables_are_redacted_in_repr() -> None:
    variables = SQLCorrectionVariables(
        message="secret user question",
        intent="ANALYTICAL_QUERY",
        operation="AGGREGATE",
        subject="orders",
        data_source_name="Analytics DB",
        plan_summary="plan with secrets",
        suggested_limit="100",
        attempt="1",
        feedback="password=super-secret",
        previous_sql=FAILED_SQL,
        schema_context="public.orders amount",
    )
    rendered = repr(variables)
    assert "secret user question" not in rendered
    assert FAILED_SQL not in rendered
    assert "super-secret" not in rendered
    assert "[REDACTED]" in rendered


def test_correction_log_and_serialize_omit_sql() -> None:
    outcome = run_async(
        correct_failed_sql(
            client=_mock_client(
                lambda _: httpx.Response(200, json=_success_response(_sql_body()))
            ),
            registry=build_sql_correction_prompt_registry(),
            params=SQLCorrectParams(
                workspace_id=uuid.uuid4(),
                organization_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                data_source_id=uuid.uuid4(),
                sql=FAILED_SQL,
                metadata=_fake_metadata(),
                violations=(_unknown_column_violation(),),
            ),
        )
    )
    context = str(correction_log_context(outcome))
    serialized = str(serialize_correction_outcome(outcome))
    assert "SELECT" not in context
    assert "SELECT" not in serialized
    assert "revenue" not in context
    error_payload = serialize_correction_error(
        SQLCorrectionValidationError(f"password={SECRET_SNIPPET}")
    )
    assert SECRET_SNIPPET not in str(error_payload)


def test_sql_correction_service_success(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    metadata = _seed_orders_catalog(db_session, data_source)
    service = SQLCorrectionService(
        db_session,
        llm_client=_mock_client(
            lambda _: httpx.Response(200, json=_success_response(_sql_body()))
        ),
    )
    result = run_async(
        service.correct(
            _authorized_params(
                user=user,
                workspace=workspace,
                organization=organization,
                data_source=data_source,
                metadata=metadata,
            )
        )
    )
    assert result.outcome.status is SQLCorrectionStatus.CORRECTED
    assert result.organization_id == organization.id
    assert result.data_source_id == data_source.id


def test_sql_correction_rejects_non_member(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    metadata = _seed_orders_catalog(db_session, data_source)
    service = SQLCorrectionService(
        db_session,
        llm_client=_mock_client(
            lambda _: httpx.Response(200, json=_success_response(_sql_body()))
        ),
    )
    with pytest.raises(SQLCorrectionAuthorizationError, match="not a member"):
        run_async(
            service.correct(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                )
            )
        )


def test_sql_correction_rejects_organization_mismatch(db_session: Session) -> None:
    user, workspace, _organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    metadata = _seed_orders_catalog(db_session, data_source)
    other_org = Organization(name="Other Org", slug=f"other-{uuid.uuid4().hex[:8]}")
    db_session.add(other_org)
    db_session.flush()
    service = SQLCorrectionService(
        db_session,
        llm_client=_mock_client(
            lambda _: httpx.Response(200, json=_success_response(_sql_body()))
        ),
    )
    with pytest.raises(SQLCorrectionAuthorizationError, match="organization"):
        run_async(
            service.correct(
                SQLCorrectParams(
                    workspace_id=workspace.id,
                    organization_id=other_org.id,
                    user_id=user.id,
                    data_source_id=data_source.id,
                    sql=FAILED_SQL,
                    metadata=metadata,
                    violations=(_unknown_column_violation(),),
                )
            )
        )


def test_sql_correction_rejects_foreign_data_source(db_session: Session) -> None:
    user_a, workspace_a, organization_a = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace_a, user=user_a)
    user_b, workspace_b, _ = _seed_workspace(db_session)
    foreign = _seed_data_source(db_session, workspace=workspace_b, user=user_b)
    metadata = _seed_orders_catalog(db_session, foreign)
    service = SQLCorrectionService(
        db_session,
        llm_client=_mock_client(
            lambda _: httpx.Response(200, json=_success_response(_sql_body()))
        ),
    )
    with pytest.raises(SQLCorrectionAuthorizationError, match="not accessible"):
        run_async(
            service.correct(
                _authorized_params(
                    user=user_a,
                    workspace=workspace_a,
                    organization=organization_a,
                    data_source=foreign,
                    metadata=metadata,
                )
            )
        )


def test_sql_correction_rejects_invented_metadata_ids(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    service = SQLCorrectionService(
        db_session,
        llm_client=_mock_client(
            lambda _: httpx.Response(200, json=_success_response(_sql_body()))
        ),
    )
    with pytest.raises(SQLCorrectionAuthorizationError, match="unauthorized"):
        run_async(
            service.correct(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=_fake_metadata(data_source.id),
                )
            )
        )


def test_sql_correction_rejects_metadata_data_source_mismatch(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    service = SQLCorrectionService(
        db_session,
        llm_client=_mock_client(
            lambda _: httpx.Response(200, json=_success_response(_sql_body()))
        ),
    )
    with pytest.raises(SQLCorrectionAuthorizationError, match="data source"):
        run_async(
            service.correct(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=_fake_metadata(uuid.uuid4()),
                )
            )
        )


def test_sql_correction_session_cross_workspace_denied(db_session: Session) -> None:
    user_a, workspace_a, organization_a = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace_a, user=user_a)
    user_b, workspace_b, organization_b = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace_b, user=user_b)
    data_source_a = _seed_data_source(db_session, workspace=workspace_a, user=user_a)
    data_source_b = _seed_data_source(db_session, workspace=workspace_b, user=user_b)
    metadata_b = _seed_orders_catalog(db_session, data_source_b)
    _, snapshot = _create_session(
        db_session,
        user=user_a,
        workspace=workspace_a,
        organization=organization_a,
        data_source=data_source_a,
    )
    service = SQLCorrectionService(
        db_session,
        llm_client=_mock_client(
            lambda _: httpx.Response(200, json=_success_response(_sql_body()))
        ),
    )
    with pytest.raises(SQLCorrectionAuthorizationError, match="session"):
        run_async(
            service.correct(
                _authorized_params(
                    user=user_b,
                    workspace=workspace_b,
                    organization=organization_b,
                    data_source=data_source_b,
                    metadata=metadata_b,
                    session_id=snapshot.session_id,
                )
            )
        )


def test_sql_correction_service_missing_schema(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    service = SQLCorrectionService(
        db_session,
        llm_client=_mock_client(
            lambda _: httpx.Response(200, json=_success_response(_sql_body()))
        ),
    )
    with pytest.raises(SQLCorrectionSchemaError):
        run_async(
            service.correct(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=empty_resolved_context(data_source.id),
                )
            )
        )


def test_sql_correction_service_llm_timeout(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    metadata = _seed_orders_catalog(db_session, data_source)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    service = SQLCorrectionService(db_session, llm_client=_mock_client(handler))
    with pytest.raises(SQLCorrectionLLMError):
        run_async(
            service.correct(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                )
            )
        )


def test_revalidation_blocks_execution_of_dangerous_correction(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    service = SQLCorrectionService(
        db_session,
        llm_client=_mock_client(
            lambda _: httpx.Response(
                200,
                json=_success_response(_sql_body(sql="DROP TABLE public.orders")),
            )
        ),
    )
    registry, client = build_postgres_mcp(db_session)
    calls: list[str] = []

    async def _executor(config: object, sql: str, limit: int) -> QueryResult:
        del config, limit
        calls.append(sql)
        return QueryResult(columns=("region",), rows=(("west",),))

    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _executor

    async def _run() -> None:
        with pytest.raises(SQLCorrectionValidationError, match="validation"):
            await service.correct_then_execute(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                ),
                mcp_client=client,
            )

    run_async(_run())
    assert calls == []


def test_correct_then_execute_requires_query_permission(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user, role=WorkspaceRole.MEMBER)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    service = SQLCorrectionService(
        db_session,
        llm_client=_mock_client(
            lambda _: httpx.Response(200, json=_success_response(_sql_body()))
        ),
    )
    _, client = build_postgres_mcp(db_session)

    async def _run() -> None:
        with pytest.raises(SQLCorrectionAuthorizationError, match="permission"):
            await service.correct_then_execute(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                ),
                mcp_client=client,
            )

    run_async(_run())


def test_revalidation_then_execution_of_corrected_sql(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    service = SQLCorrectionService(
        db_session,
        llm_client=_mock_client(
            lambda _: httpx.Response(200, json=_success_response(_sql_body()))
        ),
    )
    registry, client = build_postgres_mcp(db_session)
    calls: list[str] = []

    async def _executor(config: object, sql: str, limit: int) -> QueryResult:
        del config, limit
        calls.append(sql)
        return QueryResult(columns=("region", "amount"), rows=(("west", 10),))

    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _executor
    params = _authorized_params(
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
        metadata=metadata,
    )

    async def _run() -> None:
        correction, execution = await service.correct_then_execute(
            params, mcp_client=client, limit=25
        )
        assert correction.outcome.status is SQLCorrectionStatus.CORRECTED
        assert execution.status is SQLExecutionStatus.SUCCEEDED
        assert execution.rows == [["west", 10]]

        with pytest.raises(SQLCorrectionValidationError):
            await execute_corrected_sql(
                client=client,
                params=params,
                outcome=correction.outcome.model_copy(
                    update={
                        "status": SQLCorrectionStatus.UNCORRECTED,
                        "validated": None,
                    }
                ),
            )

    run_async(_run())
    assert len(calls) == 1
    assert "DROP" not in calls[0].upper()
    assert "DELETE" not in calls[0].upper()


def test_correction_prompt_includes_failure_and_schema() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        captured["user"] = payload["messages"][-1]["content"]
        captured["system"] = payload["messages"][0]["content"]
        return httpx.Response(200, json=_success_response(_sql_body()))

    run_async(
        correct_sql(
            client=_mock_client(handler),
            registry=build_sql_correction_prompt_registry(),
            previous_sql=FAILED_SQL,
            metadata=_fake_metadata(),
            feedback="- UNKNOWN_COLUMN identifier=revenue: Unknown column",
            attempt=1,
            message="Show revenue",
        )
    )
    assert "UNKNOWN_COLUMN" in captured["user"]
    assert "public.orders" in captured["user"]
    assert FAILED_SQL in captured["user"]
    assert "do not execute sql" in captured["system"].lower()


def test_sql_correction_system_prompt_registered() -> None:
    registry = build_sql_correction_prompt_registry()
    system = registry.get_system(SQL_CORRECTION_SYSTEM_PROMPT_ID)
    assert "untrusted" in system.content.lower()
    assert "do not execute sql" in system.content.lower()


def test_sql_correction_package_has_no_direct_execution_imports() -> None:
    forbidden = {
        "app.connectors",
        "app.connectors.postgresql",
        "app.services.sample_data",
        "app.services.credentials",
        "psycopg",
        "asyncpg",
    }
    for path in _SQL_CORRECTION_DIR.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "execute_query" not in source, path
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not imported.intersection(forbidden), path


def test_correction_core_does_not_import_execution() -> None:
    for name in ("correction.py", "analysis.py", "validation.py", "feedback.py"):
        source = (_SQL_CORRECTION_DIR / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert "app.ai.sql_execution" not in imported
        assert "app.mcp" not in imported


def test_redacting_filter_masks_secrets_in_sql_correction_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "sk-test-sql-correction-secret"
    logger = logging.getLogger("tests.sql_correction.logging")
    caplog.set_level(logging.INFO)
    logger.info("sql correction event", extra={"token": redact_secret(secret)})
    record = caplog.records[-1]
    assert RedactingFilter().filter(record) is True
    assert secret not in record.getMessage()


def test_normalize_correction_requires_sql_or_clarification() -> None:
    with pytest.raises(SQLCorrectionValidationError):
        normalize_correction_output(
            llm_result=LLMSQLOutput.model_validate(
                _sql_body(sql=None, requires_clarification=False)
            )
        )


def _capture_prompts(
    payloads: list[dict[str, Any]] | None = None,
) -> tuple[list[str], Any]:
    """Return a captured user-prompt list and an LLM handler that fills it."""
    captured: list[str] = []
    queue = list(payloads or [_sql_body()])

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        captured.append(body["messages"][-1]["content"])
        payload = queue.pop(0) if len(queue) > 1 else queue[0]
        return httpx.Response(200, json=_success_response(payload))

    return captured, handler


def _workflow_params(**overrides: Any) -> SQLCorrectParams:
    params = SQLCorrectParams(
        workspace_id=uuid.uuid4(),
        organization_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        data_source_id=uuid.uuid4(),
        sql=FAILED_SQL,
        metadata=_fake_metadata(),
        violations=(_unknown_column_violation(),),
    )
    return replace(params, **overrides)


def _run_workflow(
    params: SQLCorrectParams,
    handler: Any,
) -> SQLCorrectionOutcome:
    return run_async(
        correct_failed_sql(
            client=_mock_client(handler),
            registry=build_sql_correction_prompt_registry(),
            params=params,
        )
    )


def test_correction_clarification_stops_the_retry_loop() -> None:
    captured, handler = _capture_prompts(
        [
            _sql_body(
                sql=None,
                requires_clarification=True,
                clarification_question="Which revenue column should be used?",
            )
        ]
    )
    outcome = _run_workflow(_workflow_params(max_attempts=2), handler)
    assert outcome.status is SQLCorrectionStatus.CLARIFICATION_REQUIRED
    assert outcome.requires_clarification is True
    assert outcome.clarification_question == "Which revenue column should be used?"
    assert outcome.attempt_count == 1
    assert outcome.validated is None
    assert len(captured) == 1


def test_correction_that_returns_unchanged_sql_stops_the_loop() -> None:
    echoed = "  select REGION, revenue\nfrom PUBLIC.orders ;  "
    captured, handler = _capture_prompts([_sql_body(sql=echoed)])
    outcome = _run_workflow(_workflow_params(max_attempts=3), handler)
    assert outcome.status is SQLCorrectionStatus.UNCHANGED
    assert outcome.validated is None
    assert outcome.generated is not None
    assert outcome.attempt_count == 1
    assert len(captured) == 1


def test_correction_feedback_switches_from_execution_to_validation() -> None:
    captured, handler = _capture_prompts(
        [_sql_body(sql="SELECT region, missing FROM public.orders"), _sql_body()]
    )
    outcome = _run_workflow(
        _workflow_params(
            violations=(),
            execution_error_code="SQL_EXECUTION_DATABASE_ERROR",
            execution_error_message="column revenue does not exist",
            max_attempts=2,
        ),
        handler,
    )
    assert outcome.status is SQLCorrectionStatus.CORRECTED
    assert outcome.attempt_count == 2
    assert len(captured) == 2
    assert "SQL_EXECUTION_DATABASE_ERROR" in captured[0]
    assert "Correction attempt: 1" in captured[0]
    assert "UNKNOWN_COLUMN" in captured[1]
    assert "SQL_EXECUTION_DATABASE_ERROR" not in captured[1]
    assert "Correction attempt: 2" in captured[1]


def test_exhausted_correction_reports_last_draft_and_violations() -> None:
    drafts = [
        _sql_body(sql="SELECT region, missing_one FROM public.orders"),
        _sql_body(sql="SELECT region, missing_two FROM public.orders"),
    ]
    outcome = _run_workflow(
        _workflow_params(max_attempts=2),
        _capture_prompts(drafts)[1],
    )
    assert outcome.status is SQLCorrectionStatus.UNCORRECTED
    assert outcome.attempt_count == 2
    assert outcome.generated is not None
    assert "missing_two" in outcome.generated.sql
    assert any(
        item.code is SQLValidationViolationCode.UNKNOWN_COLUMN
        for item in outcome.violations
    )
    assert outcome.analysis is not None
    assert outcome.analysis.source is SQLErrorSource.VALIDATION


def test_correction_stops_when_a_later_draft_is_non_correctable() -> None:
    captured, handler = _capture_prompts(
        [
            _sql_body(sql="SELECT region, missing FROM public.orders"),
            _sql_body(sql="SELECT region FROM public.orders; DROP TABLE public.orders"),
        ]
    )
    outcome = _run_workflow(_workflow_params(max_attempts=3), handler)
    assert outcome.status is SQLCorrectionStatus.UNCORRECTED
    assert outcome.attempt_count == 2
    assert len(captured) == 2
    assert outcome.analysis is not None
    assert outcome.analysis.correctability is SQLErrorCorrectability.NON_CORRECTABLE


def test_correct_failed_sql_requires_a_failure_before_calling_llm() -> None:
    captured, handler = _capture_prompts()
    with pytest.raises(SQLCorrectionValidationError, match="failure"):
        _run_workflow(_workflow_params(violations=()), handler)
    assert captured == []


def test_correct_failed_sql_requires_usable_schema() -> None:
    captured, handler = _capture_prompts()
    with pytest.raises(SQLCorrectionSchemaError, match="Schema context"):
        _run_workflow(
            _workflow_params(metadata=_fake_metadata(with_tables=False)), handler
        )
    assert captured == []


def test_workflow_caps_attempts_at_the_configured_maximum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "AI_SQL_CORRECTION_MAX_ATTEMPTS", 1)
    captured, handler = _capture_prompts(
        [_sql_body(sql="SELECT region, missing FROM public.orders")]
    )
    outcome = _run_workflow(_workflow_params(max_attempts=5), handler)
    assert outcome.max_attempts == 1
    assert outcome.attempt_count == 1
    assert len(captured) == 1


def test_workflow_uses_configured_default_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "AI_SQL_CORRECTION_MAX_ATTEMPTS", 2)
    captured, handler = _capture_prompts(
        [_sql_body(sql="SELECT region, missing FROM public.orders")]
    )
    outcome = _run_workflow(_workflow_params(max_attempts=None), handler)
    assert outcome.max_attempts == 2
    assert len(captured) == 2


def test_correction_reports_schema_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "AI_MAX_METADATA_COLUMNS", 1)
    metadata = _fake_metadata()
    assert build_schema_prompt_context(metadata).truncated is True
    outcome = _run_workflow(
        _workflow_params(metadata=metadata),
        _capture_prompts()[1],
    )
    assert outcome.status is SQLCorrectionStatus.CORRECTED
    assert outcome.schema_truncated is True


def test_has_correction_failure_signals() -> None:
    assert has_correction_failure() is False
    assert has_correction_failure(violations=(_unknown_column_violation(),)) is True
    assert has_correction_failure(execution_error_code="SQL_EXECUTION_ERROR") is True
    assert has_correction_failure(execution_error_message="boom") is True


def test_violation_code_classification_is_exhaustive() -> None:
    assert CORRECTABLE_VIOLATION_CODES.isdisjoint(NON_CORRECTABLE_VIOLATION_CODES)
    assert set(SQLValidationViolationCode) == (
        CORRECTABLE_VIOLATION_CODES | NON_CORRECTABLE_VIOLATION_CODES
    )


def test_analyze_unknown_execution_code_is_non_correctable() -> None:
    analysis = analyze_sql_failure(execution_error_code="MYSTERY_FAILURE")
    assert analysis.correctability is SQLErrorCorrectability.NON_CORRECTABLE
    assert analysis.source is SQLErrorSource.EXECUTION
    assert analysis.codes == ["MYSTERY_FAILURE"]


@pytest.mark.parametrize(
    "message",
    [
        "permission denied for relation orders",
        "rate limit exceeded for this data source",
        "result set is too large",
        "the statement was cancelled",
    ],
)
def test_analyze_non_correctable_execution_messages(message: str) -> None:
    analysis = analyze_sql_failure(execution_error_message=message)
    assert analysis.correctability is SQLErrorCorrectability.NON_CORRECTABLE


def test_analyze_correctable_execution_validation_code() -> None:
    analysis = analyze_sql_failure(
        execution_error_code="SQL_EXECUTION_VALIDATION_ERROR",
        execution_error_message="column mismatch",
    )
    assert analysis.correctability is SQLErrorCorrectability.CORRECTABLE


def test_analyze_prefers_validation_source_when_both_signals_present() -> None:
    analysis = analyze_sql_failure(
        violations=(_unknown_column_violation(),),
        execution_error_code="SQL_EXECUTION_DATABASE_ERROR",
    )
    assert analysis.source is SQLErrorSource.VALIDATION
    assert analysis.correctability is SQLErrorCorrectability.CORRECTABLE
    assert analysis.codes == ["UNKNOWN_COLUMN", "SQL_EXECUTION_DATABASE_ERROR"]


def test_analysis_log_context_caps_codes() -> None:
    analysis = SQLErrorAnalysis(
        correctability=SQLErrorCorrectability.CORRECTABLE,
        source=SQLErrorSource.VALIDATION,
        codes=[f"CODE_{index}" for index in range(15)],
        reason="many codes",
    )
    context = analysis_log_context(analysis)
    assert len(cast(list[str], context["error_codes"])) == 10


def test_feedback_caps_violation_lines() -> None:
    violations = tuple(
        _violation(SQLValidationViolationCode.UNKNOWN_COLUMN, f"missing col {index}")
        for index in range(25)
    )
    text = build_correction_feedback(violations=violations, max_chars=8_000)
    assert len(text.splitlines()) == 20


def test_feedback_truncates_to_the_char_limit() -> None:
    violations = tuple(
        _violation(SQLValidationViolationCode.UNKNOWN_COLUMN, "x" * 200)
        for _ in range(5)
    )
    text = build_correction_feedback(violations=violations, max_chars=80)
    assert len(text) <= 80
    assert text.endswith("...[truncated]")


def test_feedback_uses_configured_char_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "AI_SQL_CORRECTION_MAX_FEEDBACK_CHARS", 70)
    text = build_correction_feedback(
        violations=(_violation(SQLValidationViolationCode.UNKNOWN_COLUMN, "y" * 300),)
    )
    assert len(text) <= 70
    assert text.endswith("...[truncated]")


def test_feedback_execution_defaults_and_placeholder() -> None:
    code_only = build_correction_feedback(
        execution_error_code="SQL_EXECUTION_DATABASE_ERROR"
    )
    assert code_only == "- SQL_EXECUTION_DATABASE_ERROR: The database query failed"
    message_only = build_correction_feedback(execution_error_message="syntax error")
    assert message_only == "- EXECUTION_ERROR: syntax error"
    assert build_correction_feedback() == "- UNKNOWN: SQL needs correction"


def test_correct_sql_truncates_message_plan_and_previous_sql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "AI_MAX_MESSAGE_CHARS", 12)
    monkeypatch.setattr(settings, "AI_SQL_MAX_PLAN_SUMMARY_CHARS", 64)
    monkeypatch.setattr(settings, "AI_SQL_MAX_SQL_CHARS", 45)
    captured, handler = _capture_prompts()
    long_message = "show me the revenue by region please"
    long_plan = "step " * 40
    long_sql = "SELECT region, revenue, cost, margin, tax, discount FROM public.orders"
    run_async(
        correct_sql(
            client=_mock_client(handler),
            registry=build_sql_correction_prompt_registry(),
            previous_sql=long_sql,
            metadata=_fake_metadata(),
            feedback="- UNKNOWN_COLUMN: revenue",
            attempt=3,
            message=long_message,
            plan_summary=long_plan,
        )
    )
    prompt = captured[0]
    assert long_message not in prompt
    assert long_message[:12] in prompt
    assert long_sql not in prompt
    assert prompt.count("...[truncated]") == 2
    assert "Correction attempt: 3" in prompt


def test_correct_sql_defaults_optional_prompt_fields() -> None:
    captured, handler = _capture_prompts()
    run_async(
        correct_sql(
            client=_mock_client(handler),
            registry=build_sql_correction_prompt_registry(),
            previous_sql=FAILED_SQL,
            metadata=_fake_metadata(),
            feedback="- UNKNOWN_COLUMN: revenue",
            attempt=1,
        )
    )
    prompt = captured[0]
    assert "User message:\nnone" in prompt
    assert "Detected intent: none" in prompt
    assert "Intent operation: none" in prompt
    assert "Intent subject: none" in prompt
    assert "Data source name: none" in prompt
    assert "Plan summary: none" in prompt


def test_correct_sql_blank_feedback_falls_back_to_violations() -> None:
    captured, handler = _capture_prompts()
    run_async(
        correct_sql(
            client=_mock_client(handler),
            registry=build_sql_correction_prompt_registry(),
            previous_sql=FAILED_SQL,
            metadata=_fake_metadata(),
            feedback="   ",
            attempt=1,
            violations=[_unknown_column_violation()],
        )
    )
    assert "UNKNOWN_COLUMN" in captured[0]


def test_correct_sql_rejects_empty_response_content() -> None:
    client = cast(AsyncLLMClient, _StaticContentClient(""))
    with pytest.raises(SQLCorrectionLLMError, match="empty"):
        run_async(
            correct_sql(
                client=client,
                registry=build_sql_correction_prompt_registry(),
                previous_sql=FAILED_SQL,
                metadata=_fake_metadata(),
                feedback="- UNKNOWN_COLUMN: revenue",
                attempt=1,
            )
        )


def test_correct_sql_wraps_normalization_validation_errors() -> None:
    with pytest.raises(SQLCorrectionLLMError) as exc_info:
        run_async(
            correct_sql(
                client=_mock_client(
                    lambda _: httpx.Response(
                        200,
                        json=_success_response(
                            _sql_body(sql=None, requires_clarification=False)
                        ),
                    )
                ),
                registry=build_sql_correction_prompt_registry(),
                previous_sql=FAILED_SQL,
                metadata=_fake_metadata(),
                feedback="- UNKNOWN_COLUMN: revenue",
                attempt=1,
            )
        )
    assert isinstance(exc_info.value.__cause__, SQLCorrectionValidationError)
    assert isinstance(
        map_sql_correction_error(exc_info.value), AIResponseValidationError
    )


def test_correct_sql_rejects_blank_supplied_schema_context() -> None:
    with pytest.raises(SQLCorrectionSchemaError, match="Schema context"):
        run_async(
            correct_sql(
                client=_mock_client(
                    lambda _: httpx.Response(200, json=_success_response(_sql_body()))
                ),
                registry=build_sql_correction_prompt_registry(),
                previous_sql=FAILED_SQL,
                metadata=_fake_metadata(),
                feedback="- UNKNOWN_COLUMN: revenue",
                attempt=1,
                schema_context=SchemaPromptContext(
                    text="   ",
                    table_count=0,
                    column_count=0,
                    relationship_count=0,
                    truncated=True,
                ),
            )
        )


def test_sql_is_unchanged_ignores_whitespace_case_and_semicolons() -> None:
    assert sql_is_unchanged(FAILED_SQL, f"  {FAILED_SQL.upper()} ;  ") is True
    assert sql_is_unchanged(FAILED_SQL, VALID_SQL) is False


def test_normalize_correction_stamps_correction_version_for_clarification() -> None:
    outcome = normalize_correction_output(
        llm_result=LLMSQLOutput.model_validate(
            _sql_body(
                sql=None,
                requires_clarification=True,
                clarification_question="Which column?",
            )
        )
    )
    assert outcome.requires_clarification is True
    assert outcome.generated is None
    corrected = normalize_correction_output(
        llm_result=LLMSQLOutput.model_validate(_sql_body())
    )
    assert corrected.generated is not None
    assert corrected.generated.generation_version == SQL_CORRECTION_BUNDLE_VERSION.value


def test_serialize_correction_outcome_without_analysis_and_caps_codes() -> None:
    violations = [
        _violation(SQLValidationViolationCode.UNKNOWN_COLUMN, f"missing {index}")
        for index in range(25)
    ]
    outcome = SQLCorrectionOutcome(
        status=SQLCorrectionStatus.UNCORRECTED,
        violations=violations,
        attempt_count=2,
        max_attempts=2,
    )
    payload = serialize_correction_outcome(outcome)
    assert payload["violation_count"] == 25
    assert len(cast(list[str], payload["violation_codes"])) == 20
    assert "correctability" not in payload
    assert "error_codes" not in payload
    assert payload["has_validated_sql"] is False
    assert payload["has_generated_sql"] is False


def test_serialize_correction_service_result_includes_scope_without_sql() -> None:
    outcome = _run_workflow(_workflow_params(), _capture_prompts()[1])
    result = SQLCorrectionServiceResult(
        outcome=outcome,
        data_source_id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        organization_id=uuid.uuid4(),
    )
    payload = serialize_correction_service_result(result)
    assert payload["data_source_id"] == str(result.data_source_id)
    assert payload["workspace_id"] == str(result.workspace_id)
    assert payload["organization_id"] == str(result.organization_id)
    assert payload["has_validated_sql"] is True
    rendered = str(payload)
    assert "SELECT" not in rendered
    assert "orders" not in rendered


def test_correction_log_context_reports_no_sql_length_when_absent() -> None:
    outcome = SQLCorrectionOutcome(
        status=SQLCorrectionStatus.UNCORRECTED,
        attempt_count=0,
        max_attempts=2,
    )
    context = correction_log_context(outcome)
    assert context["sql_char_count"] is None
    assert context["violation_count"] == 0
    assert "correctability" not in context


def test_sql_correction_service_requires_configured_api_key(
    db_session: Session,
) -> None:
    with pytest.raises(SQLCorrectionConfigurationError, match="API key"):
        SQLCorrectionService(db_session, llm_config=_client_config(api_key=""))


def test_sql_correction_allows_super_admin_without_membership(
    db_session: Session,
) -> None:
    owner, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=owner)
    data_source = _seed_data_source(db_session, workspace=workspace, user=owner)
    metadata = _seed_orders_catalog(db_session, data_source)
    super_admin = _create_user(db_session, role=UserRole.SUPER_ADMIN)
    params = _authorized_params(
        user=super_admin,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
        metadata=metadata,
    )

    def build(*, allow_super_admin: bool) -> SQLCorrectionService:
        return SQLCorrectionService(
            db_session,
            llm_client=_mock_client(
                lambda _: httpx.Response(200, json=_success_response(_sql_body()))
            ),
            allow_super_admin=allow_super_admin,
        )

    result = run_async(build(allow_super_admin=True).correct(params))
    assert result.outcome.status is SQLCorrectionStatus.CORRECTED
    with pytest.raises(SQLCorrectionAuthorizationError, match="not a member"):
        run_async(build(allow_super_admin=False).correct(params))


def test_sql_correction_rejects_member_without_required_permission(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user, role=WorkspaceRole.MEMBER)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    metadata = _seed_orders_catalog(db_session, data_source)
    service = SQLCorrectionService(
        db_session,
        llm_client=_mock_client(
            lambda _: httpx.Response(200, json=_success_response(_sql_body()))
        ),
        required_permission=WorkspacePermission.DATA_SOURCE_QUERY,
    )
    with pytest.raises(SQLCorrectionAuthorizationError, match="permission"):
        run_async(
            service.correct(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                )
            )
        )


def test_sql_correction_rejects_inactive_user(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    metadata = _seed_orders_catalog(db_session, data_source)
    user.is_active = False
    db_session.flush()
    service = SQLCorrectionService(
        db_session,
        llm_client=_mock_client(
            lambda _: httpx.Response(200, json=_success_response(_sql_body()))
        ),
    )
    with pytest.raises(SQLCorrectionAuthorizationError, match="User is not authorized"):
        run_async(
            service.correct(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                )
            )
        )


def test_sql_correction_rejects_unknown_workspace(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    metadata = _seed_orders_catalog(db_session, data_source)
    service = SQLCorrectionService(
        db_session,
        llm_client=_mock_client(
            lambda _: httpx.Response(200, json=_success_response(_sql_body()))
        ),
    )
    params = replace(
        _authorized_params(
            user=user,
            workspace=workspace,
            organization=organization,
            data_source=data_source,
            metadata=metadata,
        ),
        workspace_id=uuid.uuid4(),
    )
    with pytest.raises(SQLCorrectionAuthorizationError, match="Workspace"):
        run_async(service.correct(params))


def test_sql_correction_session_scoped_request_succeeds(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    metadata = _seed_orders_catalog(db_session, data_source)
    _, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
    )
    service = SQLCorrectionService(
        db_session,
        llm_client=_mock_client(
            lambda _: httpx.Response(200, json=_success_response(_sql_body()))
        ),
    )
    result = run_async(
        service.correct(
            _authorized_params(
                user=user,
                workspace=workspace,
                organization=organization,
                data_source=data_source,
                metadata=metadata,
                session_id=snapshot.session_id,
            )
        )
    )
    assert result.outcome.status is SQLCorrectionStatus.CORRECTED


def test_sql_correction_session_without_data_source_denied(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    metadata = _seed_orders_catalog(db_session, data_source)
    _, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
    )
    service = SQLCorrectionService(
        db_session,
        llm_client=_mock_client(
            lambda _: httpx.Response(200, json=_success_response(_sql_body()))
        ),
    )
    with pytest.raises(
        SQLCorrectionAuthorizationError,
        match="no authorized data source",
    ):
        run_async(
            service.correct(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                    session_id=snapshot.session_id,
                )
            )
        )


def test_sql_correction_session_data_source_mismatch_denied(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    session_source = _seed_data_source(db_session, workspace=workspace, user=user)
    other_source = _seed_data_source(db_session, workspace=workspace, user=user)
    metadata = _seed_orders_catalog(db_session, other_source)
    _, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=session_source,
    )
    service = SQLCorrectionService(
        db_session,
        llm_client=_mock_client(
            lambda _: httpx.Response(200, json=_success_response(_sql_body()))
        ),
    )
    with pytest.raises(
        SQLCorrectionAuthorizationError,
        match="does not match the analysis session",
    ):
        run_async(
            service.correct(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=other_source,
                    metadata=metadata,
                    session_id=snapshot.session_id,
                )
            )
        )


def test_sql_correction_session_organization_drift_denied(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    metadata = _seed_orders_catalog(db_session, data_source)
    _, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
    )
    drifted = snapshot.model_copy(update={"organization_id": uuid.uuid4()})
    service = SQLCorrectionService(
        db_session,
        llm_client=_mock_client(
            lambda _: httpx.Response(200, json=_success_response(_sql_body()))
        ),
        state_service=cast(AgentStateService, _FixedSnapshotStateService(drifted)),
    )
    with pytest.raises(
        SQLCorrectionAuthorizationError,
        match="organization does not match",
    ):
        run_async(
            service.correct(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                    session_id=snapshot.session_id,
                )
            )
        )


def test_correct_then_execute_restores_correction_permission(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user, role=WorkspaceRole.MEMBER)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    service = SQLCorrectionService(
        db_session,
        llm_client=_mock_client(
            lambda _: httpx.Response(200, json=_success_response(_sql_body()))
        ),
    )
    _, client = build_postgres_mcp(db_session)
    params = _authorized_params(
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
        metadata=metadata,
    )

    async def _run() -> None:
        with pytest.raises(SQLCorrectionAuthorizationError, match="permission"):
            await service.correct_then_execute(params, mcp_client=client)
        result = await service.correct(params)
        assert result.outcome.status is SQLCorrectionStatus.CORRECTED

    run_async(_run())


def test_map_sql_correction_error_covers_remaining_arms() -> None:
    try:
        raise LLMRateLimitError("slow down")
    except LLMRateLimitError as exc:
        rate_limited = SQLCorrectionLLMError("slow down")
        rate_limited.__cause__ = exc
        assert isinstance(map_sql_correction_error(rate_limited), AIProviderError)

    assert isinstance(
        map_sql_correction_error(SQLCorrectionConfigurationError("no api key")),
        AIProviderError,
    )
    assert isinstance(
        map_sql_correction_error(SQLCorrectionError("generic failure")),
        AIProviderError,
    )


def test_sql_correction_errors_expose_codes_and_redact_secrets() -> None:
    cases: list[tuple[SQLCorrectionError, SQLCorrectionErrorCode]] = [
        (
            SQLCorrectionAuthorizationError("denied"),
            SQLCorrectionErrorCode.SQL_CORRECTION_AUTHORIZATION_ERROR,
        ),
        (
            SQLCorrectionSchemaError("no schema"),
            SQLCorrectionErrorCode.SQL_CORRECTION_SCHEMA_ERROR,
        ),
        (
            SQLCorrectionValidationError("bad draft"),
            SQLCorrectionErrorCode.SQL_CORRECTION_VALIDATION_ERROR,
        ),
        (
            SQLCorrectionLLMError("provider down"),
            SQLCorrectionErrorCode.SQL_CORRECTION_LLM_ERROR,
        ),
        (
            SQLCorrectionConfigurationError("missing key"),
            SQLCorrectionErrorCode.SQL_CORRECTION_CONFIGURATION_ERROR,
        ),
        (
            SQLCorrectionError("internal"),
            SQLCorrectionErrorCode.SQL_CORRECTION_INTERNAL_ERROR,
        ),
    ]
    for error, code in cases:
        assert error.code is code
        payload = serialize_correction_error(error)
        assert payload["code"] == code.value

    leaked = serialize_correction_error(
        SQLCorrectionLLMError(f"api_key={FAKE_API_KEY} password={SECRET_SNIPPET}")
    )
    assert FAKE_API_KEY not in str(leaked)
    assert SECRET_SNIPPET not in str(leaked)


def test_sql_correction_user_template_matches_variable_model() -> None:
    class _OtherVariables(PromptVariables):
        message: str

    registry = build_sql_correction_prompt_registry()
    template = registry.get_template(SQL_CORRECTION_USER_TEMPLATE_ID)
    assert set(extract_placeholders(template.template)) == set(
        SQLCorrectionVariables.model_fields
    )
    with pytest.raises(PromptVersionError):
        registry.render_template(
            SQL_CORRECTION_USER_TEMPLATE_ID,
            _OtherVariables(message="Show revenue"),
        )
