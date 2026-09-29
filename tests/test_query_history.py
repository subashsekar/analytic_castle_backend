"""Tests for Phase 7.5 Query History persistence."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, configure_mappers

from app.ai.sql_correction.models import (
    SQLCorrectionOutcome,
    SQLCorrectionStatus,
)
from app.ai.sql_execution.models import SQLExecutionResult, SQLExecutionStatus
from app.ai.sql_generation.models import (
    GeneratedSQL,
    SQLDialect,
    SQLGenerationConfidence,
)
from app.ai.sql_validation.models import (
    SQLValidationResult,
    SQLValidationViolation,
    SQLValidationViolationCode,
    ValidatedSQL,
)
from app.core.logging import redact_secret
from app.core.security import create_access_token
from app.db.models import (
    DataSource,
    Organization,
    QueryHistory,
    User,
    Workspace,
    WorkspaceMember,
)
from app.enums import (
    DataSourceType,
    QueryHistoryStatus,
    UserRole,
    WorkspaceRole,
)
from app.schemas.query_history import (
    QueryHistoryCreate,
    QueryHistoryFilter,
    QueryHistoryRead,
)
from app.services.query_history import (
    QueryHistoryAuthorizationError,
    QueryHistoryPaginationError,
    QueryHistoryReadError,
    QueryHistoryService,
)
from tests.conftest import _create_user, run_async

PREFIX = "/api/v1/workspaces"
ORG_PREFIX = "/api/v1/organizations"
WS_PREFIX = "/api/v1/workspaces"
VALID_SQL = "SELECT region, amount FROM public.orders"


def _auth_header(user: User) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user.id)}"}


def _seed_workspace(db_session: Session) -> tuple[User, Workspace, Organization]:
    user = _create_user(db_session, role=UserRole.USER)
    organization = Organization(
        name="AnalyticCastle",
        slug=f"analyticcastle-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(organization)
    db_session.flush()
    workspace = Workspace(
        organization_id=organization.id,
        name="Analytics",
        slug=f"analytics-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(workspace)
    db_session.flush()
    db_session.add(
        WorkspaceMember(
            workspace_id=workspace.id,
            user_id=user.id,
            role=WorkspaceRole.OWNER,
        )
    )
    db_session.flush()
    return user, workspace, organization


def _seed_data_source(
    db_session: Session,
    *,
    workspace: Workspace,
    user: User,
) -> DataSource:
    data_source = DataSource(
        workspace_id=workspace.id,
        name=f"Production DB {uuid.uuid4().hex[:6]}",
        type=DataSourceType.POSTGRESQL,
        created_by=user.id,
    )
    db_session.add(data_source)
    db_session.flush()
    return data_source


def _create_via_service(
    db_session: Session,
    *,
    user: User,
    workspace: Workspace,
    organization: Organization,
    data_source: DataSource | None = None,
    status: QueryHistoryStatus = QueryHistoryStatus.SUCCEEDED,
    generated_sql: str = VALID_SQL,
    duration_ms: float = 12.5,
    result_metadata: dict | None = None,
    error_metadata: dict | None = None,
    corrected_sql: str | None = None,
) -> QueryHistoryRead:
    service = QueryHistoryService(db_session)
    return service.create(
        QueryHistoryCreate(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
            data_source_id=data_source.id if data_source is not None else None,
            generated_sql=generated_sql,
            validated_sql=generated_sql,
            corrected_sql=corrected_sql,
            status=status,
            duration_ms=duration_ms,
            result_metadata=result_metadata
            or (
                {"row_count": 2, "column_count": 2, "truncated": False}
                if status is QueryHistoryStatus.SUCCEEDED
                else None
            ),
            error_metadata=error_metadata,
        )
    )


def test_query_history_mappers_configure() -> None:
    configure_mappers()


def test_query_history_creation(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    created = _create_via_service(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
    )
    row = db_session.scalar(select(QueryHistory).where(QueryHistory.id == created.id))
    assert row is not None
    assert row.status is QueryHistoryStatus.SUCCEEDED
    assert row.generated_sql == VALID_SQL
    assert row.duration_ms == 12.5
    assert row.result_metadata == {
        "row_count": 2,
        "column_count": 2,
        "truncated": False,
    }
    assert row.error_metadata is None
    assert row.created_at is not None


def test_successful_execution_history(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    service = QueryHistoryService(db_session)
    result = SQLExecutionResult(
        status=SQLExecutionStatus.SUCCEEDED,
        columns=["region", "amount"],
        rows=[["east", 42]],
        row_count=1,
        truncated=False,
        duration_ms=18.25,
        applied_row_limit=25,
        sql_char_count=len(VALID_SQL),
    )
    recorded = service.record_from_execution_result(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        data_source_id=data_source.id,
        generated_sql=VALID_SQL,
        validated_sql=VALID_SQL,
        result=result,
    )
    assert recorded.status is QueryHistoryStatus.SUCCEEDED
    assert recorded.duration_ms == 18.25
    assert recorded.result_metadata is not None
    assert recorded.result_metadata["row_count"] == 1
    assert recorded.result_metadata["column_count"] == 2
    assert recorded.result_metadata["truncated"] is False
    assert recorded.error_metadata is None


def test_failed_and_rejected_query_history(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    service = QueryHistoryService(db_session)

    failed = service.record_from_execution_result(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        data_source_id=data_source.id,
        generated_sql=VALID_SQL,
        validated_sql=VALID_SQL,
        result=SQLExecutionResult(
            status=SQLExecutionStatus.FAILED,
            duration_ms=9.0,
            sql_char_count=len(VALID_SQL),
        ),
    )
    assert failed.status is QueryHistoryStatus.FAILED
    assert failed.error_metadata is not None
    assert failed.error_metadata["execution_status"] == "FAILED"

    rejected_execution = service.record_from_execution_result(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        data_source_id=data_source.id,
        generated_sql=VALID_SQL,
        validated_sql=VALID_SQL,
        result=SQLExecutionResult(
            status=SQLExecutionStatus.REJECTED,
            duration_ms=3.0,
            sql_char_count=len(VALID_SQL),
        ),
    )
    assert rejected_execution.status is QueryHistoryStatus.REJECTED

    validation = SQLValidationResult(
        is_valid=False,
        validated=None,
        violations=[
            SQLValidationViolation(
                code=SQLValidationViolationCode.DANGEROUS_STATEMENT,
                message="DELETE is not allowed",
            )
        ],
    )
    rejected_validation = service.record_from_validation_result(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        data_source_id=data_source.id,
        generated_sql="DELETE FROM public.orders",
        validation=validation,
    )
    assert rejected_validation is not None
    assert rejected_validation.status is QueryHistoryStatus.REJECTED
    assert rejected_validation.error_metadata is not None
    assert rejected_validation.error_metadata["source"] == "VALIDATION"
    assert (
        "DANGEROUS_STATEMENT" in rejected_validation.error_metadata["violation_codes"]
    )


def test_corrected_query_history(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    service = QueryHistoryService(db_session)
    corrected_sql = "SELECT region FROM public.orders"
    outcome = SQLCorrectionOutcome(
        status=SQLCorrectionStatus.CORRECTED,
        validated=ValidatedSQL(
            sql=corrected_sql,
            referenced_tables=["public.orders"],
            referenced_columns=["public.orders.region"],
        ),
        generated=GeneratedSQL(
            sql=corrected_sql,
            dialect=SQLDialect.POSTGRESQL,
            referenced_tables=["public.orders"],
            referenced_columns=["public.orders.region"],
            confidence=SQLGenerationConfidence.MEDIUM,
            generation_version="test",
        ),
        attempt_count=1,
        max_attempts=2,
        schema_truncated=False,
    )
    recorded = service.record_from_correction_outcome(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        data_source_id=data_source.id,
        generated_sql="SELECT * FROM public.orders",
        outcome=outcome,
        duration_ms=44.0,
    )
    assert recorded is not None
    assert recorded.status is QueryHistoryStatus.CORRECTED
    assert recorded.generated_sql == "SELECT * FROM public.orders"
    assert recorded.corrected_sql == corrected_sql
    assert recorded.validated_sql == corrected_sql
    assert recorded.duration_ms == 44.0
    assert recorded.result_metadata is not None
    assert recorded.result_metadata["attempt_count"] == 1


def test_duration_status_result_metadata(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    created = _create_via_service(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
        status=QueryHistoryStatus.SUCCEEDED,
        duration_ms=123.456,
        result_metadata={"row_count": 5, "column_count": 3, "truncated": True},
    )
    assert created.duration_ms == 123.456
    assert created.status is QueryHistoryStatus.SUCCEEDED
    assert created.result_metadata == {
        "row_count": 5,
        "column_count": 3,
        "truncated": True,
    }


def test_pagination_and_filtering(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    for index, status in enumerate(
        (
            QueryHistoryStatus.SUCCEEDED,
            QueryHistoryStatus.FAILED,
            QueryHistoryStatus.REJECTED,
            QueryHistoryStatus.CORRECTED,
            QueryHistoryStatus.SUCCEEDED,
        )
    ):
        _create_via_service(
            db_session,
            user=user,
            workspace=workspace,
            organization=organization,
            data_source=data_source,
            status=status,
            generated_sql=f"{VALID_SQL} -- {index}",
            duration_ms=float(index + 1),
            corrected_sql="SELECT 1"
            if status is QueryHistoryStatus.CORRECTED
            else None,
            error_metadata={"code": "X"}
            if status is QueryHistoryStatus.FAILED
            else None,
            result_metadata=None
            if status is not QueryHistoryStatus.SUCCEEDED
            else {"row_count": index},
        )

    service = QueryHistoryService(db_session)
    page_one = service.list_for_user(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        page=1,
        page_size=2,
    )
    assert page_one.total == 5
    assert page_one.page == 1
    assert page_one.page_size == 2
    assert len(page_one.items) == 2
    assert page_one.items[0].created_at >= page_one.items[1].created_at

    succeeded = service.list_for_user(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        filter_params=QueryHistoryFilter(status=QueryHistoryStatus.SUCCEEDED),
        page=1,
        page_size=10,
    )
    assert succeeded.total == 2
    assert all(item.status is QueryHistoryStatus.SUCCEEDED for item in succeeded.items)

    by_duration = service.list_for_user(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        filter_params=QueryHistoryFilter(min_duration_ms=4.0, max_duration_ms=5.0),
        page=1,
        page_size=10,
    )
    assert by_duration.total == 2

    start = datetime.now(UTC) - timedelta(hours=1)
    by_date = service.list_for_user(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        filter_params=QueryHistoryFilter(start_date=start),
        page=1,
        page_size=10,
    )
    assert by_date.total == 5


def test_user_workspace_organization_isolation(db_session: Session) -> None:
    user_a, workspace_a, organization_a = _seed_workspace(db_session)
    user_b, workspace_b, organization_b = _seed_workspace(db_session)
    source_a = _seed_data_source(db_session, workspace=workspace_a, user=user_a)
    source_b = _seed_data_source(db_session, workspace=workspace_b, user=user_b)
    created_a = _create_via_service(
        db_session,
        user=user_a,
        workspace=workspace_a,
        organization=organization_a,
        data_source=source_a,
    )
    _create_via_service(
        db_session,
        user=user_b,
        workspace=workspace_b,
        organization=organization_b,
        data_source=source_b,
        generated_sql="SELECT 1",
    )

    service = QueryHistoryService(db_session)
    listed_a = service.list_for_user(
        user_id=user_a.id,
        workspace_id=workspace_a.id,
        organization_id=organization_a.id,
        page=1,
        page_size=20,
    )
    assert listed_a.total == 1
    assert listed_a.items[0].id == created_a.id

    listed_b = service.list_for_user(
        user_id=user_b.id,
        workspace_id=workspace_b.id,
        organization_id=organization_b.id,
        page=1,
        page_size=20,
    )
    assert listed_b.total == 1
    assert listed_b.items[0].id != created_a.id

    # Same user id cannot read another workspace/org even with a known id.
    with pytest.raises(QueryHistoryReadError):
        service.get(
            created_a.id,
            user_id=user_a.id,
            workspace_id=workspace_b.id,
            organization_id=organization_b.id,
        )


def test_idor_protection_service_and_api(
    client: TestClient,
    db_session: Session,
) -> None:
    user_a, workspace_a, organization_a = _seed_workspace(db_session)
    user_b, _workspace_b, _organization_b = _seed_workspace(db_session)
    source_a = _seed_data_source(db_session, workspace=workspace_a, user=user_a)
    created = _create_via_service(
        db_session,
        user=user_a,
        workspace=workspace_a,
        organization=organization_a,
        data_source=source_a,
    )
    db_session.flush()

    service = QueryHistoryService(db_session)
    with pytest.raises(QueryHistoryReadError):
        service.get(
            created.id,
            user_id=user_b.id,
            workspace_id=workspace_a.id,
            organization_id=organization_a.id,
        )

    # Outsider is not a member of workspace_a.
    denied = client.get(
        f"{PREFIX}/{workspace_a.id}/query-history/{created.id}",
        headers=_auth_header(user_b),
    )
    assert denied.status_code == 403

    # Owner can read their entry.
    ok = client.get(
        f"{PREFIX}/{workspace_a.id}/query-history/{created.id}",
        headers=_auth_header(user_a),
    )
    assert ok.status_code == 200
    assert ok.json()["id"] == str(created.id)
    assert ok.json()["generated_sql"] == VALID_SQL

    # Cross-workspace membership still cannot read another org's entry by id.
    db_session.add(
        WorkspaceMember(
            workspace_id=workspace_a.id,
            user_id=user_b.id,
            role=WorkspaceRole.MEMBER,
        )
    )
    db_session.flush()
    cross = client.get(
        f"{PREFIX}/{workspace_a.id}/query-history/{created.id}",
        headers=_auth_header(user_b),
    )
    assert cross.status_code == 404

    # Foreign data source listing is not found.
    foreign_ds = client.get(
        f"{PREFIX}/{workspace_a.id}/query-history/data-sources/{uuid.uuid4()}",
        headers=_auth_header(user_a),
    )
    assert foreign_ds.status_code == 404


def test_sensitive_data_protection(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    service = QueryHistoryService(db_session)
    secret_blob = (
        "password=CustomerDbPassword!42 "
        "api_key=sk-live-abcdef "
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.aaa.bbb "
        "connection_string=postgresql://user:secret@host/db"
    )
    recorded = service.record_failed(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        data_source_id=data_source.id,
        generated_sql=VALID_SQL,
        duration_ms=1.0,
        error_metadata={
            "message": secret_blob,
            "nested": {"token": "access_token=abc123"},
        },
    )
    assert recorded.error_metadata is not None
    serialized = str(recorded.error_metadata)
    assert "CustomerDbPassword!42" not in serialized
    assert "sk-live-abcdef" not in serialized
    assert "postgresql://user:secret@host/db" not in serialized
    assert "[REDACTED]" in serialized
    assert redact_secret(secret_blob) in recorded.error_metadata["message"]


def test_concurrent_history_writes(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    service = QueryHistoryService(db_session)

    async def _write(index: int) -> uuid.UUID:
        recorded = service.create(
            QueryHistoryCreate(
                user_id=user.id,
                workspace_id=workspace.id,
                organization_id=organization.id,
                data_source_id=data_source.id,
                generated_sql=f"{VALID_SQL} -- concurrent-{index}",
                status=QueryHistoryStatus.SUCCEEDED,
                duration_ms=float(index),
                result_metadata={"row_count": index},
            )
        )
        return recorded.id

    async def _run() -> list[uuid.UUID]:
        return list(await asyncio.gather(*(_write(index) for index in range(8))))

    ids = run_async(_run())
    assert len(ids) == 8
    assert len(set(ids)) == 8
    listed = service.list_for_user(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        page=1,
        page_size=20,
    )
    assert listed.total == 8


def test_query_history_api_list_and_filter(
    client: TestClient,
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _create_via_service(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
        status=QueryHistoryStatus.SUCCEEDED,
    )
    _create_via_service(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
        status=QueryHistoryStatus.FAILED,
        generated_sql="SELECT 1",
        error_metadata={"code": "FAILED"},
        result_metadata=None,
    )
    db_session.flush()

    listed = client.get(
        f"{PREFIX}/{workspace.id}/query-history",
        headers=_auth_header(user),
        params={"page": 1, "page_size": 10, "status": "SUCCEEDED"},
    )
    assert listed.status_code == 200
    body = listed.json()
    assert body["total"] == 1
    assert body["items"][0]["status"] == "SUCCEEDED"

    by_ds = client.get(
        f"{PREFIX}/{workspace.id}/query-history/data-sources/{data_source.id}",
        headers=_auth_header(user),
    )
    assert by_ds.status_code == 200
    assert by_ds.json()["total"] == 2


def test_query_history_write_requires_workspace_membership(db_session: Session) -> None:
    owner, workspace, organization = _seed_workspace(db_session)
    outsider = _create_user(db_session, role=UserRole.USER)
    data_source = _seed_data_source(db_session, workspace=workspace, user=owner)
    service = QueryHistoryService(db_session)
    with pytest.raises(QueryHistoryAuthorizationError, match="not a member"):
        service.create(
            QueryHistoryCreate(
                user_id=outsider.id,
                workspace_id=workspace.id,
                organization_id=organization.id,
                data_source_id=data_source.id,
                generated_sql=VALID_SQL,
                status=QueryHistoryStatus.SUCCEEDED,
                duration_ms=1.0,
            )
        )


def test_query_history_scrubs_sql_literals(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    recorded = _create_via_service(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
        generated_sql=(
            "SELECT * FROM public.orders "
            "WHERE email = 'alice@example.com' AND note = $$secret$$"
        ),
    )
    assert "alice@example.com" not in recorded.generated_sql
    assert "secret" not in recorded.generated_sql
    assert "[REDACTED]" in recorded.generated_sql


def test_record_succeeded_and_rejected_helpers(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    service = QueryHistoryService(db_session)
    succeeded = service.record_succeeded(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        data_source_id=data_source.id,
        generated_sql=VALID_SQL,
        validated_sql=VALID_SQL,
        duration_ms=7.5,
        result_metadata={"row_count": 3},
    )
    assert succeeded.status is QueryHistoryStatus.SUCCEEDED
    assert succeeded.validated_sql == VALID_SQL
    assert succeeded.error_metadata is None

    rejected = service.record_rejected(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        data_source_id=None,
        generated_sql="DROP TABLE public.orders",
        duration_ms=0.0,
        error_metadata={"source": "VALIDATION"},
    )
    assert rejected.status is QueryHistoryStatus.REJECTED
    assert rejected.data_source_id is None
    assert rejected.result_metadata is None


def test_execution_result_maps_timeout_and_cancelled_to_failed(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    service = QueryHistoryService(db_session)
    for status in (SQLExecutionStatus.TIMEOUT, SQLExecutionStatus.CANCELLED):
        recorded = service.record_from_execution_result(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
            data_source_id=data_source.id,
            generated_sql=VALID_SQL,
            validated_sql=VALID_SQL,
            result=SQLExecutionResult(
                status=status,
                duration_ms=2.0,
                sql_char_count=len(VALID_SQL),
            ),
        )
        assert recorded.status is QueryHistoryStatus.FAILED
        assert recorded.error_metadata is not None
        assert recorded.error_metadata["execution_status"] == status.value
        assert recorded.result_metadata is None


def test_execution_result_records_applied_row_limit_and_truncation(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    service = QueryHistoryService(db_session)
    recorded = service.record_from_execution_result(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        data_source_id=data_source.id,
        generated_sql=VALID_SQL,
        validated_sql=VALID_SQL,
        result=SQLExecutionResult(
            status=SQLExecutionStatus.SUCCEEDED,
            columns=["region"],
            rows=[["east"], ["west"]],
            row_count=2,
            truncated=True,
            duration_ms=5.0,
            applied_row_limit=2,
            sql_char_count=len(VALID_SQL),
        ),
    )
    assert recorded.result_metadata == {
        "row_count": 2,
        "column_count": 1,
        "truncated": True,
        "applied_row_limit": 2,
    }


def test_valid_validation_result_is_not_recorded(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    service = QueryHistoryService(db_session)
    assert (
        service.record_from_validation_result(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
            data_source_id=data_source.id,
            generated_sql=VALID_SQL,
            validation=SQLValidationResult(
                is_valid=True,
                validated=ValidatedSQL(
                    sql=VALID_SQL,
                    referenced_tables=["public.orders"],
                    referenced_columns=["public.orders.region"],
                ),
            ),
        )
        is None
    )
    assert (
        db_session.scalar(select(QueryHistory).where(QueryHistory.user_id == user.id))
        is None
    )


def test_uncorrected_outcomes_are_not_recorded(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    service = QueryHistoryService(db_session)

    def record(outcome: SQLCorrectionOutcome) -> QueryHistoryRead | None:
        return service.record_from_correction_outcome(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
            data_source_id=data_source.id,
            generated_sql=VALID_SQL,
            outcome=outcome,
        )

    assert record(SQLCorrectionOutcome(status=SQLCorrectionStatus.UNCORRECTED)) is None
    assert (
        record(SQLCorrectionOutcome(status=SQLCorrectionStatus.CLARIFICATION_REQUIRED))
        is None
    )
    assert record(SQLCorrectionOutcome(status=SQLCorrectionStatus.UNCHANGED)) is None
    # CORRECTED without any SQL attached cannot be recorded either.
    assert record(SQLCorrectionOutcome(status=SQLCorrectionStatus.CORRECTED)) is None


def test_corrected_outcome_without_validated_sql_uses_generated(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    service = QueryHistoryService(db_session)
    corrected_sql = "SELECT region FROM public.orders"
    recorded = service.record_from_correction_outcome(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        data_source_id=data_source.id,
        generated_sql="SELECT * FROM public.orders",
        outcome=SQLCorrectionOutcome(
            status=SQLCorrectionStatus.CORRECTED,
            generated=GeneratedSQL(
                sql=corrected_sql,
                dialect=SQLDialect.POSTGRESQL,
                referenced_tables=["public.orders"],
                referenced_columns=["public.orders.region"],
                confidence=SQLGenerationConfidence.LOW,
                generation_version="test",
            ),
            attempt_count=2,
            max_attempts=2,
            schema_truncated=True,
        ),
        duration_ms=11.0,
    )
    assert recorded is not None
    assert recorded.corrected_sql == corrected_sql
    assert recorded.validated_sql is None
    assert recorded.result_metadata == {
        "attempt_count": 2,
        "max_attempts": 2,
        "schema_truncated": True,
    }


def test_pagination_parameter_validation(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _create_via_service(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
    )
    service = QueryHistoryService(db_session, max_page_size=5)

    def listed(page: int, page_size: int) -> None:
        service.list_for_user(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
            page=page,
            page_size=page_size,
        )

    with pytest.raises(QueryHistoryPaginationError, match="Page must be at least 1"):
        listed(page=0, page_size=5)
    with pytest.raises(
        QueryHistoryPaginationError, match="Page size must be at least 1"
    ):
        listed(page=1, page_size=0)
    with pytest.raises(QueryHistoryPaginationError, match="must not exceed 5"):
        listed(page=1, page_size=6)
    with pytest.raises(QueryHistoryPaginationError, match="exceeds available results"):
        listed(page=3, page_size=5)


def test_empty_history_returns_zero_total_page(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    service = QueryHistoryService(db_session)
    listed = service.list_for_user(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        page=4,
        page_size=10,
    )
    assert listed.total == 0
    assert listed.items == []
    assert listed.page == 4
    assert listed.page_size == 10


def test_list_for_data_source_scopes_and_filters(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    source_a = _seed_data_source(db_session, workspace=workspace, user=user)
    source_b = _seed_data_source(db_session, workspace=workspace, user=user)
    _create_via_service(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=source_a,
        status=QueryHistoryStatus.SUCCEEDED,
        duration_ms=10.0,
    )
    _create_via_service(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=source_a,
        status=QueryHistoryStatus.FAILED,
        duration_ms=90.0,
        result_metadata=None,
    )
    _create_via_service(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=source_b,
    )
    service = QueryHistoryService(db_session)
    listed = service.list_for_data_source(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        data_source_id=source_a.id,
    )
    assert listed.total == 2
    assert {item.data_source_id for item in listed.items} == {source_a.id}

    filtered = service.list_for_data_source(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        data_source_id=source_a.id,
        filter_params=QueryHistoryFilter(
            status=QueryHistoryStatus.FAILED,
            min_duration_ms=50.0,
            data_source_id=source_b.id,
        ),
    )
    # The requested data source always wins over a conflicting filter value.
    assert filtered.total == 1
    assert filtered.items[0].data_source_id == source_a.id


def test_list_rejects_foreign_workspace_and_data_source(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _other_user, other_workspace, other_organization = _seed_workspace(db_session)
    foreign_source = _seed_data_source(
        db_session,
        workspace=other_workspace,
        user=_other_user,
    )
    service = QueryHistoryService(db_session)
    with pytest.raises(QueryHistoryAuthorizationError, match="Workspace"):
        service.list_for_user(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=other_organization.id,
        )
    with pytest.raises(QueryHistoryAuthorizationError, match="Data source"):
        service.list_for_data_source(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
            data_source_id=foreign_source.id,
        )
    with pytest.raises(QueryHistoryAuthorizationError, match="Data source"):
        service.list_for_data_source(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
            data_source_id=uuid.uuid4(),
        )


def test_write_rejects_unknown_inactive_user_and_foreign_scopes(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _other_user, other_workspace, _other_org = _seed_workspace(db_session)
    foreign_source = _seed_data_source(
        db_session,
        workspace=other_workspace,
        user=_other_user,
    )
    service = QueryHistoryService(db_session)

    def create(**overrides: object) -> QueryHistoryRead:
        payload: dict[str, object] = {
            "user_id": user.id,
            "workspace_id": workspace.id,
            "organization_id": organization.id,
            "data_source_id": None,
            "generated_sql": VALID_SQL,
            "status": QueryHistoryStatus.SUCCEEDED,
            "duration_ms": 1.0,
        }
        payload.update(overrides)
        return service.create(QueryHistoryCreate(**payload))  # type: ignore[arg-type]

    with pytest.raises(QueryHistoryAuthorizationError, match="User is not authorized"):
        create(user_id=uuid.uuid4())

    with pytest.raises(QueryHistoryAuthorizationError, match="Workspace"):
        create(organization_id=uuid.uuid4())

    with pytest.raises(QueryHistoryAuthorizationError, match="Data source"):
        create(data_source_id=foreign_source.id)

    user.is_active = False
    db_session.flush()
    with pytest.raises(QueryHistoryAuthorizationError, match="User is not authorized"):
        create()


def test_get_rejects_unknown_history_id(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    service = QueryHistoryService(db_session)
    with pytest.raises(QueryHistoryReadError):
        service.get(
            uuid.uuid4(),
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
        )


def test_history_sql_is_stripped_and_bounded(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    long_sql = (
        "SELECT region FROM public.orders WHERE region <> " + "'x' AND 1=1 " * 300
    )
    recorded = _create_via_service(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
        generated_sql=f"   {long_sql[:3990]}   ",
    )
    assert len(recorded.generated_sql) <= 4000
    assert recorded.generated_sql.startswith("SELECT region")
    assert not recorded.generated_sql.endswith(" ")
    assert "'x'" not in recorded.generated_sql


def test_history_scrubs_tagged_dollar_quoted_blocks(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    recorded = _create_via_service(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
        generated_sql=(
            "SELECT $tag$super-secret-token$tag$ AS note FROM public.orders"
        ),
    )
    assert "super-secret-token" not in recorded.generated_sql
    assert "[REDACTED]" in recorded.generated_sql


def test_history_metadata_sanitization_handles_nested_types(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    service = QueryHistoryService(db_session)
    recorded = service.create(
        QueryHistoryCreate(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
            data_source_id=data_source.id,
            generated_sql=VALID_SQL,
            status=QueryHistoryStatus.SUCCEEDED,
            duration_ms=1.0,
            result_metadata={
                "row_count": 4,
                "truncated": False,
                "ratio": 0.25,
                "empty": None,
                "codes": ["OK", ("api_key=sk-live-secret",)],
                "identifier": data_source.id,
            },
        )
    )
    assert recorded.result_metadata is not None
    metadata = recorded.result_metadata
    assert metadata["row_count"] == 4
    assert metadata["truncated"] is False
    assert metadata["ratio"] == 0.25
    assert metadata["empty"] is None
    assert metadata["identifier"] == str(data_source.id)
    assert "sk-live-secret" not in str(metadata["codes"])


def test_history_filters_accept_naive_datetimes(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _create_via_service(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
    )
    service = QueryHistoryService(db_session)
    listed = service.list_for_user(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        filter_params=QueryHistoryFilter(
            start_date=datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1),
            end_date=datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=1),
        ),
    )
    assert listed.total == 1

    outside = service.list_for_user(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        filter_params=QueryHistoryFilter(
            end_date=datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1),
        ),
    )
    assert outside.total == 0


def test_query_history_api_requires_authentication(
    client: TestClient,
    db_session: Session,
) -> None:
    _user, workspace, _organization = _seed_workspace(db_session)
    db_session.flush()
    assert client.get(f"{PREFIX}/{workspace.id}/query-history").status_code == 401
    assert (
        client.get(f"{PREFIX}/{workspace.id}/query-history/{uuid.uuid4()}").status_code
        == 401
    )
    assert (
        client.get(
            f"{PREFIX}/{workspace.id}/query-history/data-sources/{uuid.uuid4()}"
        ).status_code
        == 401
    )


def test_query_history_api_rejects_invalid_pagination(
    client: TestClient,
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _create_via_service(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
    )
    db_session.flush()
    headers = _auth_header(user)
    assert (
        client.get(
            f"{PREFIX}/{workspace.id}/query-history",
            headers=headers,
            params={"page": 0},
        ).status_code
        == 422
    )
    assert (
        client.get(
            f"{PREFIX}/{workspace.id}/query-history",
            headers=headers,
            params={"page_size": 101},
        ).status_code
        == 422
    )
    assert (
        client.get(
            f"{PREFIX}/{workspace.id}/query-history",
            headers=headers,
            params={"status": "NOT_A_STATUS"},
        ).status_code
        == 422
    )
    beyond = client.get(
        f"{PREFIX}/{workspace.id}/query-history",
        headers=headers,
        params={"page": 5, "page_size": 20},
    )
    assert beyond.status_code == 400
    assert "page" in beyond.json()["detail"].lower()


def test_query_history_api_data_source_filter_and_isolation(
    client: TestClient,
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    other_user, other_workspace, other_organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    foreign_source = _seed_data_source(
        db_session,
        workspace=other_workspace,
        user=other_user,
    )
    _create_via_service(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
    )
    _create_via_service(
        db_session,
        user=other_user,
        workspace=other_workspace,
        organization=other_organization,
        data_source=foreign_source,
        generated_sql="SELECT 2",
    )
    db_session.add(
        WorkspaceMember(
            workspace_id=workspace.id,
            user_id=other_user.id,
            role=WorkspaceRole.MEMBER,
        )
    )
    db_session.flush()

    scoped = client.get(
        f"{PREFIX}/{workspace.id}/query-history",
        headers=_auth_header(user),
        params={"data_source_id": str(data_source.id)},
    )
    assert scoped.status_code == 200
    assert scoped.json()["total"] == 1

    foreign_filter = client.get(
        f"{PREFIX}/{workspace.id}/query-history",
        headers=_auth_header(user),
        params={"data_source_id": str(foreign_source.id)},
    )
    assert foreign_filter.status_code == 404

    # A workspace member never sees another user's history entries.
    other_view = client.get(
        f"{PREFIX}/{workspace.id}/query-history",
        headers=_auth_header(other_user),
    )
    assert other_view.status_code == 200
    assert other_view.json()["total"] == 0


def test_query_history_api_unknown_entry_returns_404(
    client: TestClient,
    db_session: Session,
) -> None:
    user, workspace, _organization = _seed_workspace(db_session)
    db_session.flush()
    response = client.get(
        f"{PREFIX}/{workspace.id}/query-history/{uuid.uuid4()}",
        headers=_auth_header(user),
    )
    assert response.status_code == 404


def test_query_history_api_summary_hides_metadata(
    client: TestClient,
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _create_via_service(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
        error_metadata={"message": "password=CustomerDbPassword!42"},
    )
    db_session.flush()
    listed = client.get(
        f"{PREFIX}/{workspace.id}/query-history",
        headers=_auth_header(user),
    )
    assert listed.status_code == 200
    item = listed.json()["items"][0]
    assert set(item) == {
        "id",
        "data_source_id",
        "generated_sql",
        "status",
        "duration_ms",
        "created_at",
    }
    assert "CustomerDbPassword!42" not in listed.text
