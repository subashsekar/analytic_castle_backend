"""Unit and security tests for Phase 7.2 SQL validation."""

from __future__ import annotations

import ast
import logging
import uuid
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest
from sqlalchemy.orm import Session

from app.ai.exceptions import AIContextError, AIRequestValidationError
from app.ai.metadata_types import (
    MetadataColumnCandidate,
    MetadataMatchReason,
    MetadataRelationshipCandidate,
    MetadataTableCandidate,
    ResolvedMetadataContext,
    empty_resolved_context,
)
from app.ai.sql_validation import (
    ExtractedReferences,
    QualifiedColumn,
    QualifiedTable,
    SQLValidateParams,
    SQLValidationAuthorizationError,
    SQLValidationColumnError,
    SQLValidationConfigurationError,
    SQLValidationError,
    SQLValidationErrorCode,
    SQLValidationParseError,
    SQLValidationReadonlyError,
    SQLValidationResult,
    SQLValidationSchemaError,
    SQLValidationService,
    SQLValidationTableError,
    SQLValidationViolation,
    SQLValidationViolationCode,
    build_schema_allowlist,
    ensure_readonly_sql,
    extract_sql_references,
    has_usable_allowlist,
    map_sql_validation_error,
    validate_generated_sql,
    validate_generated_sql_or_raise,
    validation_log_context,
)
from app.ai.sql_validation.identifiers import validate_identifiers
from app.ai.state import AgentStateService
from app.core.config import settings
from app.core.logging import redact_secret
from app.db.models import Organization, User, Workspace, WorkspaceMember
from app.enums import (
    DataSourceRelationshipType,
    UserRole,
    WorkspacePermission,
    WorkspaceRole,
)
from tests.conftest import _create_user
from tests.test_sql_generation import (
    _fake_metadata,
    _FixedSnapshotStateService,
    _seed_orders_catalog,
)
from tests.test_supervisor import _create_session, _seed_data_source, _seed_workspace

_SQL_VALIDATION_DIR = (
    Path(__file__).resolve().parents[1] / "app" / "ai" / "sql_validation"
)
SECRET_SNIPPET = "CustomerDbPassword!@# 42"


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
    data_source,
    metadata: ResolvedMetadataContext,
    sql: str = "SELECT region FROM public.orders",
    session_id: UUID | None = None,
) -> SQLValidateParams:
    return SQLValidateParams(
        workspace_id=workspace.id,
        organization_id=organization.id,
        user_id=user.id,
        data_source_id=data_source.id,
        sql=sql,
        metadata=metadata,
        session_id=session_id,
    )


def _codes(result) -> list[SQLValidationViolationCode]:
    return [item.code for item in result.violations]


# --- Core validation ---


def test_valid_select_passes() -> None:
    result = validate_generated_sql(
        "SELECT region, amount FROM public.orders",
        _fake_metadata(),
    )
    assert result.is_valid
    assert result.validated is not None
    assert result.validated.referenced_tables == ["public.orders"]
    assert "region" in result.validated.referenced_columns
    assert "amount" in result.validated.referenced_columns


def test_invalid_and_malformed_sql_rejected() -> None:
    empty = validate_generated_sql("   ", _fake_metadata())
    assert empty.is_valid is False
    assert SQLValidationViolationCode.EMPTY_SQL in _codes(empty)

    malformed = validate_generated_sql("SELECT FROM WHERE", _fake_metadata())
    assert malformed.is_valid is False
    assert SQLValidationViolationCode.PARSE_ERROR in _codes(malformed)

    with pytest.raises(SQLValidationParseError):
        validate_generated_sql_or_raise("SELECT FROM WHERE", _fake_metadata())


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO public.orders VALUES (1)",
        "UPDATE public.orders SET amount = 0",
        "DELETE FROM public.orders",
        "DROP TABLE public.orders",
        "ALTER TABLE public.orders ADD COLUMN x int",
        "CREATE TABLE public.orders (id int)",
        "TRUNCATE public.orders",
        "GRANT SELECT ON public.orders TO u",
        "REVOKE SELECT ON public.orders FROM u",
        "MERGE INTO public.orders USING s ON TRUE WHEN MATCHED THEN DELETE",
        "COPY public.orders TO STDOUT",
        "CALL do_thing()",
        "BEGIN",
        "SELECT * FROM public.orders FOR UPDATE",
        "SELECT * INTO copies FROM public.orders",
    ],
)
def test_write_and_destructive_statements_rejected(sql: str) -> None:
    result = validate_generated_sql(sql, _fake_metadata())
    assert result.is_valid is False
    assert set(_codes(result)) & {
        SQLValidationViolationCode.DANGEROUS_STATEMENT,
        SQLValidationViolationCode.NOT_READONLY,
        SQLValidationViolationCode.MULTI_STATEMENT,
    }
    with pytest.raises(SQLValidationReadonlyError):
        validate_generated_sql_or_raise(sql, _fake_metadata())


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT pg_sleep(1)",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT dblink_exec('dbname=x', 'SELECT 1')",
    ],
)
def test_dangerous_readonly_functions_rejected(sql: str) -> None:
    result = validate_generated_sql(sql, _fake_metadata())
    assert result.is_valid is False
    assert SQLValidationViolationCode.DANGEROUS_STATEMENT in _codes(result)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1; SELECT 2",
        "SELECT region FROM public.orders; DELETE FROM public.orders",
        "SELECT * FROM public.orders; DROP TABLE public.orders",
    ],
)
def test_multi_statements_rejected(sql: str) -> None:
    result = validate_generated_sql(sql, _fake_metadata())
    assert result.is_valid is False
    assert SQLValidationViolationCode.MULTI_STATEMENT in _codes(result)


@pytest.mark.parametrize(
    "sql",
    [
        "WITH x AS (DELETE FROM public.orders RETURNING *) SELECT * FROM x",
        "WITH x AS (INSERT INTO public.orders VALUES (1) RETURNING *) SELECT * FROM x",
        "WITH x AS (UPDATE public.orders SET amount = 1 RETURNING *) SELECT * FROM x",
    ],
)
def test_cte_write_attempts_rejected(sql: str) -> None:
    result = validate_generated_sql(sql, _fake_metadata())
    assert result.is_valid is False
    assert SQLValidationViolationCode.DANGEROUS_STATEMENT in _codes(result)


@pytest.mark.parametrize(
    "sql",
    [
        "SeLeCt region FrOm public.orders",
        "   SELECT\n\tregion\nFROM\tpublic.orders   ",
        "SELECT region FROM public.orders /* DELETE FROM public.orders */",
        "SELECT region FROM public.orders -- DROP TABLE public.orders",
        "SELECT /* inject */ region FROM public.orders",
        "SELECT 'delete' AS note, region FROM public.orders",
        "SELECT $$ drop table public.orders $$ AS note, region FROM public.orders",
    ],
)
def test_comment_whitespace_case_bypasses_do_not_allow_writes(sql: str) -> None:
    result = validate_generated_sql(sql, _fake_metadata())
    assert result.is_valid is True
    assert result.validated is not None


def test_comment_cannot_hide_second_statement() -> None:
    result = validate_generated_sql(
        "SELECT region FROM public.orders; /* DELETE FROM public.orders */ SELECT 2",
        _fake_metadata(),
    )
    assert result.is_valid is False
    assert SQLValidationViolationCode.MULTI_STATEMENT in _codes(result)


def test_unknown_table_rejected() -> None:
    result = validate_generated_sql(
        "SELECT id FROM public.secrets",
        _fake_metadata(),
    )
    assert result.is_valid is False
    assert SQLValidationViolationCode.UNKNOWN_TABLE in _codes(result)
    with pytest.raises(SQLValidationTableError):
        validate_generated_sql_or_raise(
            "SELECT id FROM public.secrets",
            _fake_metadata(),
        )


def test_unknown_column_rejected() -> None:
    result = validate_generated_sql(
        "SELECT password FROM public.orders",
        _fake_metadata(),
    )
    assert result.is_valid is False
    assert SQLValidationViolationCode.UNKNOWN_COLUMN in _codes(result)
    with pytest.raises(SQLValidationColumnError):
        validate_generated_sql_or_raise(
            "SELECT password FROM public.orders",
            _fake_metadata(),
        )


def test_star_selections_rejected() -> None:
    for sql in (
        "SELECT * FROM public.orders",
        "SELECT o.* FROM public.orders o",
        "SELECT region, * FROM public.orders",
    ):
        result = validate_generated_sql(sql, _fake_metadata())
        assert result.is_valid is False
        assert SQLValidationViolationCode.STAR_SELECTION in _codes(result)
    with pytest.raises(SQLValidationColumnError):
        validate_generated_sql_or_raise(
            "SELECT * FROM public.orders",
            _fake_metadata(),
        )


def test_tableless_and_cte_only_sql_rejected() -> None:
    select_literal = validate_generated_sql("SELECT 1", _fake_metadata())
    assert select_literal.is_valid is False
    assert SQLValidationViolationCode.INSUFFICIENT_SCHEMA in _codes(select_literal)

    no_from = validate_generated_sql("SELECT region", _fake_metadata())
    assert no_from.is_valid is False
    assert SQLValidationViolationCode.INSUFFICIENT_SCHEMA in _codes(no_from)
    assert SQLValidationViolationCode.UNKNOWN_COLUMN in _codes(no_from)

    cte_only = validate_generated_sql(
        "WITH orders AS (SELECT 1 AS region) SELECT region FROM orders",
        _fake_metadata(),
    )
    assert cte_only.is_valid is False
    assert SQLValidationViolationCode.INSUFFICIENT_SCHEMA in _codes(cte_only)


def test_ambiguous_unqualified_column_rejected() -> None:
    orders_id = uuid.uuid4()
    customers_id = uuid.uuid4()
    meta = ResolvedMetadataContext(
        data_source_id=uuid.uuid4(),
        tables=[
            MetadataTableCandidate(
                table_id=orders_id,
                schema_name="public",
                table_name="orders",
                match_reason=MetadataMatchReason.EXACT,
                relevance_score=100,
            ),
            MetadataTableCandidate(
                table_id=customers_id,
                schema_name="public",
                table_name="customers",
                match_reason=MetadataMatchReason.EXACT,
                relevance_score=80,
            ),
        ],
        columns=[
            MetadataColumnCandidate(
                column_id=uuid.uuid4(),
                table_id=orders_id,
                schema_name="public",
                table_name="orders",
                column_name="region",
                data_type="text",
                match_reason=MetadataMatchReason.EXACT,
                relevance_score=90,
            ),
            MetadataColumnCandidate(
                column_id=uuid.uuid4(),
                table_id=customers_id,
                schema_name="public",
                table_name="customers",
                column_name="region",
                data_type="text",
                match_reason=MetadataMatchReason.EXACT,
                relevance_score=90,
            ),
        ],
        relationships=[],
    )
    result = validate_generated_sql(
        "SELECT region FROM public.orders o JOIN public.customers c ON TRUE",
        meta,
    )
    assert result.is_valid is False
    assert SQLValidationViolationCode.AMBIGUOUS_COLUMN in _codes(result)


def test_violation_messages_omit_identifiers() -> None:
    result = validate_generated_sql(
        "SELECT password FROM public.secrets",
        _fake_metadata(),
    )
    assert result.is_valid is False
    for violation in result.violations:
        assert violation.identifier is None
        assert "password" not in violation.message.lower()
        assert "secrets" not in violation.message.lower()
        assert "public" not in violation.message.lower()


def test_aliases_joins_and_subqueries_preserved() -> None:
    meta = _fake_metadata(with_customers_relationship=True)

    aliased = validate_generated_sql(
        "SELECT o.region, o.amount FROM public.orders AS o",
        meta,
    )
    assert aliased.is_valid
    assert aliased.validated is not None
    assert aliased.validated.referenced_tables == ["public.orders"]

    joined = validate_generated_sql(
        "SELECT o.region, c.id "
        "FROM public.orders o "
        "JOIN public.customers c ON o.customer_id = c.id",
        meta,
    )
    assert joined.is_valid
    assert joined.validated is not None
    assert joined.validated.referenced_tables == [
        "public.customers",
        "public.orders",
    ]

    subquery = validate_generated_sql(
        "SELECT region FROM (SELECT region FROM public.orders) AS s",
        meta,
    )
    assert subquery.is_valid
    assert subquery.validated is not None
    assert subquery.validated.referenced_tables == ["public.orders"]

    cte = validate_generated_sql(
        "WITH revenue AS (SELECT region, amount FROM public.orders) "
        "SELECT region FROM revenue",
        meta,
    )
    assert cte.is_valid
    assert cte.validated is not None
    assert cte.validated.referenced_tables == ["public.orders"]


def test_cross_schema_access_rejected() -> None:
    result = validate_generated_sql(
        "SELECT id FROM other.orders",
        _fake_metadata(),
    )
    assert result.is_valid is False
    assert SQLValidationViolationCode.CROSS_SCHEMA in _codes(result)


def test_unauthorized_table_via_join_rejected() -> None:
    result = validate_generated_sql(
        "SELECT o.region FROM public.orders o "
        "JOIN public.customers c ON o.customer_id = c.id",
        _fake_metadata(),
    )
    assert result.is_valid is False
    assert SQLValidationViolationCode.UNKNOWN_TABLE in _codes(result)


def test_empty_metadata_raises_schema_error() -> None:
    with pytest.raises(SQLValidationSchemaError):
        validate_generated_sql("SELECT 1", empty_resolved_context(uuid.uuid4()))


def test_sql_too_long_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AI_SQL_MAX_SQL_CHARS", 20)
    result = validate_generated_sql(
        "SELECT region FROM public.orders WHERE region = 'west'",
        _fake_metadata(),
    )
    assert result.is_valid is False
    assert SQLValidationViolationCode.SQL_TOO_LONG in _codes(result)


def test_extract_resolves_aliases_and_ignores_cte_names() -> None:
    refs = extract_sql_references(
        "WITH cte AS (SELECT region FROM public.orders) "
        "SELECT o.amount FROM public.orders o JOIN cte ON TRUE",
        default_schema="public",
    )
    assert {table.key() for table in refs.tables} == {"public.orders"}
    assert "cte" in refs.cte_names
    assert any(
        column.table_name == "orders" and column.column_name == "amount"
        for column in refs.columns
    )


def test_allowlist_includes_relationship_columns() -> None:
    allowlist = build_schema_allowlist(_fake_metadata(with_customers_relationship=True))
    assert "public.orders" in allowlist.tables
    assert "public.customers" in allowlist.tables
    assert "public.orders.customer_id" in allowlist.columns
    assert "public.customers.id" in allowlist.columns


def test_validation_log_context_omits_sql_text() -> None:
    result = validate_generated_sql(
        "SELECT region FROM public.orders",
        _fake_metadata(),
    )
    context = validation_log_context(result)
    rendered = str(context)
    assert "SELECT" not in rendered
    assert "orders" not in rendered
    assert context["is_valid"] is True
    assert context["has_validated_sql"] is True


def test_map_sql_validation_error_to_ai_errors() -> None:
    assert isinstance(
        map_sql_validation_error(SQLValidationAuthorizationError("denied")),
        AIContextError,
    )
    assert isinstance(
        map_sql_validation_error(SQLValidationReadonlyError("bad")),
        AIRequestValidationError,
    )
    assert isinstance(
        map_sql_validation_error(SQLValidationTableError("table")),
        AIRequestValidationError,
    )


@pytest.mark.parametrize(
    "sql",
    [
        "DeLeTe FrOm public.orders",
        "/* comment */ DELETE FROM public.orders",
        "WITH x AS (SELECT 1) DELETE FROM public.orders",
        "SELECT region FROM public.orders UNION ALL SELECT region FROM pg_catalog.pg_user",
        (
            "SELECT region FROM public.orders WHERE region IN "
            "(SELECT rolname FROM pg_roles)"
        ),
    ],
)
def test_security_bypass_attempts_rejected(sql: str) -> None:
    result = validate_generated_sql(sql, _fake_metadata())
    assert result.is_valid is False


# --- Authorized service ---


def test_sql_validation_service_accepts_authorized_select(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    metadata = _seed_orders_catalog(db_session, data_source)
    service = SQLValidationService(db_session)
    outcome = service.validate(
        _authorized_params(
            user=user,
            workspace=workspace,
            organization=organization,
            data_source=data_source,
            metadata=metadata,
            sql="SELECT region, amount FROM public.orders",
        )
    )
    assert outcome.result.is_valid
    assert outcome.data_source_id == data_source.id


def test_sql_validation_rejects_non_member(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    metadata = _seed_orders_catalog(db_session, data_source)
    service = SQLValidationService(db_session)
    with pytest.raises(SQLValidationAuthorizationError, match="not a member"):
        service.validate(
            _authorized_params(
                user=user,
                workspace=workspace,
                organization=organization,
                data_source=data_source,
                metadata=metadata,
            )
        )


def test_sql_validation_rejects_organization_mismatch(db_session: Session) -> None:
    user, workspace, _organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    metadata = _seed_orders_catalog(db_session, data_source)
    other_org = Organization(name="Other Org", slug=f"other-{uuid.uuid4().hex[:8]}")
    db_session.add(other_org)
    db_session.flush()
    service = SQLValidationService(db_session)
    with pytest.raises(SQLValidationAuthorizationError, match="organization"):
        service.validate(
            SQLValidateParams(
                workspace_id=workspace.id,
                organization_id=other_org.id,
                user_id=user.id,
                data_source_id=data_source.id,
                sql="SELECT region FROM public.orders",
                metadata=metadata,
            )
        )


def test_sql_validation_rejects_foreign_data_source(db_session: Session) -> None:
    user_a, workspace_a, organization_a = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace_a, user=user_a)
    user_b, workspace_b, _ = _seed_workspace(db_session)
    foreign = _seed_data_source(db_session, workspace=workspace_b, user=user_b)
    metadata = _seed_orders_catalog(db_session, foreign)
    service = SQLValidationService(db_session)
    with pytest.raises(SQLValidationAuthorizationError, match="not accessible"):
        service.validate(
            _authorized_params(
                user=user_a,
                workspace=workspace_a,
                organization=organization_a,
                data_source=foreign,
                metadata=metadata,
            )
        )


def test_sql_validation_rejects_invented_metadata_ids(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    service = SQLValidationService(db_session)
    with pytest.raises(SQLValidationAuthorizationError, match="unauthorized"):
        service.validate(
            _authorized_params(
                user=user,
                workspace=workspace,
                organization=organization,
                data_source=data_source,
                metadata=_fake_metadata(data_source.id),
            )
        )


def test_sql_validation_rejects_metadata_data_source_mismatch(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    service = SQLValidationService(db_session)
    with pytest.raises(SQLValidationAuthorizationError, match="does not match"):
        service.validate(
            _authorized_params(
                user=user,
                workspace=workspace,
                organization=organization,
                data_source=data_source,
                metadata=_fake_metadata(uuid.uuid4()),
            )
        )


def test_sql_validation_session_cross_workspace_denied(db_session: Session) -> None:
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
    service = SQLValidationService(db_session)
    with pytest.raises(SQLValidationAuthorizationError, match="not accessible"):
        service.validate(
            _authorized_params(
                user=user_b,
                workspace=workspace_b,
                organization=organization_b,
                data_source=data_source_b,
                metadata=metadata_b,
                session_id=snapshot.session_id,
            )
        )


def test_sql_validation_session_data_source_mismatch(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source_a = _seed_data_source(db_session, workspace=workspace, user=user)
    data_source_b = _seed_data_source(db_session, workspace=workspace, user=user)
    metadata_a = _seed_orders_catalog(db_session, data_source_a)
    _, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source_b,
    )
    service = SQLValidationService(db_session)
    with pytest.raises(SQLValidationAuthorizationError, match="does not match"):
        service.validate(
            _authorized_params(
                user=user,
                workspace=workspace,
                organization=organization,
                data_source=data_source_a,
                metadata=metadata_a,
                session_id=snapshot.session_id,
            )
        )


def test_sql_validation_does_not_execute_or_log_sql(
    db_session: Session, caplog: pytest.LogCaptureFixture
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    metadata = _seed_orders_catalog(db_session, data_source)
    sql = f"SELECT region FROM public.orders -- {SECRET_SNIPPET}"
    service = SQLValidationService(db_session)
    with caplog.at_level(logging.INFO):
        outcome = service.validate(
            _authorized_params(
                user=user,
                workspace=workspace,
                organization=organization,
                data_source=data_source,
                metadata=metadata,
                sql=sql,
            )
        )
    assert outcome.result.is_valid
    joined = " ".join(record.getMessage() for record in caplog.records)
    assert SECRET_SNIPPET not in joined
    assert "SELECT region" not in joined


def test_redactor_still_covers_secrets_in_validation_messages() -> None:
    assert SECRET_SNIPPET not in redact_secret(f"password={SECRET_SNIPPET}")
    result = validate_generated_sql(
        f"SELECT region FROM public.orders -- {SECRET_SNIPPET}",
        _fake_metadata(),
    )
    context = validation_log_context(result)
    assert SECRET_SNIPPET not in str(context)


def test_sql_validation_package_has_no_execution_imports() -> None:
    allowed_connectors = {
        "app.connectors.readonly_sql",
        "app.connectors.exceptions",
    }
    forbidden = {
        "app.mcp",
        "app.connectors.postgresql",
        "app.services.sample_data",
        "app.services.credentials",
        "psycopg",
        "asyncpg",
    }
    for path in _SQL_VALIDATION_DIR.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "execute_query" not in source
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not imported.intersection(forbidden), path
        connector_imports = {
            name for name in imported if name.startswith("app.connectors")
        }
        assert connector_imports <= allowed_connectors, path


def test_sql_validation_rejects_catalog_system_tables() -> None:
    meta = ResolvedMetadataContext(
        data_source_id=uuid.uuid4(),
        tables=[
            MetadataTableCandidate(
                table_id=uuid.uuid4(),
                schema_name="public",
                table_name="orders",
                match_reason=MetadataMatchReason.EXACT,
                relevance_score=100,
            )
        ],
        columns=[
            MetadataColumnCandidate(
                column_id=uuid.uuid4(),
                table_id=uuid.uuid4(),
                schema_name="public",
                table_name="orders",
                column_name="region",
                data_type="text",
                match_reason=MetadataMatchReason.EXACT,
                relevance_score=90,
            )
        ],
        relationships=[],
    )
    result = validate_generated_sql(
        "SELECT table_name FROM information_schema.tables",
        meta,
    )
    assert result.is_valid is False
    assert SQLValidationViolationCode.CROSS_SCHEMA in _codes(result)


def test_unresolved_column_qualifier_rejected() -> None:
    result = validate_generated_sql(
        "SELECT ghost.region FROM public.orders",
        _fake_metadata(),
    )
    assert result.is_valid is False
    assert SQLValidationViolationCode.UNAUTHORIZED_IDENTIFIER in _codes(result)


def test_relationship_metadata_builds_joinable_allowlist() -> None:
    orders_id = uuid.uuid4()
    customers_id = uuid.uuid4()
    meta = ResolvedMetadataContext(
        data_source_id=uuid.uuid4(),
        tables=[
            MetadataTableCandidate(
                table_id=orders_id,
                schema_name="public",
                table_name="orders",
                match_reason=MetadataMatchReason.EXACT,
                relevance_score=100,
            ),
            MetadataTableCandidate(
                table_id=customers_id,
                schema_name="public",
                table_name="customers",
                match_reason=MetadataMatchReason.EXACT,
                relevance_score=80,
            ),
        ],
        columns=[],
        relationships=[
            MetadataRelationshipCandidate(
                relationship_id=uuid.uuid4(),
                source_schema="public",
                source_table="orders",
                source_column="customer_id",
                source_table_id=orders_id,
                source_column_id=uuid.uuid4(),
                target_schema="public",
                target_table="customers",
                target_column="id",
                target_table_id=customers_id,
                target_column_id=uuid.uuid4(),
                relationship_type=DataSourceRelationshipType.MANY_TO_ONE,
            )
        ],
    )
    result = validate_generated_sql(
        "SELECT o.customer_id, c.id FROM public.orders o "
        "JOIN public.customers c ON o.customer_id = c.id",
        meta,
    )
    assert result.is_valid


def test_relationships_do_not_invent_tables() -> None:
    orders_id = uuid.uuid4()
    customers_id = uuid.uuid4()
    meta = ResolvedMetadataContext(
        data_source_id=uuid.uuid4(),
        tables=[
            MetadataTableCandidate(
                table_id=orders_id,
                schema_name="public",
                table_name="orders",
                match_reason=MetadataMatchReason.EXACT,
                relevance_score=100,
            )
        ],
        columns=[],
        relationships=[
            MetadataRelationshipCandidate(
                relationship_id=uuid.uuid4(),
                source_schema="public",
                source_table="orders",
                source_column="customer_id",
                source_table_id=orders_id,
                source_column_id=uuid.uuid4(),
                target_schema="public",
                target_table="customers",
                target_column="id",
                target_table_id=customers_id,
                target_column_id=uuid.uuid4(),
                relationship_type=DataSourceRelationshipType.MANY_TO_ONE,
            )
        ],
    )
    allowlist = build_schema_allowlist(meta)
    assert "public.orders" in allowlist.tables
    assert "public.customers" not in allowlist.tables
    assert "public.orders.customer_id" in allowlist.columns
    assert "public.customers.id" not in allowlist.columns

    result = validate_generated_sql(
        "SELECT o.customer_id FROM public.orders o "
        "JOIN public.customers c ON o.customer_id = c.id",
        meta,
    )
    assert result.is_valid is False
    assert SQLValidationViolationCode.UNKNOWN_TABLE in _codes(result)


# --- Read-only normalization ---


def test_trailing_semicolon_statement_is_normalized_and_valid() -> None:
    result = validate_generated_sql(
        "  SELECT region FROM public.orders;  ",
        _fake_metadata(),
    )
    assert result.is_valid is True
    assert result.validated is not None
    assert result.validated.sql == "SELECT region FROM public.orders"


def test_ensure_readonly_sql_normalizes_or_raises() -> None:
    assert (
        ensure_readonly_sql("  SELECT region FROM public.orders;  ")
        == "SELECT region FROM public.orders"
    )
    with pytest.raises(SQLValidationReadonlyError, match="read-only"):
        ensure_readonly_sql("DELETE FROM public.orders")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT pg_write_file('/tmp/x', 'y', false)",
        "SELECT pg_ls_dir('/etc')",
        "SELECT pg_read_binary_file('/etc/passwd')",
        "SELECT dblink('dbname=x', 'SELECT 1')",
        "SELECT lo_import('/etc/passwd')",
        "SELECT lo_export(1, '/tmp/x')",
        "SELECT set_config('search_path', 'public', false)",
        "SELECT current_setting('data_directory')",
    ],
)
def test_remaining_dangerous_functions_rejected(sql: str) -> None:
    result = validate_generated_sql(sql, _fake_metadata())
    assert result.is_valid is False
    assert SQLValidationViolationCode.DANGEROUS_STATEMENT in _codes(result)


def test_anonymous_code_block_rejected_as_dangerous() -> None:
    result = validate_generated_sql("DO $$ BEGIN END $$", _fake_metadata())
    assert result.is_valid is False
    assert SQLValidationViolationCode.DANGEROUS_STATEMENT in _codes(result)


def test_non_string_sql_is_reported_as_empty() -> None:
    result = validate_generated_sql(cast(str, None), _fake_metadata())
    assert result.is_valid is False
    assert _codes(result) == [SQLValidationViolationCode.EMPTY_SQL]


def test_max_sql_chars_keyword_overrides_settings() -> None:
    result = validate_generated_sql(
        "SELECT region FROM public.orders",
        _fake_metadata(),
        max_sql_chars=5,
    )
    assert result.is_valid is False
    assert _codes(result) == [SQLValidationViolationCode.SQL_TOO_LONG]


# --- Raising wrapper ---


def test_validate_generated_sql_or_raise_returns_validated_sql() -> None:
    validated = validate_generated_sql_or_raise(
        "SELECT region FROM public.orders",
        _fake_metadata(),
    )
    assert validated.sql == "SELECT region FROM public.orders"
    assert validated.referenced_tables == ["public.orders"]


def test_validate_generated_sql_or_raise_maps_remaining_codes() -> None:
    with pytest.raises(SQLValidationSchemaError):
        validate_generated_sql_or_raise("SELECT 1", _fake_metadata())

    with pytest.raises(SQLValidationColumnError):
        validate_generated_sql_or_raise(
            "SELECT unknown.region FROM public.orders",
            _fake_metadata(),
        )

    with pytest.raises(SQLValidationError) as excinfo:
        validate_generated_sql_or_raise("   ", _fake_metadata())
    assert type(excinfo.value) is SQLValidationError


def test_referenced_columns_merge_qualified_and_unqualified_names() -> None:
    result = validate_generated_sql(
        "SELECT o.region, amount FROM public.orders o",
        _fake_metadata(),
    )
    assert result.is_valid is True
    assert result.validated is not None
    assert result.validated.referenced_columns == ["amount", "public.orders.region"]


# --- Reference extraction ---


def test_extract_applies_default_schema_to_unqualified_tables() -> None:
    references = extract_sql_references(
        "SELECT region FROM orders",
        default_schema="analytics",
    )
    assert [table.key() for table in references.tables] == ["analytics.orders"]


def test_validation_default_schema_setting_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "AI_SQL_VALIDATION_DEFAULT_SCHEMA", "analytics")
    drifted = validate_generated_sql("SELECT region FROM orders", _fake_metadata())
    assert drifted.is_valid is False
    assert SQLValidationViolationCode.CROSS_SCHEMA in _codes(drifted)

    overridden = validate_generated_sql(
        "SELECT region FROM orders",
        _fake_metadata(),
        default_schema="public",
    )
    assert overridden.is_valid is True


def test_extract_records_derived_aliases_and_skips_their_columns() -> None:
    references = extract_sql_references(
        "SELECT s.region FROM (SELECT region FROM public.orders) s",
    )
    assert "s" in references.derived_aliases
    assert [table.key() for table in references.tables] == ["public.orders"]
    assert all(column.table_name != "s" for column in references.columns)


def test_extract_resolves_alias_and_real_table_name() -> None:
    result = validate_generated_sql(
        "SELECT public.orders.region, o.amount FROM public.orders o",
        _fake_metadata(),
    )
    assert result.is_valid is True


def test_extract_wraps_unexpected_parser_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_unexpected(*args: object, **kwargs: object) -> object:
        raise RuntimeError("sqlglot exploded")

    monkeypatch.setattr(
        "app.ai.sql_validation.extract.sqlglot.parse_one",
        raise_unexpected,
    )
    with pytest.raises(SQLValidationParseError, match="could not be parsed"):
        extract_sql_references("SELECT region FROM public.orders")


def test_extract_rejects_empty_parse_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.ai.sql_validation.extract.sqlglot.parse_one",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(SQLValidationParseError, match="could not be parsed"):
        extract_sql_references("SELECT region FROM public.orders")


# --- Identifier checks ---


def test_duplicate_unknown_table_violations_are_deduplicated() -> None:
    result = validate_generated_sql(
        "SELECT s.id FROM public.secrets s JOIN public.vault v ON TRUE",
        _fake_metadata(),
    )
    assert result.is_valid is False
    assert _codes(result) == [SQLValidationViolationCode.UNKNOWN_TABLE]


def test_star_selection_and_unknown_qualifier_reported_together() -> None:
    result = validate_generated_sql(
        "SELECT foo.region, * FROM public.orders",
        _fake_metadata(),
    )
    assert result.is_valid is False
    assert set(_codes(result)) == {
        SQLValidationViolationCode.STAR_SELECTION,
        SQLValidationViolationCode.UNAUTHORIZED_IDENTIFIER,
    }


def test_validate_identifiers_without_tables_reports_schema_and_column() -> None:
    allowlist = build_schema_allowlist(_fake_metadata())
    violations = validate_identifiers(
        ExtractedReferences(columns=(QualifiedColumn(None, None, "region"),)),
        allowlist,
    )
    assert [item.code for item in violations] == [
        SQLValidationViolationCode.INSUFFICIENT_SCHEMA,
        SQLValidationViolationCode.UNKNOWN_COLUMN,
    ]


def test_qualified_identifier_keys_are_lowercased() -> None:
    assert QualifiedTable(schema_name="PUBLIC", table_name="Orders").key() == (
        "public.orders"
    )
    assert (
        QualifiedColumn(
            schema_name="Public",
            table_name="Orders",
            column_name="Region",
        ).key()
        == "public.orders.region"
    )
    assert (
        QualifiedColumn(
            schema_name=None, table_name="orders", column_name="region"
        ).key()
        is None
    )


# --- Allowlist construction ---


def test_allowlist_adds_tables_implied_by_columns_and_time_columns() -> None:
    orders_id = uuid.uuid4()
    events_id = uuid.uuid4()
    meta = ResolvedMetadataContext(
        data_source_id=uuid.uuid4(),
        tables=[],
        columns=[
            MetadataColumnCandidate(
                column_id=uuid.uuid4(),
                table_id=orders_id,
                schema_name=" Public ",
                table_name="Orders",
                column_name="Region",
                data_type="text",
                match_reason=MetadataMatchReason.EXACT,
                relevance_score=90,
            ),
            MetadataColumnCandidate(
                column_id=uuid.uuid4(),
                table_id=orders_id,
                schema_name="   ",
                table_name="orders",
                column_name="skipped",
                data_type="text",
                match_reason=MetadataMatchReason.EXACT,
                relevance_score=10,
            ),
        ],
        resolved_time_columns=[
            MetadataColumnCandidate(
                column_id=uuid.uuid4(),
                table_id=events_id,
                schema_name="public",
                table_name="events",
                column_name="occurred_at",
                data_type="timestamptz",
                match_reason=MetadataMatchReason.EXACT,
                relevance_score=70,
            )
        ],
    )
    allowlist = build_schema_allowlist(meta)
    assert allowlist.tables == frozenset({"public.orders", "public.events"})
    assert "public.orders.region" in allowlist.columns
    assert "public.events.occurred_at" in allowlist.columns
    assert "skipped" not in allowlist.column_names
    assert allowlist.column_names == frozenset({"region", "occurred_at"})


def test_has_usable_allowlist_requires_tables_or_columns() -> None:
    assert has_usable_allowlist(build_schema_allowlist(_fake_metadata())) is True
    empty = build_schema_allowlist(empty_resolved_context(uuid.uuid4()))
    assert has_usable_allowlist(empty) is False
    columns_only = build_schema_allowlist(
        _fake_metadata().model_copy(update={"tables": []})
    )
    assert has_usable_allowlist(columns_only) is True


# --- Observability ---


def test_validation_log_context_for_invalid_result_caps_codes() -> None:
    violations = [
        SQLValidationViolation(
            code=SQLValidationViolationCode.UNKNOWN_COLUMN,
            message=f"violation {index}",
        )
        for index in range(12)
    ]
    context = validation_log_context(
        SQLValidationResult(is_valid=False, violations=violations)
    )
    assert context["violation_count"] == 12
    assert len(cast(list[str], context["violation_codes"])) == 10
    assert context["has_validated_sql"] is False
    assert context["sql_char_count"] is None


def test_map_sql_validation_error_covers_remaining_arms() -> None:
    assert isinstance(
        map_sql_validation_error(SQLValidationParseError()),
        AIRequestValidationError,
    )
    assert isinstance(
        map_sql_validation_error(SQLValidationSchemaError()),
        AIRequestValidationError,
    )
    assert isinstance(
        map_sql_validation_error(SQLValidationColumnError()),
        AIRequestValidationError,
    )
    assert isinstance(
        map_sql_validation_error(SQLValidationError("boom")),
        AIRequestValidationError,
    )


def test_sql_validation_errors_expose_codes_and_redact_secrets() -> None:
    assert (
        SQLValidationAuthorizationError().code
        is SQLValidationErrorCode.SQL_VALIDATION_AUTHORIZATION_ERROR
    )
    assert (
        SQLValidationReadonlyError().code
        is SQLValidationErrorCode.SQL_VALIDATION_READONLY_ERROR
    )
    assert (
        SQLValidationParseError().code
        is SQLValidationErrorCode.SQL_VALIDATION_PARSE_ERROR
    )
    assert (
        SQLValidationSchemaError().code
        is SQLValidationErrorCode.SQL_VALIDATION_SCHEMA_ERROR
    )
    assert (
        SQLValidationTableError().code
        is SQLValidationErrorCode.SQL_VALIDATION_TABLE_ERROR
    )
    assert (
        SQLValidationColumnError().code
        is SQLValidationErrorCode.SQL_VALIDATION_COLUMN_ERROR
    )
    assert (
        SQLValidationConfigurationError().code
        is SQLValidationErrorCode.SQL_VALIDATION_CONFIGURATION_ERROR
    )
    assert (
        SQLValidationError("boom").code
        is SQLValidationErrorCode.SQL_VALIDATION_INTERNAL_ERROR
    )
    message = str(SQLValidationReadonlyError(f"password={SECRET_SNIPPET}"))
    assert "CustomerDbPassword" not in message
    assert "[REDACTED]" in message


# --- Service authorization ---


def test_sql_validation_allows_super_admin_without_membership(
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
    result = SQLValidationService(db_session).validate(params)
    assert result.result.is_valid is True
    with pytest.raises(SQLValidationAuthorizationError, match="not a member"):
        SQLValidationService(db_session, allow_super_admin=False).validate(params)


def test_sql_validation_rejects_member_without_required_permission(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user, role=WorkspaceRole.MEMBER)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    metadata = _seed_orders_catalog(db_session, data_source)
    service = SQLValidationService(
        db_session,
        required_permission=WorkspacePermission.DATA_SOURCE_QUERY,
    )
    with pytest.raises(SQLValidationAuthorizationError, match="lacks permission"):
        service.validate(
            _authorized_params(
                user=user,
                workspace=workspace,
                organization=organization,
                data_source=data_source,
                metadata=metadata,
            )
        )


def test_sql_validation_rejects_inactive_user(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    metadata = _seed_orders_catalog(db_session, data_source)
    user.is_active = False
    db_session.flush()
    with pytest.raises(SQLValidationAuthorizationError, match="User is not authorized"):
        SQLValidationService(db_session).validate(
            _authorized_params(
                user=user,
                workspace=workspace,
                organization=organization,
                data_source=data_source,
                metadata=metadata,
            )
        )


def test_sql_validation_session_scoped_request_succeeds(db_session: Session) -> None:
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
    result = SQLValidationService(db_session).validate(
        _authorized_params(
            user=user,
            workspace=workspace,
            organization=organization,
            data_source=data_source,
            metadata=metadata,
            session_id=snapshot.session_id,
        )
    )
    assert result.result.is_valid is True


def test_sql_validation_session_without_data_source_denied(
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
    with pytest.raises(
        SQLValidationAuthorizationError,
        match="no authorized data source",
    ):
        SQLValidationService(db_session).validate(
            _authorized_params(
                user=user,
                workspace=workspace,
                organization=organization,
                data_source=data_source,
                metadata=metadata,
                session_id=snapshot.session_id,
            )
        )


def test_sql_validation_session_organization_drift_denied(
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
    service = SQLValidationService(
        db_session,
        state_service=cast(AgentStateService, _FixedSnapshotStateService(drifted)),
    )
    with pytest.raises(
        SQLValidationAuthorizationError,
        match="organization does not match",
    ):
        service.validate(
            _authorized_params(
                user=user,
                workspace=workspace,
                organization=organization,
                data_source=data_source,
                metadata=metadata,
                session_id=snapshot.session_id,
            )
        )


def test_sql_validation_service_returns_violations_without_raising(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    metadata = _seed_orders_catalog(db_session, data_source)
    result = SQLValidationService(db_session).validate(
        _authorized_params(
            user=user,
            workspace=workspace,
            organization=organization,
            data_source=data_source,
            metadata=metadata,
            sql="DROP TABLE public.orders",
        )
    )
    assert result.result.is_valid is False
    assert result.result.validated is None
    assert SQLValidationViolationCode.DANGEROUS_STATEMENT in _codes(result.result)
    assert result.data_source_id == data_source.id


# --- Output aliases, CTEs, window functions (chat analysis regressions) ---


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT region, SUM(amount) AS total FROM public.orders GROUP BY region ORDER BY total DESC",
        "SELECT COUNT(*) AS n FROM public.orders",
        "WITH r AS (SELECT region, SUM(amount) AS total FROM public.orders GROUP BY region) "
        "SELECT region, total, total - LAG(total) OVER (ORDER BY region) AS delta FROM r ORDER BY delta",
        "SELECT t.region, t.total FROM (SELECT region, SUM(amount) AS total "
        "FROM public.orders GROUP BY region) t WHERE t.total > 10",
        "WITH r AS (SELECT region, SUM(amount) AS total FROM public.orders GROUP BY region) SELECT * FROM r",
    ],
)
def test_output_aliases_ctes_and_windows_are_not_catalog_columns(sql: str) -> None:
    result = validate_generated_sql(sql, _fake_metadata())
    assert result.is_valid is True, result.violations


@pytest.mark.parametrize(
    ("sql", "code", "identifier"),
    [
        ("SELECT discount FROM public.orders", SQLValidationViolationCode.UNKNOWN_COLUMN, "discount"),
        # SELECT aliases are not visible in WHERE of the same query.
        ("SELECT amount AS a FROM public.orders WHERE a > 5", SQLValidationViolationCode.UNKNOWN_COLUMN, "a"),
        # Physical columns inside a CTE are still checked.
        (
            "WITH x AS (SELECT salary AS s FROM public.orders) SELECT s FROM x",
            SQLValidationViolationCode.UNKNOWN_COLUMN,
            "salary",
        ),
        ("SELECT * FROM public.orders", SQLValidationViolationCode.STAR_SELECTION, None),
    ],
)
def test_alias_support_does_not_weaken_allowlist(sql, code, identifier) -> None:
    result = validate_generated_sql(sql, _fake_metadata())
    assert result.is_valid is False
    matching = [v for v in result.violations if v.code is code]
    assert matching
    assert matching[0].identifier == identifier
