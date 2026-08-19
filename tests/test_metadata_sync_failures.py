from __future__ import annotations

import asyncio
import logging
import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.connectors import ConnectorQueryError
from app.core.logging import RedactingFilter
from app.db.models import (
    DataSourceMetadataSync,
    DataSourceSchema,
    DataSourceTable,
)
from app.enums import DataSourceTableType, MetadataSyncStatus
from app.services.discovery_exceptions import MetadataDiscoveryError
from app.services.metadata_persist import validate_discovery_result
from app.services.metadata_sync import (
    MetadataSyncService,
    release_metadata_sync_lock,
    try_acquire_metadata_sync_lock,
)
from app.services.metadata_sync_exceptions import (
    ConcurrentMetadataSyncError,
    MetadataSyncError,
    MetadataSyncPersistenceError,
    MetadataSyncValidationError,
)
from tests.conftest import run_async
from tests.test_metadata_sync import (
    CUSTOMER_PASSWORD,
    StubDiscovery,
    _column,
    _result,
    _schema_names,
    _seed_data_source,
    _sync,
)

CONNECTION_URI = (
    f"postgresql://readonly:{CUSTOMER_PASSWORD}@db.internal.example:5432/analytics"
)


class DiscoveryFailure(StubDiscovery):
    async def discover(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        raise MetadataDiscoveryError(
            f"password={CUSTOMER_PASSWORD} connection_string={CONNECTION_URI}"
        )


class InvalidDiscovery(StubDiscovery):
    def __init__(self) -> None:
        super().__init__(_result())
        duplicate = _result().columns[0]
        self.result = _result(
            columns=(duplicate, duplicate),
            relationships=(),
        )


class SlowDiscovery(StubDiscovery):
    async def discover(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        await asyncio.sleep(0.2)
        self.calls += 1
        assert self.result is not None
        return self.result


def test_discovery_failure_preserves_existing_metadata(db_session: Session) -> None:
    data_source = _seed_data_source(db_session)
    _sync(db_session, data_source, _result())
    schema_count_before = db_session.scalar(
        select(func.count()).select_from(DataSourceSchema)
    )
    table_count_before = db_session.scalar(
        select(func.count()).select_from(DataSourceTable)
    )

    service = MetadataSyncService(db_session, discovery_service=DiscoveryFailure())
    with pytest.raises(MetadataSyncError, match="Unable to discover database metadata"):
        run_async(
            service.synchronize(data_source.id, workspace_id=data_source.workspace_id)
        )

    status = service.get_status(data_source.id, workspace_id=data_source.workspace_id)
    assert status.status is MetadataSyncStatus.FAILED
    assert status.error_message == "Unable to discover database metadata"
    assert status.schema_count == 1
    assert (
        db_session.scalar(select(func.count()).select_from(DataSourceSchema))
        == schema_count_before
    )
    assert (
        db_session.scalar(select(func.count()).select_from(DataSourceTable))
        == table_count_before
    )


def test_persistence_failure_rolls_back_metadata(db_session: Session) -> None:
    data_source = _seed_data_source(db_session)
    _sync(db_session, data_source, _result())
    schema_count_before = db_session.scalar(
        select(func.count()).select_from(DataSourceSchema)
    )
    service = MetadataSyncService(
        db_session,
        discovery_service=StubDiscovery(_result(schemas=("analytics", "public"))),
    )

    with (
        patch(
            "app.services.metadata_sync.persist_discovered_metadata",
            side_effect=SQLAlchemyError("boom"),
        ),
        pytest.raises(MetadataSyncPersistenceError, match="Unable to save metadata"),
    ):
        run_async(
            service.synchronize(data_source.id, workspace_id=data_source.workspace_id)
        )

    status = service.get_status(data_source.id, workspace_id=data_source.workspace_id)
    assert status.status is MetadataSyncStatus.FAILED
    assert status.error_message == "Unable to save metadata"
    assert _schema_names(db_session, data_source.id) == ["public"]
    assert (
        db_session.scalar(select(func.count()).select_from(DataSourceSchema))
        == schema_count_before
    )


def test_invalid_discovery_result_fails_without_persisting(db_session: Session) -> None:
    data_source = _seed_data_source(db_session)
    service = MetadataSyncService(db_session, discovery_service=InvalidDiscovery())

    with pytest.raises(MetadataSyncValidationError, match="invalid"):
        run_async(
            service.synchronize(data_source.id, workspace_id=data_source.workspace_id)
        )

    assert db_session.scalar(select(func.count()).select_from(DataSourceSchema)) == 0
    status = service.get_status(data_source.id, workspace_id=data_source.workspace_id)
    assert status.status is MetadataSyncStatus.FAILED


def test_connection_failure_sets_failed_status(db_session: Session) -> None:
    data_source = _seed_data_source(db_session)
    service = MetadataSyncService(
        db_session,
        discovery_service=StubDiscovery(
            _result(),
            fail=ConnectorQueryError(CONNECTION_URI),
        ),
    )

    with pytest.raises(MetadataSyncError, match="Unable to connect"):
        run_async(
            service.synchronize(data_source.id, workspace_id=data_source.workspace_id)
        )

    status = service.get_status(data_source.id, workspace_id=data_source.workspace_id)
    assert status.status is MetadataSyncStatus.FAILED
    assert status.error_message == "Unable to connect to the data source"
    assert db_session.scalar(select(func.count()).select_from(DataSourceSchema)) == 0


def test_sync_errors_and_logs_omit_credentials(
    db_session: Session,
    caplog: pytest.LogCaptureFixture,
) -> None:
    data_source = _seed_data_source(db_session)
    assert data_source.connection is not None
    ciphertext = data_source.connection.encrypted_password
    logger = logging.getLogger("app.services.metadata_sync")
    logger.addFilter(RedactingFilter())
    service = MetadataSyncService(db_session, discovery_service=DiscoveryFailure())

    with (
        caplog.at_level(logging.WARNING, logger=logger.name),
        pytest.raises(MetadataSyncError),
    ):
        run_async(
            service.synchronize(data_source.id, workspace_id=data_source.workspace_id)
        )

    text = caplog.text
    assert CUSTOMER_PASSWORD not in text
    assert CONNECTION_URI not in text
    assert ciphertext not in text
    assert "Metadata sync failed" in text
    assert str(data_source.id) in text


def test_concurrent_sync_rejected_for_same_data_source(db_session: Session) -> None:
    data_source = _seed_data_source(db_session)
    service = MetadataSyncService(
        db_session, discovery_service=SlowDiscovery(_result())
    )
    gate = asyncio.Event()

    async def _run() -> None:
        gate.set()
        await service.synchronize(data_source.id, workspace_id=data_source.workspace_id)

    async def _exercise() -> None:
        first = asyncio.create_task(_run())
        await gate.wait()
        with pytest.raises(ConcurrentMetadataSyncError):
            await service.synchronize(
                data_source.id, workspace_id=data_source.workspace_id
            )
        await first

    run_async(_exercise())


def test_lock_released_after_failure_allows_retry(db_session: Session) -> None:
    data_source = _seed_data_source(db_session)
    failing = MetadataSyncService(db_session, discovery_service=DiscoveryFailure())
    with pytest.raises(MetadataSyncError):
        run_async(
            failing.synchronize(data_source.id, workspace_id=data_source.workspace_id)
        )

    success = MetadataSyncService(
        db_session, discovery_service=StubDiscovery(_result())
    )
    outcome = run_async(
        success.synchronize(data_source.id, workspace_id=data_source.workspace_id)
    )
    assert outcome.status is MetadataSyncStatus.SUCCESS


def test_lock_released_after_cancellation_allows_retry(db_session: Session) -> None:
    data_source = _seed_data_source(db_session)
    service = MetadataSyncService(
        db_session, discovery_service=SlowDiscovery(_result())
    )

    async def _exercise() -> None:
        task = asyncio.create_task(
            service.synchronize(data_source.id, workspace_id=data_source.workspace_id)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run_async(_exercise())

    outcome = run_async(
        service.synchronize(data_source.id, workspace_id=data_source.workspace_id)
    )
    assert outcome.status is MetadataSyncStatus.SUCCESS


def test_try_acquire_and_release_lock_helpers() -> None:
    data_source_id = uuid.uuid4()
    assert try_acquire_metadata_sync_lock(data_source_id) is True
    assert try_acquire_metadata_sync_lock(data_source_id) is False
    release_metadata_sync_lock(data_source_id)
    assert try_acquire_metadata_sync_lock(data_source_id) is True
    release_metadata_sync_lock(data_source_id)


def test_validate_discovery_result_rejects_orphan_column() -> None:
    with pytest.raises(MetadataSyncValidationError):
        validate_discovery_result(
            _result(
                tables=(("public", "users", DataSourceTableType.TABLE),),
                columns=(
                    _column("public", "users", "id", position=1, primary_key=True),
                    _column("public", "orders", "id", position=1, primary_key=True),
                ),
                relationships=(),
            )
        )


def test_running_status_recorded_before_discovery(db_session: Session) -> None:
    data_source = _seed_data_source(db_session)
    seen: list[MetadataSyncStatus] = []

    class StatusProbe(StubDiscovery):
        async def discover(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            row = db_session.scalar(
                select(DataSourceMetadataSync).where(
                    DataSourceMetadataSync.data_source_id == data_source.id
                )
            )
            if row is not None:
                seen.append(row.status)
            return await super().discover(*args, **kwargs)

    service = MetadataSyncService(db_session, discovery_service=StatusProbe(_result()))
    run_async(
        service.synchronize(data_source.id, workspace_id=data_source.workspace_id)
    )
    assert seen == [MetadataSyncStatus.RUNNING]
