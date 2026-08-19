"""Create, list, update, and delete data sources.

Encrypts credentials through the credential service before persistence.
Does not test external connections, decrypt passwords, or implement HTTP
or RBAC checks.
"""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.connectors import UnsupportedConnectorError, is_implemented
from app.db.models import DataSource, DataSourceConnection, WorkspaceMember
from app.enums import DataSourceStatus, DataSourceType
from app.services.credentials import encrypt_secret

logger = logging.getLogger(__name__)


class DataSourceWriteError(Exception):
    """A data source could not be written."""

    def __init__(self, message: str = "Unable to save data source") -> None:
        super().__init__(message)


class DataSourceService:
    """Persist data sources and their encrypted connection settings."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        created_by: UUID,
        workspace_id: UUID,
        name: str,
        source_type: DataSourceType,
        host: str,
        port: int,
        database_name: str,
        username: str,
        password: str,
        ssl_mode: str,
    ) -> DataSource:
        if not is_implemented(source_type):
            raise UnsupportedConnectorError(
                f"Connector type {source_type.value} is not implemented"
            )

        encrypted_password = encrypt_secret(password)

        data_source = DataSource(
            workspace_id=workspace_id,
            name=name,
            type=source_type,
            status=DataSourceStatus.INACTIVE,
            created_by=created_by,
            connection=DataSourceConnection(
                host=host,
                port=port,
                database_name=database_name,
                username=username,
                encrypted_password=encrypted_password,
                ssl_mode=ssl_mode,
            ),
        )
        self._session.add(data_source)
        try:
            self._session.commit()
        except IntegrityError:
            self._session.rollback()
            raise DataSourceWriteError("Unable to save data source") from None

        self._session.refresh(data_source)
        logger.info(
            "Created data source data_source_id=%s workspace_id=%s type=%s",
            data_source.id,
            workspace_id,
            source_type.value,
        )
        return data_source

    def list_accessible(
        self,
        *,
        workspace_id: UUID | None = None,
        member_user_id: UUID | None = None,
    ) -> list[DataSource]:
        stmt = select(DataSource).order_by(DataSource.created_at.desc())
        if workspace_id is not None:
            stmt = stmt.where(DataSource.workspace_id == workspace_id)
        if member_user_id is not None:
            stmt = stmt.join(
                WorkspaceMember,
                (WorkspaceMember.workspace_id == DataSource.workspace_id)
                & (WorkspaceMember.user_id == member_user_id),
            )
        return list(self._session.scalars(stmt).unique().all())

    def update(self, data_source: DataSource, *, name: str | None) -> DataSource:
        if name is not None:
            data_source.name = name
        try:
            self._session.commit()
        except IntegrityError:
            self._session.rollback()
            raise DataSourceWriteError("Unable to save data source") from None
        self._session.refresh(data_source)
        return data_source

    def delete(self, data_source: DataSource) -> None:
        self._session.delete(data_source)
        try:
            self._session.commit()
        except IntegrityError:
            self._session.rollback()
            raise DataSourceWriteError("Unable to delete data source") from None
        logger.info("Deleted data source data_source_id=%s", data_source.id)
