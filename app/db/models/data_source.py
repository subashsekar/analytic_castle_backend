import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, Enum, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.db.session import Base
from app.enums import DataSourceStatus, DataSourceType

if TYPE_CHECKING:
    from app.db.models.data_source_connection import DataSourceConnection
    from app.db.models.data_source_metadata_sync import DataSourceMetadataSync
    from app.db.models.data_source_schema import DataSourceSchema
    from app.db.models.user import User
    from app.db.models.workspace import Workspace


class DataSource(Base):
    __tablename__ = "data_sources"
    __table_args__ = (
        CheckConstraint(
            "length(btrim(name)) > 0",
            name="ck_data_sources_name_not_empty",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[DataSourceType] = mapped_column(
        Enum(DataSourceType, name="data_source_type", native_enum=True),
        nullable=False,
    )
    status: Mapped[DataSourceStatus] = mapped_column(
        Enum(DataSourceStatus, name="data_source_status", native_enum=True),
        nullable=False,
        default=DataSourceStatus.INACTIVE,
        server_default=DataSourceStatus.INACTIVE.value,
    )
    created_by: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id"),
        index=True,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )  # ORM updates only; raw SQL will not bump this.
    last_tested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    workspace: Mapped["Workspace"] = relationship(back_populates="data_sources")
    created_by_user: Mapped["User"] = relationship(
        back_populates="created_data_sources",
        foreign_keys=[created_by],
    )
    connection: Mapped["DataSourceConnection | None"] = relationship(
        back_populates="data_source",
        cascade="all, delete-orphan",
        uselist=False,
        passive_deletes=True,
    )
    schemas: Mapped[list["DataSourceSchema"]] = relationship(
        back_populates="data_source",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    metadata_sync: Mapped["DataSourceMetadataSync | None"] = relationship(
        back_populates="data_source",
        cascade="all, delete-orphan",
        uselist=False,
        passive_deletes=True,
    )

    @validates("name")
    def _normalize_name(self, _key: str, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("name cannot be empty")
        return stripped

    def __repr__(self) -> str:
        return f"DataSource(id={self.id!r}, name={self.name!r}, type={self.type!r})"
