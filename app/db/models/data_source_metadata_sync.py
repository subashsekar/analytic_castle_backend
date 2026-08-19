import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base
from app.enums import MetadataSyncStatus

if TYPE_CHECKING:
    from app.db.models.data_source import DataSource


class DataSourceMetadataSync(Base):
    """Current metadata-synchronization state for one data source.

    One row per data source. Counts reflect the last successful sync and are
    left unchanged when a later attempt fails.
    """

    __tablename__ = "data_source_metadata_syncs"
    __table_args__ = (
        UniqueConstraint(
            "data_source_id",
            name="uq_data_source_metadata_syncs_data_source_id",
        ),
        CheckConstraint(
            "schema_count IS NULL OR schema_count >= 0",
            name="ck_data_source_metadata_syncs_schema_count_non_negative",
        ),
        CheckConstraint(
            "table_count IS NULL OR table_count >= 0",
            name="ck_data_source_metadata_syncs_table_count_non_negative",
        ),
        CheckConstraint(
            "column_count IS NULL OR column_count >= 0",
            name="ck_data_source_metadata_syncs_column_count_non_negative",
        ),
        CheckConstraint(
            "relationship_count IS NULL OR relationship_count >= 0",
            name="ck_data_source_metadata_syncs_relationship_count_non_negative",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    data_source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("data_sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[MetadataSyncStatus] = mapped_column(
        Enum(MetadataSyncStatus, name="metadata_sync_status", native_enum=True),
        nullable=False,
        default=MetadataSyncStatus.PENDING,
        server_default=MetadataSyncStatus.PENDING.value,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    schema_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    table_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    column_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    relationship_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
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

    data_source: Mapped["DataSource"] = relationship(back_populates="metadata_sync")

    def __repr__(self) -> str:
        return (
            f"DataSourceMetadataSync(id={self.id!r}, "
            f"data_source_id={self.data_source_id!r}, status={self.status!r})"
        )
