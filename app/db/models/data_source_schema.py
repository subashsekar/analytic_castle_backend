import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.db.session import Base

if TYPE_CHECKING:
    from app.db.models.data_source import DataSource
    from app.db.models.data_source_table import DataSourceTable


class DataSourceSchema(Base):
    __tablename__ = "data_source_schemas"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    data_source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("data_sources.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
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

    __table_args__ = (
        UniqueConstraint(
            "data_source_id",
            "name",
            name="uq_data_source_schemas_data_source_id_name",
        ),
        CheckConstraint(
            "length(btrim(name)) > 0",
            name="ck_data_source_schemas_name_not_empty",
        ),
        Index(
            "ix_data_source_schemas_data_source_id_lower_name",
            "data_source_id",
            func.lower(name),
        ),
    )

    data_source: Mapped["DataSource"] = relationship(back_populates="schemas")
    tables: Mapped[list["DataSourceTable"]] = relationship(
        back_populates="schema",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    @validates("name")
    def _normalize_name(self, _key: str, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("name cannot be empty")
        return stripped

    def __repr__(self) -> str:
        return f"DataSourceSchema(id={self.id!r}, name={self.name!r})"
