import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.db.session import Base
from app.enums import DataSourceTableType

if TYPE_CHECKING:
    from app.db.models.data_source_column import DataSourceColumn
    from app.db.models.data_source_relationship import DataSourceRelationship
    from app.db.models.data_source_schema import DataSourceSchema


class DataSourceTable(Base):
    __tablename__ = "data_source_tables"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    schema_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("data_source_schemas.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    table_type: Mapped[DataSourceTableType] = mapped_column(
        Enum(DataSourceTableType, name="data_source_table_type", native_enum=True),
        nullable=False,
        default=DataSourceTableType.TABLE,
        server_default=DataSourceTableType.TABLE.value,
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
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
            "schema_id",
            "name",
            name="uq_data_source_tables_schema_id_name",
        ),
        CheckConstraint(
            "length(btrim(name)) > 0",
            name="ck_data_source_tables_name_not_empty",
        ),
        Index(
            "ix_data_source_tables_schema_id_lower_name",
            "schema_id",
            func.lower(name),
        ),
    )

    schema: Mapped["DataSourceSchema"] = relationship(back_populates="tables")
    columns: Mapped[list["DataSourceColumn"]] = relationship(
        back_populates="table",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    source_relationships: Mapped[list["DataSourceRelationship"]] = relationship(
        back_populates="source_table",
        foreign_keys="DataSourceRelationship.source_table_id",
        passive_deletes=True,
    )
    target_relationships: Mapped[list["DataSourceRelationship"]] = relationship(
        back_populates="target_table",
        foreign_keys="DataSourceRelationship.target_table_id",
        passive_deletes=True,
    )

    @validates("name")
    def _normalize_name(self, _key: str, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("name cannot be empty")
        return stripped

    def __repr__(self) -> str:
        return (
            f"DataSourceTable(id={self.id!r}, name={self.name!r}, "
            f"table_type={self.table_type!r})"
        )
