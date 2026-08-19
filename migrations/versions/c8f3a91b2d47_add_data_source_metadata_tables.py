"""add data source metadata tables

Revision ID: c8f3a91b2d47
Revises: ae4daa14818c
Create Date: 2026-08-17 11:55:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c8f3a91b2d47"
down_revision: Union[str, Sequence[str], None] = "ae4daa14818c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "data_source_schemas",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("data_source_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "length(btrim(name)) > 0",
            name="ck_data_source_schemas_name_not_empty",
        ),
        sa.ForeignKeyConstraint(
            ["data_source_id"], ["data_sources.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "data_source_id",
            "name",
            name="uq_data_source_schemas_data_source_id_name",
        ),
    )
    op.create_index(
        op.f("ix_data_source_schemas_data_source_id"),
        "data_source_schemas",
        ["data_source_id"],
        unique=False,
    )
    op.create_table(
        "data_source_tables",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("schema_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column(
            "table_type",
            sa.Enum("TABLE", "VIEW", name="data_source_table_type"),
            server_default="TABLE",
            nullable=False,
        ),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "length(btrim(name)) > 0",
            name="ck_data_source_tables_name_not_empty",
        ),
        sa.ForeignKeyConstraint(
            ["schema_id"], ["data_source_schemas.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "schema_id",
            "name",
            name="uq_data_source_tables_schema_id_name",
        ),
    )
    op.create_index(
        op.f("ix_data_source_tables_schema_id"),
        "data_source_tables",
        ["schema_id"],
        unique=False,
    )
    op.create_table(
        "data_source_columns",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("table_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("ordinal_position", sa.Integer(), nullable=False),
        sa.Column("data_type", sa.String(length=255), nullable=False),
        sa.Column("database_type", sa.String(length=255), nullable=False),
        sa.Column("is_nullable", sa.Boolean(), nullable=False),
        sa.Column("default_value", sa.Text(), nullable=True),
        sa.Column(
            "is_primary_key",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "is_unique",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "length(btrim(name)) > 0",
            name="ck_data_source_columns_name_not_empty",
        ),
        sa.CheckConstraint(
            "ordinal_position > 0",
            name="ck_data_source_columns_ordinal_position_positive",
        ),
        sa.ForeignKeyConstraint(
            ["table_id"], ["data_source_tables.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "id",
            "table_id",
            name="uq_data_source_columns_id_table_id",
        ),
        sa.UniqueConstraint(
            "table_id",
            "name",
            name="uq_data_source_columns_table_id_name",
        ),
    )
    op.create_index(
        op.f("ix_data_source_columns_table_id"),
        "data_source_columns",
        ["table_id"],
        unique=False,
    )
    op.create_index(
        "ix_data_source_columns_table_id_ordinal_position",
        "data_source_columns",
        ["table_id", "ordinal_position"],
        unique=False,
    )
    op.create_table(
        "data_source_relationships",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_table_id", sa.Uuid(), nullable=False),
        sa.Column("source_column_id", sa.Uuid(), nullable=False),
        sa.Column("target_table_id", sa.Uuid(), nullable=False),
        sa.Column("target_column_id", sa.Uuid(), nullable=False),
        sa.Column(
            "relationship_type",
            sa.Enum(
                "ONE_TO_ONE",
                "ONE_TO_MANY",
                "MANY_TO_ONE",
                "MANY_TO_MANY",
                name="data_source_relationship_type",
            ),
            nullable=False,
        ),
        sa.Column("constraint_name", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["source_column_id", "source_table_id"],
            ["data_source_columns.id", "data_source_columns.table_id"],
            ondelete="CASCADE",
            name="fk_data_source_relationships_source_column_table",
        ),
        sa.ForeignKeyConstraint(
            ["source_column_id"],
            ["data_source_columns.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_table_id"],
            ["data_source_tables.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["target_column_id", "target_table_id"],
            ["data_source_columns.id", "data_source_columns.table_id"],
            ondelete="CASCADE",
            name="fk_data_source_relationships_target_column_table",
        ),
        sa.ForeignKeyConstraint(
            ["target_column_id"],
            ["data_source_columns.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["target_table_id"],
            ["data_source_tables.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_data_source_relationships_source_table_id"),
        "data_source_relationships",
        ["source_table_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_data_source_relationships_target_table_id"),
        "data_source_relationships",
        ["target_table_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_data_source_relationships_target_table_id"),
        table_name="data_source_relationships",
    )
    op.drop_index(
        op.f("ix_data_source_relationships_source_table_id"),
        table_name="data_source_relationships",
    )
    op.drop_table("data_source_relationships")
    op.drop_index(
        "ix_data_source_columns_table_id_ordinal_position",
        table_name="data_source_columns",
    )
    op.drop_index(
        op.f("ix_data_source_columns_table_id"),
        table_name="data_source_columns",
    )
    op.drop_table("data_source_columns")
    op.drop_index(
        op.f("ix_data_source_tables_schema_id"),
        table_name="data_source_tables",
    )
    op.drop_table("data_source_tables")
    op.drop_index(
        op.f("ix_data_source_schemas_data_source_id"),
        table_name="data_source_schemas",
    )
    op.drop_table("data_source_schemas")
    sa.Enum(name="data_source_relationship_type").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="data_source_table_type").drop(op.get_bind(), checkfirst=True)
