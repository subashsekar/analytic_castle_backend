"""Parse SQL and extract physical table/column references via sqlglot.

Column references are resolved per query scope so that names produced inside
the statement (SELECT aliases, CTE outputs, derived-table outputs) are not
mistaken for catalog columns, while every physical column is still reported
for allowlist checks.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError as SQLGlotParseError
from sqlglot.optimizer.scope import Scope, traverse_scope

from app.ai.sql_validation.errors import SQLValidationParseError
from app.ai.sql_validation.models import (
    ExtractedReferences,
    QualifiedColumn,
    QualifiedTable,
)


def extract_sql_references(
    sql: str,
    *,
    default_schema: str = "public",
) -> ExtractedReferences:
    """Extract catalog table/column refs from a single PostgreSQL statement."""
    try:
        expression = sqlglot.parse_one(sql, read="postgres")
    except SQLGlotParseError as exc:
        raise SQLValidationParseError("SQL could not be parsed") from exc
    except Exception as exc:
        # Unexpected parse failures are treated as invalid SQL.
        raise SQLValidationParseError("SQL could not be parsed") from exc

    if expression is None:
        raise SQLValidationParseError("SQL could not be parsed")

    cte_names = frozenset(
        alias.lower() for cte in expression.find_all(exp.CTE) if (alias := cte.alias)
    )
    derived_aliases = frozenset(
        alias.lower()
        for subquery in expression.find_all(exp.Subquery)
        if (alias := subquery.alias_or_name)
    )
    non_catalog = cte_names | derived_aliases

    alias_to_table: dict[str, QualifiedTable] = {}
    tables: list[QualifiedTable] = []
    seen_tables: set[str] = set()

    for table in expression.find_all(exp.Table):
        qualified = _physical_table(table, non_catalog, default_schema)
        if qualified is None:
            continue
        key = qualified.key()
        if key not in seen_tables:
            tables.append(qualified)
            seen_tables.add(key)
        alias = (table.alias or "").strip().lower()
        if alias:
            alias_to_table[alias] = qualified
        alias_to_table[qualified.table_name] = qualified

    collector = _ColumnCollector()
    scoped = _collect_scoped_columns(
        expression,
        collector=collector,
        non_catalog=non_catalog,
        alias_to_table=alias_to_table,
        default_schema=default_schema,
    )
    if not scoped:
        _collect_flat_columns(
            expression,
            collector=collector,
            non_catalog=non_catalog,
            alias_to_table=alias_to_table,
        )

    return ExtractedReferences(
        tables=tuple(tables),
        columns=tuple(collector.columns),
        cte_names=cte_names,
        derived_aliases=derived_aliases,
        has_star_selection=collector.has_star_selection or _has_unsafe_star(expression),
    )


class _ColumnCollector:
    def __init__(self) -> None:
        self.columns: list[QualifiedColumn] = []
        self.has_star_selection = False
        self._seen: set[tuple[str | None, str | None, str]] = set()

    def add(self, entry: QualifiedColumn) -> None:
        marker = (entry.schema_name, entry.table_name, entry.column_name)
        if marker in self._seen:
            return
        self._seen.add(marker)
        self.columns.append(entry)


def _physical_table(
    table: exp.Table,
    non_catalog: frozenset[str],
    default_schema: str,
) -> QualifiedTable | None:
    name = (table.name or "").strip()
    if not name:
        return None
    name_l = name.lower()
    if not table.db and name_l in non_catalog:
        return None
    schema = (table.db or default_schema).strip().lower() or default_schema.lower()
    return QualifiedTable(schema_name=schema, table_name=name_l)


def _collect_scoped_columns(
    expression: exp.Expression,
    *,
    collector: _ColumnCollector,
    non_catalog: frozenset[str],
    alias_to_table: dict[str, QualifiedTable],
    default_schema: str,
) -> bool:
    """Resolve columns per scope. Returns False when scope analysis is unavailable."""
    try:
        scopes = list(traverse_scope(expression))
    except Exception:  # noqa: BLE001 — fall back to conservative flat extraction
        return False
    if not scopes:
        return False

    visited: set[int] = set()
    all_output_names: set[str] = set()

    for scope in scopes:
        physical: dict[str, QualifiedTable] = {}
        derived_outputs: dict[str, frozenset[str] | None] = {}
        for source_name, source in scope.sources.items():
            key = source_name.lower()
            if isinstance(source, exp.Table):
                qualified = _physical_table(source, non_catalog, default_schema)
                if qualified is not None:
                    physical[key] = qualified
            elif isinstance(source, Scope):
                derived_outputs[key] = _output_names(source.expression)

        own_outputs = _output_names(scope.expression) or frozenset()
        all_output_names.update(own_outputs)
        derived_names: set[str] = set()
        derived_has_unknown_outputs = False
        for outputs in derived_outputs.values():
            if outputs is None:
                derived_has_unknown_outputs = True
            else:
                derived_names.update(outputs)

        for column in scope.columns:
            visited.add(id(column))
            col_name = (column.name or "").strip()
            if not col_name or col_name == "*":
                continue
            col_l = col_name.lower()
            table_ref = (column.table or "").strip().lower() or None

            if table_ref is not None:
                if table_ref in derived_outputs or table_ref in non_catalog:
                    continue
                mapped = physical.get(table_ref) or alias_to_table.get(table_ref)
                if mapped is not None:
                    collector.add(
                        QualifiedColumn(
                            schema_name=mapped.schema_name,
                            table_name=mapped.table_name,
                            column_name=col_l,
                        )
                    )
                else:
                    collector.add(
                        QualifiedColumn(schema_name=None, table_name=table_ref, column_name=col_l)
                    )
                continue

            # Unqualified: output of a CTE/derived source in this scope.
            if col_l in derived_names:
                continue
            if not physical and derived_has_unknown_outputs:
                continue
            # SELECT aliases are only visible to GROUP BY / ORDER BY of the same query.
            if col_l in own_outputs and _in_group_or_order(column, scope.expression):
                continue
            collector.add(QualifiedColumn(schema_name=None, table_name=None, column_name=col_l))

    # Columns outside analyzed scopes (e.g. ORDER BY alias refs sqlglot elides).
    for column in expression.find_all(exp.Column):
        if id(column) in visited:
            continue
        col_name = (column.name or "").strip()
        if not col_name or col_name == "*":
            continue
        col_l = col_name.lower()
        table_ref = (column.table or "").strip().lower() or None
        if table_ref is None and col_l in all_output_names:
            continue
        _add_flat_column(column, collector, non_catalog, alias_to_table)
    return True


def _collect_flat_columns(
    expression: exp.Expression,
    *,
    collector: _ColumnCollector,
    non_catalog: frozenset[str],
    alias_to_table: dict[str, QualifiedTable],
) -> None:
    for column in expression.find_all(exp.Column):
        col_name = (column.name or "").strip()
        if not col_name or col_name == "*":
            continue
        _add_flat_column(column, collector, non_catalog, alias_to_table)


def _add_flat_column(
    column: exp.Column,
    collector: _ColumnCollector,
    non_catalog: frozenset[str],
    alias_to_table: dict[str, QualifiedTable],
) -> None:
    col_l = column.name.strip().lower()
    table_ref = (column.table or "").strip().lower() or None
    if table_ref is not None:
        if table_ref in non_catalog:
            return
        mapped = alias_to_table.get(table_ref)
        if mapped is not None:
            collector.add(
                QualifiedColumn(
                    schema_name=mapped.schema_name,
                    table_name=mapped.table_name,
                    column_name=col_l,
                )
            )
        else:
            collector.add(QualifiedColumn(schema_name=None, table_name=table_ref, column_name=col_l))
        return
    collector.add(QualifiedColumn(schema_name=None, table_name=None, column_name=col_l))


def _output_names(node: exp.Expression) -> frozenset[str] | None:
    """Projection names of a query; None when a star makes them unknown."""
    if isinstance(node, exp.Select):
        names: set[str] = set()
        for projection in node.expressions:
            if isinstance(projection, exp.Star) or (
                isinstance(projection, exp.Column) and projection.is_star
            ):
                return None
            name = projection.alias_or_name
            if name:
                names.add(name.lower())
        return frozenset(names)
    if isinstance(node, exp.SetOperation):
        return _output_names(node.left)
    if isinstance(node, exp.Subquery):
        return _output_names(node.this)
    return frozenset()


def _in_group_or_order(column: exp.Column, select: exp.Expression) -> bool:
    node: exp.Expression | None = column.parent
    while node is not None and node is not select:
        if isinstance(node, exp.Group) or (
            isinstance(node, exp.Order) and node.parent is select
        ):
            return True
        if isinstance(node, (exp.Window, exp.Where, exp.Having, exp.Join)):
            return False
        node = node.parent
    return False


def _has_unsafe_star(expression: exp.Expression) -> bool:
    """Star projections over physical tables expose unlisted columns.

    Stars inside aggregates (COUNT(*)) are safe. Stars that read only from
    CTE/derived outputs are safe because those outputs were checked in scope.
    """
    for star in expression.find_all(exp.Star):
        parent = star.parent
        if isinstance(parent, exp.AggFunc) or isinstance(parent, exp.Count):
            continue
        select = star.find_ancestor(exp.Select)
        if select is None:
            return True
        if not _star_reads_only_derived(select):
            return True
    for column in expression.find_all(exp.Column):
        if column.is_star:
            select = column.find_ancestor(exp.Select)
            if select is None or not _star_reads_only_derived(select, column.table):
                return True
    return False


def _star_reads_only_derived(select: exp.Select, qualifier: str | None = None) -> bool:
    try:
        scope = next(
            (s for s in traverse_scope(select.root()) if s.expression is select),
            None,
        )
    except Exception:  # noqa: BLE001
        return False
    if scope is None or not scope.sources:
        return False
    if qualifier:
        source = scope.sources.get(qualifier) or scope.sources.get(qualifier.lower())
        return isinstance(source, Scope) and _output_names(source.expression) is not None
    return all(
        isinstance(source, Scope) and _output_names(source.expression) is not None
        for source in scope.sources.values()
    )
