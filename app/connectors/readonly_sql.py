"""Validate that a SQL string is a single read-only statement.

This is application-level defense. PostgreSQL sessions must still be opened
with ``default_transaction_read_only=on``. The validator does not execute SQL.
"""

from __future__ import annotations

import re

from app.connectors.exceptions import ConnectorQueryError
from app.core.config import settings

_CONTROL_HEAD = frozenset(
    {
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "truncate",
        "create",
        "grant",
        "revoke",
        "merge",
        "call",
        "do",
        "begin",
        "commit",
        "rollback",
        "savepoint",
        "set",
        "reset",
        "start",
        "vacuum",
        "copy",
        "listen",
        "notify",
        "load",
        "lock",
        "refresh",
        "reindex",
        "cluster",
        "analyze",
        "comment",
        "explain",
        "declare",
        "fetch",
        "move",
        "close",
        "prepare",
        "execute",
        "deallocate",
        "discard",
        "checkpoint",
        "security",
    }
)
_FORBIDDEN_BODY = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|"
    r"copy|call|do|vacuum|listen|notify|load|lock|refresh|merge|"
    r"reindex|cluster|begin|commit|rollback|savepoint|"
    r"prepare|execute|deallocate)\b",
    re.IGNORECASE,
)
_ROW_LOCK = re.compile(
    r"\bfor\s+(update|no\s+key\s+update|share|no\s+key\s+share)\b",
    re.IGNORECASE,
)
_SELECT_INTO = re.compile(r"\bselect\b[\s\S]*\binto\b", re.IGNORECASE)
_LEADING_WORD = re.compile(r"[A-Za-z_]+")


def validate_readonly_sql(query: str) -> str:
    """Return a single stripped read-only statement, or raise."""
    if not isinstance(query, str):
        raise ConnectorQueryError("Unable to query the PostgreSQL data source")
    if len(query) > settings.MCP_QUERY_MAX_SQL_CHARS:
        raise ConnectorQueryError("Unable to query the PostgreSQL data source")
    stripped_comments = _strip_sql_comments(query)
    statement = stripped_comments.strip()
    if statement.endswith(";"):
        statement = statement[:-1].rstrip()
    if not statement or ";" in _mask_literals(statement):
        raise ConnectorQueryError("Unable to query the PostgreSQL data source")
    scanned = _mask_literals(statement)
    head_match = _LEADING_WORD.match(scanned.lstrip("(").lstrip())
    if head_match is None:
        raise ConnectorQueryError("Unable to query the PostgreSQL data source")
    head = head_match.group(0).lower()
    if head in _CONTROL_HEAD or head not in {"select", "with"}:
        raise ConnectorQueryError("Unable to query the PostgreSQL data source")
    if _FORBIDDEN_BODY.search(scanned):
        raise ConnectorQueryError("Unable to query the PostgreSQL data source")
    if _ROW_LOCK.search(scanned) or _SELECT_INTO.search(scanned):
        raise ConnectorQueryError("Unable to query the PostgreSQL data source")
    return statement


def _strip_sql_comments(query: str) -> str:
    """Remove ``--`` and ``/* */`` comments without touching string literals."""
    out: list[str] = []
    index = 0
    length = len(query)
    while index < length:
        char = query[index]
        nxt = query[index + 1] if index + 1 < length else ""
        if char == "'":
            chunk, index = _read_single_quoted(query, index)
            out.append(chunk)
            continue
        if char == "$" and _dollar_tag_at(query, index) is not None:
            chunk, index = _read_dollar_quoted(query, index)
            out.append(chunk)
            continue
        if char == "-" and nxt == "-":
            newline = query.find("\n", index)
            index = length if newline < 0 else newline
            continue
        if char == "/" and nxt == "*":
            end = query.find("*/", index + 2)
            if end < 0:
                raise ConnectorQueryError("Unable to query the PostgreSQL data source")
            index = end + 2
            out.append(" ")
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _mask_literals(query: str) -> str:
    out: list[str] = []
    index = 0
    length = len(query)
    while index < length:
        char = query[index]
        if char == "'":
            chunk, index = _read_single_quoted(query, index)
            out.append(" " * len(chunk))
            continue
        if char == "$" and _dollar_tag_at(query, index) is not None:
            chunk, index = _read_dollar_quoted(query, index)
            out.append(" " * len(chunk))
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _read_single_quoted(query: str, start: int) -> tuple[str, int]:
    index = start + 1
    length = len(query)
    while index < length:
        if query[index] == "'":
            if index + 1 < length and query[index + 1] == "'":
                index += 2
                continue
            return query[start : index + 1], index + 1
        index += 1
    raise ConnectorQueryError("Unable to query the PostgreSQL data source")


def _dollar_tag_at(query: str, start: int) -> str | None:
    if start >= len(query) or query[start] != "$":
        return None
    index = start + 1
    while index < len(query) and (query[index].isalnum() or query[index] == "_"):
        index += 1
    if index < len(query) and query[index] == "$":
        return query[start : index + 1]
    return None


def _read_dollar_quoted(query: str, start: int) -> tuple[str, int]:
    tag = _dollar_tag_at(query, start)
    if tag is None:
        return query[start], start + 1
    end = query.find(tag, start + len(tag))
    if end < 0:
        raise ConnectorQueryError("Unable to query the PostgreSQL data source")
    close = end + len(tag)
    return query[start:close], close
