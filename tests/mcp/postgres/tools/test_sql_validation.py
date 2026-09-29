from __future__ import annotations

import pytest

from app.connectors.exceptions import ConnectorQueryError
from app.connectors.readonly_sql import validate_readonly_sql


def test_select_and_cte_are_allowed() -> None:
    assert validate_readonly_sql("SELECT 1") == "SELECT 1"
    assert validate_readonly_sql("  select\n    *\nfrom users  ") == (
        "select\n    *\nfrom users"
    )
    cte = """
    WITH data AS (
        SELECT *
        FROM users
    )
    SELECT *
    FROM data
    """
    assert validate_readonly_sql(cte).lower().startswith("with")


@pytest.mark.parametrize(
    "query",
    [
        "INSERT INTO t VALUES (1)",
        "InSeRt InTo t VALUES (1)",
        "UPDATE t SET x = 1",
        "DELETE FROM t",
        "DROP TABLE t",
        "ALTER TABLE t ADD COLUMN x int",
        "TRUNCATE t",
        "CREATE TABLE t (id int)",
        "GRANT SELECT ON t TO u",
        "REVOKE SELECT ON t FROM u",
        "MERGE INTO t USING s ON t.id = s.id WHEN MATCHED THEN DELETE",
        "CALL do_thing()",
        "DO $$ BEGIN DELETE FROM t; END $$",
        "BEGIN",
        "COMMIT",
        "ROLLBACK",
        "SAVEPOINT x",
        "SET search_path = public",
        "RESET ALL",
        "SELECT * FROM t FOR UPDATE",
        "SELECT * INTO copies FROM t",
        "WITH data AS (DELETE FROM users RETURNING *) SELECT * FROM data",
        "WITH data AS (INSERT INTO t VALUES (1) RETURNING *) SELECT * FROM data",
        "WITH data AS (UPDATE t SET x = 1 RETURNING *) SELECT * FROM data",
        "SELECT * FROM users; DELETE FROM users",
        "SELECT * FROM users; SELECT * FROM orders",
        "SELECT 1; DROP TABLE users",
    ],
)
def test_unsafe_sql_is_rejected(query: str) -> None:
    with pytest.raises(ConnectorQueryError):
        validate_readonly_sql(query)


def test_comments_cannot_hide_a_second_statement() -> None:
    with pytest.raises(ConnectorQueryError):
        validate_readonly_sql("SELECT * FROM users; /* DELETE FROM users */ SELECT 2")
    allowed = validate_readonly_sql("SELECT * FROM users /* DELETE FROM users */")
    assert "DELETE" not in allowed
    allowed_line = validate_readonly_sql("SELECT * FROM users -- DROP TABLE users")
    assert "DROP" not in allowed_line
    assert validate_readonly_sql("SELECT /* malicious content */ * FROM users")


def test_string_literals_are_not_treated_as_keywords() -> None:
    assert validate_readonly_sql("SELECT 'delete' AS note FROM users")
    assert validate_readonly_sql("SELECT $$ drop table users $$ AS note")
    assert validate_readonly_sql("SELECT * FROM information_schema.tables")


@pytest.mark.parametrize(
    "query",
    [
        "SELECT pg_sleep(5)",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT dblink('dbname=x', 'SELECT 1')",
        "SELECT lo_import('/tmp/x')",
        "SELECT set_config('x', 'y', false)",
        "SELECT current_setting('is_superuser')",
    ],
)
def test_dangerous_functions_are_rejected(query: str) -> None:
    with pytest.raises(ConnectorQueryError):
        validate_readonly_sql(query)


def test_leading_parenthesis_select_is_allowed() -> None:
    assert validate_readonly_sql("(SELECT 1)")
