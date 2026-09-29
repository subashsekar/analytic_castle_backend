"""Bootstrap local PostgreSQL role and database for pytest.

When the dedicated ``analyticcastle`` database is unavailable, tests fall back to
the ``postgres`` maintenance database using the same credentials (see
``tests/conftest.py``).
"""

from __future__ import annotations

import os
import sys

import psycopg

TARGET_DB = "analyticcastle"
TARGET_USER = "analyticcastle"
TARGET_PASSWORD = "analyticcastle"
TARGET_URL = (
    "postgresql+psycopg://analyticcastle:analyticcastle@127.0.0.1:5432/analyticcastle"
)

ADMIN_CANDIDATES = [
    os.environ.get("POSTGRES_ADMIN_URL"),
    "postgresql://postgres@127.0.0.1:5432/postgres",
    "postgresql://postgres:postgres@127.0.0.1:5432/postgres",
    "postgresql://analyticcastle:analyticcastle@127.0.0.1:5432/postgres",
]


def _ready(url: str) -> bool:
    dsn = url.replace("postgresql+psycopg://", "postgresql://", 1)
    try:
        with psycopg.connect(dsn, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def _bootstrap(admin_dsn: str) -> None:
    with psycopg.connect(admin_dsn, connect_timeout=3) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_roles WHERE rolname = %s",
                (TARGET_USER,),
            )
            if cur.fetchone() is None:
                cur.execute(
                    f"CREATE ROLE {TARGET_USER} WITH LOGIN PASSWORD %s",
                    (TARGET_PASSWORD,),
                )
                print(f"created role {TARGET_USER}")
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (TARGET_DB,),
            )
            if cur.fetchone() is None:
                cur.execute(f'CREATE DATABASE "{TARGET_DB}" OWNER {TARGET_USER}')
                print(f"created database {TARGET_DB}")
            else:
                print(f"database {TARGET_DB} already exists")


def main() -> int:
    if _ready(TARGET_URL):
        print("test database already ready")
        return 0
    fallback = TARGET_URL.replace(f"/{TARGET_DB}", "/postgres")
    if _ready(fallback):
        print("dedicated database missing; tests can use postgres fallback")
        return 0

    last_error: Exception | None = None
    for candidate in ADMIN_CANDIDATES:
        if not candidate:
            continue
        try:
            _bootstrap(candidate)
            if _ready(TARGET_URL):
                print("test database bootstrap complete")
                return 0
        except Exception as exc:
            last_error = exc
            print(f"bootstrap attempt failed for {candidate}: {type(exc).__name__}: {exc}")

    if last_error is not None:
        print(f"failed to bootstrap test database: {last_error}")
    else:
        print("failed to bootstrap test database: no admin connection available")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
