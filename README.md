# AnalyticCastle Backend

Backend for **AnalyticCastle**, an AI Data Analyst Employee.

Phase 1 is a FastAPI foundation: configuration, logging, PostgreSQL/Supabase session setup, Alembic, and health endpoints.

## Requirements

- Python 3.11+
- PostgreSQL (local or [Supabase](https://supabase.com))

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -e ".[dev]"
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

## Environment setup

```bash
cp .env.example .env
```

Edit `.env` and set `DATABASE_URL` to your Postgres or Supabase connection string. Use the SQLAlchemy form `postgresql+psycopg://...`. Never commit `.env`.

For local Postgres, create a dedicated database and role if they do not exist:

```sql
CREATE ROLE analyticcastle LOGIN PASSWORD 'analyticcastle';
CREATE DATABASE analyticcastle OWNER analyticcastle;
```

| Variable | Purpose |
| --- | --- |
| `APP_NAME` | Application name |
| `APP_ENV` | Environment (`local`, `test`, `production`, …) |
| `DEBUG` | Verbose errors and debug logging |
| `DATABASE_URL` | SQLAlchemy Postgres URL |

## Running the API

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- `GET /` → `{"status": "ok"}`
- `GET /health` → `{"status": "ok"}`

StackPilot is an optional dev dependency for local orchestration (`pip install -e ".[dev]"`). Phase 1 is a single API process, so run it with uvicorn.

## Running tests

```bash
pytest
```

## Running Alembic

Create a revision after models are added:

```bash
alembic revision --autogenerate -m "describe the change"
alembic upgrade head
```

Other useful commands:

```bash
alembic current
alembic downgrade -1
```
