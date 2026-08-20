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

Set `DATA_SOURCE_ENCRYPTION_KEY` to a 32-byte key before starting the API. Generate one locally and paste the output into `.env` (the command prints the key to your terminal only):

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Do not reuse `JWT_SECRET_KEY` or `DATABASE_URL` as the encryption key.

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
| `JWT_SECRET_KEY` | Secret used to sign access and refresh tokens |
| `JWT_ALGORITHM` | JWT signing algorithm (`HS256`) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Access token lifetime |
| `REFRESH_TOKEN_EXPIRE_DAYS` | Refresh token lifetime |
| `EMAIL_FROM` | Sender address used by the email service |
| `EMAIL_FROM_NAME` | Display name in the From header |
| `EMAIL_PROVIDER` | `console` (log only) or `smtp` |
| `FRONTEND_URL` | Frontend origin used to build verification and reset links |
| `EMAIL_VERIFICATION_EXPIRE_MINUTES` | Email verification token lifetime |
| `PASSWORD_RESET_EXPIRE_MINUTES` | Password reset token lifetime |
| `EMAIL_VERIFICATION_RESEND_COOLDOWN_SECONDS` | Minimum wait between verification emails |
| `SMTP_HOST` | SMTP server host (required for `smtp`) |
| `SMTP_PORT` | SMTP port |
| `SMTP_USERNAME` | SMTP username |
| `SMTP_PASSWORD` | SMTP password / app password |
| `SMTP_USE_TLS` | Use STARTTLS |
| `SMTP_TIMEOUT_SECONDS` | SMTP connection timeout |
| `CORS_ALLOWED_ORIGINS` | Comma-separated browser origins allowed to call the API |
| `CORS_ALLOW_CREDENTIALS` | Whether credentialed CORS requests are allowed |
| `CORS_ALLOWED_METHODS` | Comma-separated CORS methods |
| `CORS_ALLOWED_HEADERS` | Comma-separated CORS request headers |
| `RATE_LIMIT_ENABLED` | Enable endpoint rate limits and login throttling |
| `RATE_LIMIT_LOGIN` | Login request limit (`5/minute`) |
| `RATE_LIMIT_REGISTER` | Registration request limit |
| `RATE_LIMIT_REFRESH` | Refresh request limit |
| `RATE_LIMIT_PASSWORD_RESET` | Forgot/reset password request limit |
| `RATE_LIMIT_VERIFY_EMAIL_SEND` | Resend-verification request limit |
| `RATE_LIMIT_CHANGE_PASSWORD` | Change-password request limit |
| `RATE_LIMIT_TEST_CONNECTION` | Data-source connection-test request limit |
| `RATE_LIMIT_METADATA_SEARCH` | Metadata search request limit |
| `RATE_LIMIT_METADATA_SYNC` | Metadata synchronization request limit |
| `RATE_LIMIT_SAMPLE_DATA` | Sample-data request limit |
| `LOGIN_MAX_FAILED_ATTEMPTS` | Failed logins before a temporary cooldown |
| `LOGIN_LOCKOUT_SECONDS` | Cooldown after too many failed logins |
| `SECURITY_HEADERS_ENABLED` | Add API security headers to responses |
| `MAX_REQUEST_BODY_BYTES` | Maximum request `Content-Length` |
| `TRUST_PROXY_HEADERS` | Trust `X-Forwarded-For` when behind a known proxy |
| `HEALTH_CHECK_TIMEOUT_SECONDS` | Timeout for the readiness database check |
| `POSTGRES_CONNECT_TIMEOUT` | Timeout in seconds when connecting to a customer PostgreSQL data source |
| `POSTGRES_DISCOVERY_TIMEOUT` | Timeout in seconds for PostgreSQL metadata discovery catalog queries |
| `METADATA_DISCOVERY_MAX_SCHEMAS` | Maximum schemas returned by PostgreSQL metadata discovery |
| `METADATA_DISCOVERY_MAX_TABLES` | Maximum tables returned by PostgreSQL metadata discovery |
| `METADATA_DISCOVERY_MAX_COLUMNS` | Maximum columns returned by PostgreSQL metadata discovery |
| `METADATA_DISCOVERY_MAX_RELATIONSHIPS` | Maximum foreign-key relationships returned by discovery |
| `METADATA_SEARCH_DEFAULT_LIMIT` | Default number of internal metadata search results (`50`) |
| `METADATA_SEARCH_MAX_LIMIT` | Maximum number of internal metadata search results (`100`) |
| `METADATA_SEARCH_MAX_QUERY_LENGTH` | Maximum metadata search query length (`255`) |
| `METADATA_API_DEFAULT_PAGE_SIZE` | Default page size for metadata list APIs (`50`) |
| `METADATA_API_MAX_PAGE_SIZE` | Maximum page size for metadata list APIs (`100`) |
| `SAMPLE_DATA_DEFAULT_LIMIT` | Default number of sample rows retrieved from a customer table (`10`) |
| `SAMPLE_DATA_MAX_LIMIT` | Maximum number of sample rows (`100`). Requested limits above this are capped. |
| `SAMPLE_DATA_MAX_COLUMNS` | Maximum columns included in a sample (`100`) |
| `SAMPLE_DATA_MAX_VALUE_CHARS` | Maximum characters returned for a single text sample value (`1024`) |
| `SAMPLE_DATA_MAX_JSON_CHARS` | Maximum characters allowed for a serialized JSON sample value (`4096`) |
| `MCP_QUERY_DEFAULT_LIMIT` | Default row limit for the PostgreSQL MCP query tool (`100`) |
| `MCP_QUERY_MAX_LIMIT` | Maximum row limit for the PostgreSQL MCP query tool (`1000`) |
| `MCP_QUERY_TIMEOUT_SECONDS` | Timeout for MCP query execution |
| `MCP_QUERY_MAX_SQL_CHARS` | Maximum SQL length accepted by the MCP query tool |
| `MCP_QUERY_MAX_RESULT_CHARS` | Maximum serialized MCP query result size |
| `MCP_QUERY_MAX_VALUE_CHARS` | Maximum characters for a single MCP query cell |
| `MCP_QUERY_MAX_JSON_CHARS` | Maximum characters for JSON/array MCP query values |
| `DATA_SOURCE_ENCRYPTION_KEY` | 32-byte AES-256 key for customer data-source passwords (64-char hex or url-safe base64). Generate with `python -c "import secrets; print(secrets.token_hex(32))"`. Never reuse `JWT_SECRET_KEY`. |
| `LLM_PROVIDER` | LLM provider (`openai`, `openrouter`, or `openai_compatible`) |
| `LLM_API_KEY` | Provider API key. Never commit a real key. The API starts without one; AI chat then returns 503 |
| `LLM_BASE_URL` | Optional override. Required when `LLM_PROVIDER=openai_compatible` |
| `LLM_MODEL` | Model name sent to the provider (`gpt-4o-mini`) |
| `LLM_TIMEOUT_SECONDS` | Provider HTTP timeout |
| `LLM_TEMPERATURE` | Sampling temperature |
| `LLM_MAX_OUTPUT_TOKENS` | Maximum completion tokens requested from the provider |
| `AI_MAX_MESSAGE_CHARS` | Maximum user message length for AI chat |
| `AI_MAX_CONTEXT_CHARS` | Maximum prompt context size, including metadata snippets |
| `AI_MAX_OUTPUT_CHARS` | Maximum accepted model output size |
| `AI_METADATA_SEARCH_LIMIT` | Maximum metadata search hits included in AI prompt context |
| `AI_METADATA_RESOLVE_SEARCH_LIMIT` | Maximum catalog search hits per intent concept during metadata resolution |
| `AI_MAX_METADATA_TABLES` | Maximum table candidates returned in AI metadata context (`20`) |
| `AI_MAX_METADATA_COLUMNS` | Maximum column candidates returned in AI metadata context (`50`) |
| `AI_MAX_METADATA_RELATIONSHIPS` | Maximum relationship candidates returned in AI metadata context (`30`) |
| `RATE_LIMIT_AI_CHAT` | AI chat request limit (`10/minute`) |

## Running the API

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- `GET /` → `{"status": "ok"}`
- `GET /health` → application and database readiness
- `GET /health/live` → process liveness (no database query)
- `GET /health/ready` → application and database readiness
- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `GET /api/v1/auth/me`
- `POST /api/v1/auth/refresh`
- `POST /api/v1/auth/logout`
- `POST /api/v1/auth/verify-email`
- `POST /api/v1/auth/verify-email/send`
- `POST /api/v1/auth/resend-verification`
- `POST /api/v1/auth/forgot-password`
- `POST /api/v1/auth/reset-password`
- `POST /api/v1/auth/change-password`
- `POST /api/v1/organizations`
- `GET /api/v1/organizations`
- `GET /api/v1/organizations/{organization_id}`
- `PATCH /api/v1/organizations/{organization_id}`
- `DELETE /api/v1/organizations/{organization_id}`
- `POST /api/v1/workspaces`
- `GET /api/v1/workspaces`
- `GET /api/v1/workspaces/{workspace_id}`
- `PATCH /api/v1/workspaces/{workspace_id}`
- `DELETE /api/v1/workspaces/{workspace_id}`
- `GET /api/v1/workspaces/{workspace_id}/members`
- `POST /api/v1/workspaces/{workspace_id}/members`
- `PATCH /api/v1/workspaces/{workspace_id}/members/{user_id}`
- `DELETE /api/v1/workspaces/{workspace_id}/members/{user_id}`
- `POST /api/v1/data-sources`
- `GET /api/v1/data-sources`
- `GET /api/v1/data-sources/{data_source_id}`
- `PATCH /api/v1/data-sources/{data_source_id}`
- `DELETE /api/v1/data-sources/{data_source_id}`
- `POST /api/v1/data-sources/{data_source_id}/test-connection`
- `GET /api/v1/data-sources/{data_source_id}/schemas`
- `GET /api/v1/data-sources/{data_source_id}/schemas/{schema_id}`
- `GET /api/v1/data-sources/{data_source_id}/tables`
- `GET /api/v1/data-sources/{data_source_id}/tables/{table_id}`
- `GET /api/v1/data-sources/{data_source_id}/tables/{table_id}/columns`
- `GET /api/v1/data-sources/{data_source_id}/relationships`
- `GET /api/v1/data-sources/{data_source_id}/metadata/search`
- `POST /api/v1/data-sources/{data_source_id}/metadata/sync`
- `GET /api/v1/data-sources/{data_source_id}/metadata/sync-status`
- `POST /api/v1/data-sources/{data_source_id}/tables/{table_id}/sample`
- `POST /api/v1/ai/chat`

Every response includes `X-Request-ID`. If the client sends a valid `X-Request-ID`, it is preserved; otherwise the API generates one. Request completion is logged with method, path, status, duration, and request ID. Request bodies and secrets are not logged.

Authorization is applied with reusable FastAPI dependencies. Authentication still answers who the user is; authorization answers whether that user may perform the action. Roles are loaded from the database on each request, not from the JWT.

- `get_current_user()` — require a valid access token
- `require_roles(...)` — global role check (`SUPER_ADMIN` > `ADMIN` > `USER`)
- `get_organization_access()` — require access to the path organization
- `require_organization_role(...)` — require a workspace role within that organization
- `require_organization_owner()` — require ownership of the organization's founding workspace
- `get_workspace_member()` — require membership in the path workspace
- `require_workspace_role(...)` — workspace role check (`OWNER` > `ADMIN` > `MEMBER`)
- `require_workspace_permission(...)` — workspace permission check

`SUPER_ADMIN` does not bypass workspace isolation unless a dependency is called with `allow_super_admin=True`.

StackPilot is an optional dev dependency for local orchestration (`pip install -e ".[dev]"`). Phase 1 is a single API process, so run it with uvicorn.

## Running tests

```bash
pytest
```

PostgreSQL connector integration tests run only when `TEST_POSTGRES_HOST`, `TEST_POSTGRES_DATABASE`, `TEST_POSTGRES_USERNAME`, and `TEST_POSTGRES_PASSWORD` are set. `TEST_POSTGRES_PORT` defaults to `5432`. If those variables are unset, the integration tests are skipped.

PostgreSQL metadata discovery is an internal service. It inspects a connected customer database and returns schemas, tables, views, columns, types, keys, and foreign-key relationships. It does not persist metadata. Discovery queries are read-only and use the existing PostgreSQL connector timeout. If a configured discovery limit is exceeded, discovery fails instead of truncating results.

PostgreSQL metadata synchronization is an internal service. It runs Phase 4.2 discovery, then persists the result into AnalyticCastle metadata tables in one application-database transaction. Synchronization is idempotent, replaces stale metadata for that data source only, and records sync status. Discovery completes before the persist transaction begins.

Metadata HTTP APIs expose persisted schemas, tables, columns, and relationships for an authorized workspace data source. They also expose metadata search, synchronous metadata synchronization, sync status, and masked sample rows. Metadata GET endpoints read the application metadata database and do not query the customer database. Synchronization and sample-data endpoints are rate-limited.

Internal metadata search reads the same persisted metadata tables. It is scoped to a workspace data source, matches schema/table/column names case-insensitively with partial matching, can match table and column descriptions, and ranks exact matches ahead of prefix and contains matches. Search runs in PostgreSQL and does not expose credentials.

Safe sample-data retrieval is an internal service. It loads a discovered table from AnalyticCastle metadata, runs a read-only `SELECT` with an explicit column list and SQL `LIMIT` through the existing PostgreSQL connector, then masks PII and secrets before returning rows. Default sample size is 10 rows (maximum 100). Callers cannot request unmasked values.

The AI analyst exposes `POST /api/v1/ai/chat`. It authenticates the user, authorizes the workspace data source, detects a structured analytical intent, resolves relevant Phase 4 catalog metadata for that intent, and returns a non-executable request plan plus compact metadata context. It does not generate SQL, execute queries, or call the customer database. Unit tests use a fake provider. Live LLM tests run only when `TEST_LLM_API_KEY` is set.

The in-process PostgreSQL MCP query tool (`postgres.query`) executes a single read-only SQL statement against an authorized data source through the existing connector. It does not accept connection strings or credentials as tool arguments. Query parameters are not supported; bind values are not interpolated into SQL. Live query tests run only when `TEST_POSTGRES_HOST` and related variables are set.


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
