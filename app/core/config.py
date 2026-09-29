import base64
import binascii
import re

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENCRYPTION_KEY_BYTES = 32
_HEX_KEY_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_PLACEHOLDER_ENCRYPTION_KEYS = frozenset(
    {
        "default-key",
        "development-key",
        "secret",
        "changeme",
        "change-me",
        "password",
    }
)
_PLACEHOLDER_ENCRYPTION_MARKERS = (
    "default-key",
    "development-key",
    "changeme",
    "change-me",
)

_RATE_LIMIT_UNITS: dict[str, int] = {
    "second": 1,
    "seconds": 1,
    "minute": 60,
    "minutes": 60,
    "hour": 3600,
    "hours": 3600,
}


def parse_rate_limit(value: str) -> tuple[int, int]:
    """Parse a compact limit such as ``5/minute`` into ``(times, window_seconds)``."""
    raw = value.strip().lower()
    times_text, separator, unit = raw.partition("/")
    if not separator or not times_text or not unit:
        raise ValueError("Rate limit must look like '5/minute'")
    try:
        times = int(times_text)
    except ValueError as exc:
        raise ValueError("Rate limit count must be an integer") from exc
    window_seconds = _RATE_LIMIT_UNITS.get(unit)
    if times < 1 or window_seconds is None:
        raise ValueError("Rate limit must look like '5/minute'")
    return times, window_seconds


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def decode_data_source_encryption_key(value: str) -> bytes:
    """Decode a 32-byte AES-256 key from hex or url-safe base64.

    Generate a development key with:
    ``python -c "import secrets; print(secrets.token_hex(32))"``
    """
    stripped = value.strip()
    if not stripped:
        raise ValueError(
            "DATA_SOURCE_ENCRYPTION_KEY is required. Generate a 32-byte key "
            'with: python -c "import secrets; print(secrets.token_hex(32))"'
        )
    lowered = stripped.lower()
    if lowered in _PLACEHOLDER_ENCRYPTION_KEYS or any(
        marker in lowered for marker in _PLACEHOLDER_ENCRYPTION_MARKERS
    ):
        raise ValueError("DATA_SOURCE_ENCRYPTION_KEY must not use a placeholder value")

    decoded: bytes | None = None
    if _HEX_KEY_PATTERN.fullmatch(stripped):
        decoded = bytes.fromhex(stripped)
    else:
        padded = stripped + "=" * ((-len(stripped)) % 4)
        try:
            decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
        except (ValueError, binascii.Error, UnicodeEncodeError) as exc:
            raise ValueError(
                "DATA_SOURCE_ENCRYPTION_KEY must be a 64-character hex string "
                "or url-safe base64 encoding of 32 bytes"
            ) from exc
    if len(decoded) != _ENCRYPTION_KEY_BYTES:
        raise ValueError("DATA_SOURCE_ENCRYPTION_KEY must decode to 32 bytes")
    return decoded


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "AnalyticCastle"
    APP_ENV: str = "local"
    DEBUG: bool = False
    DATABASE_URL: str = Field(repr=False)

    JWT_SECRET_KEY: str = Field(min_length=32, repr=False)
    JWT_ALGORITHM: str = "HS256"
    DATA_SOURCE_ENCRYPTION_KEY: str = Field(repr=False)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    EMAIL_PROVIDER: str = "console"
    EMAIL_FROM: str = "noreply@analyticcastle.local"
    EMAIL_FROM_NAME: str = "AnalyticCastle"
    FRONTEND_URL: str = "http://localhost:3000"
    EMAIL_VERIFICATION_EXPIRE_MINUTES: int = 1440
    PASSWORD_RESET_EXPIRE_MINUTES: int = 60
    EMAIL_VERIFICATION_RESEND_COOLDOWN_SECONDS: int = 60

    SMTP_HOST: str = ""
    SMTP_PORT: int = Field(default=587, ge=1, le=65535)
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = Field(default="", repr=False)
    SMTP_USE_TLS: bool = True
    SMTP_TIMEOUT_SECONDS: float = Field(default=30.0, gt=0, le=300)

    CORS_ALLOWED_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOWED_METHODS: str = "GET,POST,PUT,PATCH,DELETE,OPTIONS"
    CORS_ALLOWED_HEADERS: str = "Authorization,Content-Type,Accept,X-Request-ID"

    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_LOGIN: str = "10/minute"
    RATE_LIMIT_REGISTER: str = "5/minute"
    RATE_LIMIT_REFRESH: str = "30/minute"
    RATE_LIMIT_PASSWORD_RESET: str = "5/minute"
    RATE_LIMIT_VERIFY_EMAIL_SEND: str = "5/minute"
    RATE_LIMIT_CHANGE_PASSWORD: str = "5/minute"
    RATE_LIMIT_TEST_CONNECTION: str = "10/minute"
    RATE_LIMIT_METADATA_SEARCH: str = "30/minute"
    RATE_LIMIT_METADATA_SYNC: str = "5/minute"
    RATE_LIMIT_SAMPLE_DATA: str = "10/minute"
    RATE_LIMIT_AI_CHAT: str = "10/minute"
    RATE_LIMIT_MCP_QUERY: str = "10/minute"
    RATE_LIMIT_MCP_SAMPLE: str = "10/minute"
    RATE_LIMIT_MCP_METADATA: str = "30/minute"
    LOGIN_MAX_FAILED_ATTEMPTS: int = Field(default=5, ge=1)
    LOGIN_LOCKOUT_SECONDS: int = Field(default=60, ge=1)

    SECURITY_HEADERS_ENABLED: bool = True
    MAX_REQUEST_BODY_BYTES: int = Field(default=1_048_576, ge=1024)
    TRUST_PROXY_HEADERS: bool = False
    HEALTH_CHECK_TIMEOUT_SECONDS: float = Field(default=2.0, gt=0, le=30)
    POSTGRES_CONNECT_TIMEOUT: float = Field(default=10.0, gt=0, le=120)
    POSTGRES_DISCOVERY_TIMEOUT: float = Field(default=30.0, gt=0, le=300)
    METADATA_DISCOVERY_MAX_SCHEMAS: int = Field(default=100, ge=1, le=10_000)
    METADATA_DISCOVERY_MAX_TABLES: int = Field(default=2_000, ge=1, le=100_000)
    METADATA_DISCOVERY_MAX_COLUMNS: int = Field(default=20_000, ge=1, le=1_000_000)
    METADATA_DISCOVERY_MAX_RELATIONSHIPS: int = Field(default=5_000, ge=1, le=100_000)
    METADATA_SEARCH_DEFAULT_LIMIT: int = Field(default=50, ge=1, le=100)
    METADATA_SEARCH_MAX_LIMIT: int = Field(default=100, ge=1, le=100)
    METADATA_SEARCH_MAX_QUERY_LENGTH: int = Field(default=255, ge=1, le=1_000)
    METADATA_API_DEFAULT_PAGE_SIZE: int = Field(default=50, ge=1, le=100)
    METADATA_API_MAX_PAGE_SIZE: int = Field(default=100, ge=1, le=100)
    SAMPLE_DATA_DEFAULT_LIMIT: int = Field(default=10, ge=1, le=100)
    SAMPLE_DATA_MAX_LIMIT: int = Field(default=100, ge=1, le=100)
    SAMPLE_DATA_MAX_COLUMNS: int = Field(default=100, ge=1, le=1_000)
    SAMPLE_DATA_MAX_VALUE_CHARS: int = Field(default=1_024, ge=32, le=1_000_000)
    SAMPLE_DATA_MAX_JSON_CHARS: int = Field(default=4_096, ge=32, le=1_000_000)
    MCP_QUERY_DEFAULT_LIMIT: int = Field(default=100, ge=1, le=10_000)
    MCP_QUERY_MAX_LIMIT: int = Field(default=1_000, ge=1, le=10_000)
    MCP_QUERY_TIMEOUT_SECONDS: float = Field(default=30.0, gt=0, le=300)
    MCP_QUERY_MAX_SQL_CHARS: int = Field(default=10_000, ge=1, le=100_000)
    MCP_QUERY_MAX_RESULT_CHARS: int = Field(default=262_144, ge=1_024, le=2_000_000)
    MCP_QUERY_MAX_VALUE_CHARS: int = Field(default=1_024, ge=32, le=1_000_000)
    MCP_QUERY_MAX_JSON_CHARS: int = Field(default=4_096, ge=32, le=1_000_000)

    LLM_PROVIDER: str = "openai"
    LLM_API_KEY: str = Field(default="", repr=False)
    LLM_BASE_URL: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TIMEOUT_SECONDS: float = Field(default=30.0, gt=0, le=120)
    LLM_TEMPERATURE: float = Field(default=0.2, ge=0, le=2)
    LLM_MAX_OUTPUT_TOKENS: int = Field(default=4_096, ge=1, le=8_192)
    LLM_MAX_RETRIES: int = Field(default=3, ge=0, le=10)
    LLM_RETRY_BASE_BACKOFF_SECONDS: float = Field(default=0.5, gt=0, le=60)
    LLM_RETRY_MAX_BACKOFF_SECONDS: float = Field(default=30.0, gt=0, le=300)
    OPENROUTER_API_KEY: str = Field(default="", repr=False)
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    AI_MAX_MESSAGE_CHARS: int = Field(default=4_000, ge=1, le=32_000)
    AI_MAX_CONTEXT_CHARS: int = Field(default=8_000, ge=1, le=64_000)
    AI_MAX_CONTEXT_MESSAGES: int = Field(default=100, ge=1, le=1_000)
    AI_MAX_OUTPUT_CHARS: int = Field(default=8_000, ge=1, le=64_000)
    AI_CONVERSATION_DEFAULT_PAGE_SIZE: int = Field(default=50, ge=1, le=100)
    AI_CONVERSATION_MAX_PAGE_SIZE: int = Field(default=100, ge=1, le=100)
    AI_LLM_CONTEXT_CHARS: int = Field(default=8_000, ge=1, le=64_000)
    AI_METADATA_SEARCH_LIMIT: int = Field(default=10, ge=1, le=50)
    AI_METADATA_RESOLVE_SEARCH_LIMIT: int = Field(default=50, ge=1, le=100)
    AI_MAX_METADATA_TABLES: int = Field(default=20, ge=1, le=100)
    AI_MAX_METADATA_COLUMNS: int = Field(default=50, ge=1, le=200)
    AI_MAX_METADATA_RELATIONSHIPS: int = Field(default=30, ge=1, le=100)
    AI_MAX_RESULT_LIMIT: int = Field(default=100, ge=1, le=10_000)
    AI_MAX_PLAN_METRICS: int = Field(default=10, ge=1, le=50)
    AI_MAX_PLAN_DIMENSIONS: int = Field(default=10, ge=1, le=50)
    AI_MAX_PLAN_FILTERS: int = Field(default=20, ge=1, le=50)
    AI_MAX_FILTER_VALUES: int = Field(default=25, ge=1, le=100)
    AI_MAX_CONCEPT_CHARS: int = Field(default=128, ge=16, le=512)
    AI_SQL_MAX_SQL_CHARS: int = Field(default=10_000, ge=1, le=100_000)
    AI_SQL_MAX_SCHEMA_CHARS: int = Field(default=8_000, ge=256, le=64_000)
    AI_SQL_MAX_PLAN_SUMMARY_CHARS: int = Field(default=2_000, ge=64, le=8_000)
    AI_SQL_MAX_EXPLANATION_CHARS: int = Field(default=1_000, ge=32, le=4_000)
    AI_SQL_MAX_ASSUMPTIONS: int = Field(default=20, ge=1, le=50)
    AI_SQL_VALIDATION_DEFAULT_SCHEMA: str = Field(default="public", min_length=1, max_length=63)
    AI_SQL_CORRECTION_MAX_ATTEMPTS: int = Field(default=2, ge=1, le=5)
    AI_SQL_CORRECTION_MAX_FEEDBACK_CHARS: int = Field(default=2_000, ge=64, le=8_000)
    # Chat Phase 8. Independent agents run concurrently, so the budget is the
    # wall-clock cap for all agents together; the chat request also spends time
    # on intent + SQL before this. Keep the frontend timeout above
    # intent/SQL time + budget (≈ 150–180s with defaults).
    AI_CHAT_PHASE8_BUDGET_SECONDS: float = Field(default=75.0, gt=5, le=300)
    AI_CHAT_PHASE8_AGENT_TIMEOUT_SECONDS: float = Field(default=30.0, gt=1, le=120)
    # Root cause makes several LLM calls (hypotheses, evidence SQL, re-rank).
    AI_CHAT_PHASE8_RCA_TIMEOUT_SECONDS: float = Field(default=55.0, gt=1, le=240)
    AI_CHAT_PHASE8_RCA_MAX_INVESTIGATION_QUERIES: int = Field(default=1, ge=0, le=5)
    AI_CHAT_PHASE8_MAX_TOKENS: int = Field(default=1_200, ge=256, le=4_096)
    AI_CHAT_PHASE8_PROMPT_ROWS: int = Field(default=20, ge=1, le=100)
    # Deterministic catalog-only query used when LLM SQL cannot be validated.
    AI_CHAT_DETERMINISTIC_SQL_FALLBACK: bool = True

    @field_validator("JWT_ALGORITHM")
    @classmethod
    def only_hs256(cls, value: str) -> str:
        if value != "HS256":
            raise ValueError("Only HS256 is supported")
        return value

    @field_validator("EMAIL_PROVIDER")
    @classmethod
    def validate_email_provider(cls, value: str) -> str:
        provider = value.strip().lower()
        if provider not in {"console", "smtp"}:
            raise ValueError("EMAIL_PROVIDER must be 'console' or 'smtp'")
        return provider

    @field_validator("LLM_PROVIDER")
    @classmethod
    def validate_llm_provider(cls, value: str) -> str:
        provider = value.strip().lower()
        if provider not in {"openai", "openrouter", "openai_compatible"}:
            raise ValueError(
                "LLM_PROVIDER must be 'openai', 'openrouter', or 'openai_compatible'"
            )
        return provider

    @field_validator(
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        mode="before",
    )
    @classmethod
    def strip_llm_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("SMTP_PASSWORD", mode="before")
    @classmethod
    def strip_smtp_password_quotes(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().strip('"').strip("'")
        return value

    @field_validator("DATA_SOURCE_ENCRYPTION_KEY")
    @classmethod
    def validate_encryption_key(cls, value: str) -> str:
        decode_data_source_encryption_key(value)
        return value.strip()

    @field_validator(
        "RATE_LIMIT_LOGIN",
        "RATE_LIMIT_REGISTER",
        "RATE_LIMIT_REFRESH",
        "RATE_LIMIT_PASSWORD_RESET",
        "RATE_LIMIT_VERIFY_EMAIL_SEND",
        "RATE_LIMIT_CHANGE_PASSWORD",
        "RATE_LIMIT_TEST_CONNECTION",
        "RATE_LIMIT_METADATA_SEARCH",
        "RATE_LIMIT_METADATA_SYNC",
        "RATE_LIMIT_SAMPLE_DATA",
        "RATE_LIMIT_AI_CHAT",
        "RATE_LIMIT_MCP_QUERY",
        "RATE_LIMIT_MCP_SAMPLE",
        "RATE_LIMIT_MCP_METADATA",
    )
    @classmethod
    def validate_rate_limit_spec(cls, value: str) -> str:
        parse_rate_limit(value)
        return value

    @model_validator(mode="after")
    def validate_runtime_settings(self) -> "Settings":
        if self.EMAIL_PROVIDER == "smtp":
            missing = [
                name
                for name, value in (
                    ("SMTP_HOST", self.SMTP_HOST),
                    ("SMTP_USERNAME", self.SMTP_USERNAME),
                    ("SMTP_PASSWORD", self.SMTP_PASSWORD),
                )
                if not str(value).strip()
            ]
            if missing:
                raise ValueError(
                    "SMTP settings required when EMAIL_PROVIDER=smtp: "
                    + ", ".join(missing)
                )
        encryption_key = self.DATA_SOURCE_ENCRYPTION_KEY.strip()
        if encryption_key == self.JWT_SECRET_KEY.strip():
            raise ValueError(
                "DATA_SOURCE_ENCRYPTION_KEY must be independent from JWT_SECRET_KEY"
            )
        if encryption_key == self.DATABASE_URL.strip():
            raise ValueError(
                "DATA_SOURCE_ENCRYPTION_KEY must be independent from DATABASE_URL"
            )
        if self.METADATA_SEARCH_DEFAULT_LIMIT > self.METADATA_SEARCH_MAX_LIMIT:
            raise ValueError(
                "METADATA_SEARCH_DEFAULT_LIMIT cannot exceed METADATA_SEARCH_MAX_LIMIT"
            )
        if self.SAMPLE_DATA_DEFAULT_LIMIT > self.SAMPLE_DATA_MAX_LIMIT:
            raise ValueError(
                "SAMPLE_DATA_DEFAULT_LIMIT cannot exceed SAMPLE_DATA_MAX_LIMIT"
            )
        if self.METADATA_API_DEFAULT_PAGE_SIZE > self.METADATA_API_MAX_PAGE_SIZE:
            raise ValueError(
                "METADATA_API_DEFAULT_PAGE_SIZE cannot exceed METADATA_API_MAX_PAGE_SIZE"
            )
        if self.AI_METADATA_RESOLVE_SEARCH_LIMIT > self.METADATA_SEARCH_MAX_LIMIT:
            raise ValueError(
                "AI_METADATA_RESOLVE_SEARCH_LIMIT cannot exceed "
                "METADATA_SEARCH_MAX_LIMIT"
            )
        if self.MCP_QUERY_DEFAULT_LIMIT > self.MCP_QUERY_MAX_LIMIT:
            raise ValueError(
                "MCP_QUERY_DEFAULT_LIMIT cannot exceed MCP_QUERY_MAX_LIMIT"
            )
        if self.LLM_PROVIDER == "openai_compatible" and not self.LLM_BASE_URL:
            raise ValueError(
                "LLM_BASE_URL is required when LLM_PROVIDER=openai_compatible"
            )
        if self.LLM_RETRY_BASE_BACKOFF_SECONDS > self.LLM_RETRY_MAX_BACKOFF_SECONDS:
            raise ValueError(
                "LLM_RETRY_BASE_BACKOFF_SECONDS cannot exceed "
                "LLM_RETRY_MAX_BACKOFF_SECONDS"
            )
        if not self.is_production:
            return self
        if self.DEBUG:
            raise ValueError("DEBUG must be false in production")
        secret = self.JWT_SECRET_KEY.strip().lower()
        if "change-me" in secret:
            raise ValueError(
                "JWT_SECRET_KEY must not use the example development value"
            )
        if not self.DATABASE_URL.startswith("postgresql"):
            raise ValueError("DATABASE_URL must be a PostgreSQL URL in production")
        return self

    @property
    def is_production(self) -> bool:
        return self.APP_ENV.lower() in {"prod", "production"}

    @property
    def expose_debug_details(self) -> bool:
        return self.DEBUG and not self.is_production

    @property
    def hsts_enabled(self) -> bool:
        return self.SECURITY_HEADERS_ENABLED and self.is_production

    @property
    def cors_allowed_origins(self) -> list[str]:
        origins = _split_csv(self.CORS_ALLOWED_ORIGINS)
        if self.is_production or self.CORS_ALLOW_CREDENTIALS:
            return [origin for origin in origins if origin != "*"]
        return origins

    @property
    def cors_allowed_methods(self) -> list[str]:
        return [method.upper() for method in _split_csv(self.CORS_ALLOWED_METHODS)]

    @property
    def cors_allowed_headers(self) -> list[str]:
        return _split_csv(self.CORS_ALLOWED_HEADERS)

    @property
    def data_source_encryption_key_bytes(self) -> bytes:
        return decode_data_source_encryption_key(self.DATA_SOURCE_ENCRYPTION_KEY)

    def rate_limit_spec(self, scope: str) -> str:
        mapping = {
            "login": self.RATE_LIMIT_LOGIN,
            "register": self.RATE_LIMIT_REGISTER,
            "refresh": self.RATE_LIMIT_REFRESH,
            "forgot-password": self.RATE_LIMIT_PASSWORD_RESET,
            "reset-password": self.RATE_LIMIT_PASSWORD_RESET,
            "resend-verification": self.RATE_LIMIT_VERIFY_EMAIL_SEND,
            "change-password": self.RATE_LIMIT_CHANGE_PASSWORD,
            "test-connection": self.RATE_LIMIT_TEST_CONNECTION,
            "metadata-search": self.RATE_LIMIT_METADATA_SEARCH,
            "metadata-sync": self.RATE_LIMIT_METADATA_SYNC,
            "sample-data": self.RATE_LIMIT_SAMPLE_DATA,
            "ai-chat": self.RATE_LIMIT_AI_CHAT,
            "mcp-query": self.RATE_LIMIT_MCP_QUERY,
            "mcp-sample": self.RATE_LIMIT_MCP_SAMPLE,
            "mcp-metadata": self.RATE_LIMIT_MCP_METADATA,
        }
        try:
            return mapping[scope]
        except KeyError as exc:
            raise ValueError(f"Unknown rate limit scope: {scope}") from exc


settings = Settings()  # type: ignore[call-arg]
