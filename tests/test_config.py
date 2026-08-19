import base64

import pytest
from pydantic import ValidationError

from app.core.config import Settings, settings


def test_production_rejects_debug_mode() -> None:
    with pytest.raises(ValidationError):
        Settings(
            APP_ENV="production",
            DEBUG=True,
            DATABASE_URL=settings.DATABASE_URL,
            JWT_SECRET_KEY="production-secret-key-that-is-long-enough",
        )


def test_production_rejects_example_jwt_secret() -> None:
    with pytest.raises(ValidationError):
        Settings(
            APP_ENV="production",
            DEBUG=False,
            DATABASE_URL=settings.DATABASE_URL,
            JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
        )


def test_production_rejects_non_postgres_database_url() -> None:
    with pytest.raises(ValidationError):
        Settings(
            APP_ENV="production",
            DEBUG=False,
            DATABASE_URL="sqlite:///./app.db",
            JWT_SECRET_KEY="production-secret-key-that-is-long-enough",
        )


def test_local_development_allows_debug() -> None:
    configured = Settings(
        APP_ENV="local",
        DEBUG=True,
        DATABASE_URL=settings.DATABASE_URL,
        JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
    )

    assert configured.expose_debug_details is True
    assert configured.is_production is False


def test_production_accepts_explicit_secure_settings() -> None:
    configured = Settings(
        APP_ENV="production",
        DEBUG=False,
        DATABASE_URL=settings.DATABASE_URL,
        JWT_SECRET_KEY="production-secret-key-that-is-long-enough",
    )

    assert configured.is_production is True
    assert configured.expose_debug_details is False


def test_postgres_connect_timeout_is_positive() -> None:
    configured = Settings(
        APP_ENV="local",
        DATABASE_URL=settings.DATABASE_URL,
        JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
        POSTGRES_CONNECT_TIMEOUT=10.0,
    )

    assert configured.POSTGRES_CONNECT_TIMEOUT == 10.0
    assert settings.POSTGRES_CONNECT_TIMEOUT > 0


def test_postgres_connect_timeout_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(
            APP_ENV="local",
            DATABASE_URL=settings.DATABASE_URL,
            JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
            POSTGRES_CONNECT_TIMEOUT=0,
        )


def test_postgres_discovery_timeout_has_default() -> None:
    configured = Settings(
        APP_ENV="local",
        DATABASE_URL=settings.DATABASE_URL,
        JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
    )

    assert configured.POSTGRES_DISCOVERY_TIMEOUT == 30.0
    assert settings.POSTGRES_DISCOVERY_TIMEOUT > 0


def test_postgres_discovery_timeout_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(
            APP_ENV="local",
            DATABASE_URL=settings.DATABASE_URL,
            JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
            POSTGRES_DISCOVERY_TIMEOUT=0,
        )


def test_encryption_key_is_required() -> None:
    with pytest.raises(ValidationError):
        Settings(
            APP_ENV="local",
            DATABASE_URL=settings.DATABASE_URL,
            JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
            DATA_SOURCE_ENCRYPTION_KEY="",
        )


@pytest.mark.parametrize(
    "value",
    ["secret", "changeme", "default-key", "development-key", "change-me"],
)
def test_encryption_key_rejects_placeholders(value: str) -> None:
    with pytest.raises(ValidationError):
        Settings(
            APP_ENV="local",
            DATABASE_URL=settings.DATABASE_URL,
            JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
            DATA_SOURCE_ENCRYPTION_KEY=value,
        )


def test_encryption_key_must_decode_to_32_bytes() -> None:
    with pytest.raises(ValidationError):
        Settings(
            APP_ENV="local",
            DATABASE_URL=settings.DATABASE_URL,
            JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
            DATA_SOURCE_ENCRYPTION_KEY="not-a-valid-32-byte-key-value!!!!",
        )


def test_encryption_key_must_differ_from_jwt_secret() -> None:
    shared = "aa" * 32
    with pytest.raises(ValidationError, match="independent from JWT_SECRET_KEY"):
        Settings(
            APP_ENV="local",
            DATABASE_URL=settings.DATABASE_URL,
            JWT_SECRET_KEY=shared,
            DATA_SOURCE_ENCRYPTION_KEY=shared,
        )


def test_encryption_key_must_differ_from_database_url() -> None:
    key = "bb" * 32
    with pytest.raises(ValidationError, match="independent from DATABASE_URL"):
        Settings(
            APP_ENV="local",
            DATABASE_URL=key,
            JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
            DATA_SOURCE_ENCRYPTION_KEY=key,
        )


def test_encryption_key_accepts_hex_and_urlsafe_base64() -> None:
    hex_key = "cc" * 32
    configured = Settings(
        APP_ENV="local",
        DATABASE_URL=settings.DATABASE_URL,
        JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
        DATA_SOURCE_ENCRYPTION_KEY=hex_key,
    )
    assert configured.data_source_encryption_key_bytes == bytes.fromhex(hex_key)

    raw = bytes.fromhex("dd" * 32)
    b64_key = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    configured_b64 = Settings(
        APP_ENV="local",
        DATABASE_URL=settings.DATABASE_URL,
        JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
        DATA_SOURCE_ENCRYPTION_KEY=b64_key,
    )
    assert configured_b64.data_source_encryption_key_bytes == raw


def test_settings_repr_omits_secrets() -> None:
    rendered = repr(settings)

    assert settings.DATA_SOURCE_ENCRYPTION_KEY not in rendered
    assert settings.JWT_SECRET_KEY not in rendered
    assert settings.DATABASE_URL not in rendered
    assert "DATA_SOURCE_ENCRYPTION_KEY" not in rendered
    assert "LLM_API_KEY" not in rendered


def test_metadata_discovery_limits_have_defaults() -> None:
    configured = Settings(
        APP_ENV="local",
        DATABASE_URL=settings.DATABASE_URL,
        JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
    )

    assert configured.METADATA_DISCOVERY_MAX_SCHEMAS == 100
    assert configured.METADATA_DISCOVERY_MAX_TABLES == 2_000
    assert configured.METADATA_DISCOVERY_MAX_COLUMNS == 20_000
    assert configured.METADATA_DISCOVERY_MAX_RELATIONSHIPS == 5_000


def test_metadata_discovery_limits_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(
            APP_ENV="local",
            DATABASE_URL=settings.DATABASE_URL,
            JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
            METADATA_DISCOVERY_MAX_SCHEMAS=0,
        )


def test_metadata_search_limits_have_defaults() -> None:
    configured = Settings(
        APP_ENV="local",
        DATABASE_URL=settings.DATABASE_URL,
        JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
    )

    assert configured.METADATA_SEARCH_DEFAULT_LIMIT == 50
    assert configured.METADATA_SEARCH_MAX_LIMIT == 100


def test_metadata_search_limits_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(
            APP_ENV="local",
            DATABASE_URL=settings.DATABASE_URL,
            JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
            METADATA_SEARCH_DEFAULT_LIMIT=0,
        )


def test_metadata_search_default_limit_cannot_exceed_max() -> None:
    with pytest.raises(ValidationError):
        Settings(
            APP_ENV="local",
            DATABASE_URL=settings.DATABASE_URL,
            JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
            METADATA_SEARCH_DEFAULT_LIMIT=100,
            METADATA_SEARCH_MAX_LIMIT=50,
        )


def test_sample_data_limits_have_defaults() -> None:
    configured = Settings(
        APP_ENV="local",
        DATABASE_URL=settings.DATABASE_URL,
        JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
    )

    assert configured.SAMPLE_DATA_DEFAULT_LIMIT == 10
    assert configured.SAMPLE_DATA_MAX_LIMIT == 100
    assert configured.SAMPLE_DATA_MAX_COLUMNS == 100
    assert configured.SAMPLE_DATA_MAX_VALUE_CHARS == 1_024
    assert configured.SAMPLE_DATA_MAX_JSON_CHARS == 4_096


def test_sample_data_limits_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(
            APP_ENV="local",
            DATABASE_URL=settings.DATABASE_URL,
            JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
            SAMPLE_DATA_DEFAULT_LIMIT=0,
        )


def test_sample_data_default_limit_cannot_exceed_max() -> None:
    with pytest.raises(ValidationError):
        Settings(
            APP_ENV="local",
            DATABASE_URL=settings.DATABASE_URL,
            JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
            SAMPLE_DATA_DEFAULT_LIMIT=100,
            SAMPLE_DATA_MAX_LIMIT=50,
        )


def test_metadata_api_page_size_defaults() -> None:
    configured = Settings(
        APP_ENV="local",
        DATABASE_URL=settings.DATABASE_URL,
        JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
    )

    assert configured.METADATA_API_DEFAULT_PAGE_SIZE == 50
    assert configured.METADATA_API_MAX_PAGE_SIZE == 100
    assert configured.METADATA_SEARCH_MAX_QUERY_LENGTH == 255


def test_metadata_api_page_size_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(
            APP_ENV="local",
            DATABASE_URL=settings.DATABASE_URL,
            JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
            METADATA_API_DEFAULT_PAGE_SIZE=0,
        )


def test_metadata_api_default_page_size_cannot_exceed_max() -> None:
    with pytest.raises(ValidationError):
        Settings(
            APP_ENV="local",
            DATABASE_URL=settings.DATABASE_URL,
            JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
            METADATA_API_DEFAULT_PAGE_SIZE=100,
            METADATA_API_MAX_PAGE_SIZE=50,
        )


def test_metadata_search_query_length_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(
            APP_ENV="local",
            DATABASE_URL=settings.DATABASE_URL,
            JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
            METADATA_SEARCH_MAX_QUERY_LENGTH=0,
        )


def test_metadata_api_rate_limit_defaults() -> None:
    configured = Settings(
        APP_ENV="local",
        DATABASE_URL=settings.DATABASE_URL,
        JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
    )

    assert configured.RATE_LIMIT_METADATA_SEARCH == "30/minute"
    assert configured.RATE_LIMIT_METADATA_SYNC == "5/minute"
    assert configured.RATE_LIMIT_SAMPLE_DATA == "10/minute"
    assert configured.rate_limit_spec("metadata-search") == "30/minute"
    assert configured.rate_limit_spec("metadata-sync") == "5/minute"
    assert configured.rate_limit_spec("sample-data") == "10/minute"


def test_ai_limits_have_defaults() -> None:
    configured = Settings(
        APP_ENV="local",
        DATABASE_URL=settings.DATABASE_URL,
        JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
    )

    assert configured.LLM_PROVIDER == "openai"
    assert configured.LLM_MODEL == "gpt-4o-mini"
    assert configured.LLM_TIMEOUT_SECONDS == 30.0
    assert configured.LLM_TEMPERATURE == 0.2
    assert configured.LLM_MAX_OUTPUT_TOKENS == 1_024
    assert configured.AI_MAX_MESSAGE_CHARS == 4_000
    assert configured.AI_MAX_CONTEXT_CHARS == 8_000
    assert configured.AI_MAX_OUTPUT_CHARS == 8_000
    assert configured.AI_METADATA_SEARCH_LIMIT == 10
    assert configured.AI_METADATA_RESOLVE_SEARCH_LIMIT == 50
    assert configured.AI_MAX_METADATA_TABLES == 20
    assert configured.AI_MAX_METADATA_COLUMNS == 50
    assert configured.AI_MAX_METADATA_RELATIONSHIPS == 30
    assert configured.AI_MAX_RESULT_LIMIT == 100
    assert configured.AI_MAX_PLAN_METRICS == 10
    assert configured.AI_MAX_PLAN_DIMENSIONS == 10
    assert configured.AI_MAX_PLAN_FILTERS == 20
    assert configured.AI_MAX_FILTER_VALUES == 25
    assert configured.AI_MAX_CONCEPT_CHARS == 128
    assert configured.RATE_LIMIT_AI_CHAT == "10/minute"
    assert configured.rate_limit_spec("ai-chat") == "10/minute"


def test_ai_timeout_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(
            APP_ENV="local",
            DATABASE_URL=settings.DATABASE_URL,
            JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
            LLM_TIMEOUT_SECONDS=0,
        )


def test_ai_message_limit_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(
            APP_ENV="local",
            DATABASE_URL=settings.DATABASE_URL,
            JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
            AI_MAX_MESSAGE_CHARS=0,
        )


def test_llm_provider_must_be_supported() -> None:
    with pytest.raises(ValidationError):
        Settings(
            APP_ENV="local",
            DATABASE_URL=settings.DATABASE_URL,
            JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
            LLM_PROVIDER="langchain",
        )
