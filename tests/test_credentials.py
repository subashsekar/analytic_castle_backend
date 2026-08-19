from __future__ import annotations

import logging
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.connectors.types import ConnectorConfig
from app.core.config import settings
from app.core.logging import RedactingFilter, redact_secret
from app.db.models import (
    DataSource,
    DataSourceConnection,
    DataSourceType,
    Organization,
    User,
    UserRole,
    Workspace,
)
from app.schemas import DataSourceConnectionRead, DataSourceRead
from app.services.credentials import (
    CIPHERTEXT_VERSION,
    MAX_SECRET_LENGTH,
    CredentialDecryptionError,
    CredentialEncryptionError,
    InvalidEncryptionKeyError,
    connector_config_from_connection,
    decrypt_secret,
    encrypt_secret,
)

PLAINTEXT = "password123"
CUSTOMER_PASSWORD = "CustomerDbPassword!@# 42"
KEY_A = bytes.fromhex("11" * 32)
KEY_B = bytes.fromhex("22" * 32)


def _user() -> User:
    return User(
        first_name="Ada",
        last_name="Lovelace",
        email=f"ada-{uuid.uuid4().hex[:8]}@example.com",
        password_hash="hashed-password",
        role=UserRole.USER,
    )


def _seed_connection(
    db_session: Session,
    *,
    encrypted_password: str,
) -> DataSourceConnection:
    user = _user()
    organization = Organization(
        name="AnalyticCastle",
        slug=f"analyticcastle-{uuid.uuid4().hex[:8]}",
    )
    db_session.add_all([user, organization])
    db_session.flush()
    workspace = Workspace(
        organization_id=organization.id,
        name="Analytics",
        slug=f"analytics-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(workspace)
    db_session.flush()
    data_source = DataSource(
        workspace_id=workspace.id,
        name="Production Analytics DB",
        type=DataSourceType.POSTGRESQL,
        created_by=user.id,
    )
    db_session.add(data_source)
    db_session.flush()
    connection = DataSourceConnection(
        data_source_id=data_source.id,
        host="db.internal.example",
        database_name="analytics",
        username="readonly",
        encrypted_password=encrypted_password,
    )
    db_session.add(connection)
    db_session.flush()
    return connection


def test_encrypt_then_decrypt_round_trip() -> None:
    ciphertext = encrypt_secret(PLAINTEXT, key=KEY_A)

    assert ciphertext != PLAINTEXT
    assert ciphertext.startswith(f"{CIPHERTEXT_VERSION}:")
    assert decrypt_secret(ciphertext, key=KEY_A) == PLAINTEXT


def test_same_plaintext_produces_different_ciphertext() -> None:
    first = encrypt_secret(PLAINTEXT, key=KEY_A)
    second = encrypt_secret(PLAINTEXT, key=KEY_A)

    assert first != second
    assert decrypt_secret(first, key=KEY_A) == PLAINTEXT
    assert decrypt_secret(second, key=KEY_A) == PLAINTEXT


def test_ciphertext_uses_versioned_format() -> None:
    ciphertext = encrypt_secret(PLAINTEXT, key=KEY_A)
    parts = ciphertext.split(":")

    assert len(parts) == 3
    assert parts[0] == CIPHERTEXT_VERSION
    assert parts[1]
    assert parts[2]


def test_modified_ciphertext_fails() -> None:
    ciphertext = encrypt_secret(PLAINTEXT, key=KEY_A)
    version, nonce, payload = ciphertext.split(":")
    flipped = "A" if payload[0] != "A" else "B"
    modified = f"{version}:{nonce}:{flipped}{payload[1:]}"

    with pytest.raises(CredentialDecryptionError, match="Unable to decrypt"):
        decrypt_secret(modified, key=KEY_A)


def test_wrong_key_fails() -> None:
    ciphertext = encrypt_secret(PLAINTEXT, key=KEY_A)

    with pytest.raises(CredentialDecryptionError, match="Unable to decrypt"):
        decrypt_secret(ciphertext, key=KEY_B)


def test_wrong_key_does_not_return_plaintext() -> None:
    ciphertext = encrypt_secret(PLAINTEXT, key=KEY_A)

    with pytest.raises(CredentialDecryptionError) as exc_info:
        decrypt_secret(ciphertext, key=KEY_B)

    assert PLAINTEXT not in str(exc_info.value)
    assert "InvalidTag" not in str(exc_info.value)
    assert "AESGCM" not in str(exc_info.value)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "invalid",
        "v1:only-two-parts",
        "v1::payload",
        "v1:nonce:",
        "not-a-version:abc:def",
        "v1:@@@:@@@",
        "v1:nonce:payload:extra",
    ],
)
def test_malformed_ciphertext_fails(value: str) -> None:
    with pytest.raises(CredentialDecryptionError, match="Unable to decrypt"):
        decrypt_secret(value, key=KEY_A)


def test_unsupported_version_fails() -> None:
    ciphertext = encrypt_secret(PLAINTEXT, key=KEY_A)
    _, nonce, payload = ciphertext.split(":")
    upgraded = f"v2:{nonce}:{payload}"

    with pytest.raises(CredentialDecryptionError, match="Unable to decrypt"):
        decrypt_secret(upgraded, key=KEY_A)


def test_decrypt_does_not_treat_invalid_input_as_plaintext() -> None:
    with pytest.raises(CredentialDecryptionError):
        decrypt_secret("invalid", key=KEY_A)


def test_empty_plaintext_is_rejected() -> None:
    with pytest.raises(CredentialEncryptionError, match="empty"):
        encrypt_secret("", key=KEY_A)


def test_none_inputs_are_rejected() -> None:
    with pytest.raises(CredentialEncryptionError):
        encrypt_secret(None, key=KEY_A)  # type: ignore[arg-type]
    with pytest.raises(CredentialDecryptionError):
        decrypt_secret(None, key=KEY_A)  # type: ignore[arg-type]


def test_plaintext_over_max_length_is_rejected() -> None:
    with pytest.raises(CredentialEncryptionError, match="maximum length"):
        encrypt_secret("x" * (MAX_SECRET_LENGTH + 1), key=KEY_A)


def test_max_length_secret_fits_column() -> None:
    ciphertext = encrypt_secret("x" * MAX_SECRET_LENGTH, key=KEY_A)
    column_length = DataSourceConnection.__table__.c.encrypted_password.type.length

    assert column_length is not None
    assert len(ciphertext) <= column_length
    assert decrypt_secret(ciphertext, key=KEY_A) == "x" * MAX_SECRET_LENGTH


def test_invalid_key_length_is_rejected() -> None:
    ciphertext = encrypt_secret(PLAINTEXT, key=KEY_A)
    with pytest.raises(InvalidEncryptionKeyError):
        encrypt_secret(PLAINTEXT, key=b"too-short")
    with pytest.raises(InvalidEncryptionKeyError):
        decrypt_secret(ciphertext, key=b"too-short")


def test_encrypt_uses_configured_key_by_default() -> None:
    ciphertext = encrypt_secret(CUSTOMER_PASSWORD)

    assert decrypt_secret(ciphertext) == CUSTOMER_PASSWORD
    assert ciphertext != CUSTOMER_PASSWORD


def test_stored_password_is_not_plaintext(db_session: Session) -> None:
    ciphertext = encrypt_secret(CUSTOMER_PASSWORD)
    connection = _seed_connection(db_session, encrypted_password=ciphertext)
    db_session.expire_all()

    loaded = db_session.scalar(
        select(DataSourceConnection).where(DataSourceConnection.id == connection.id)
    )
    assert loaded is not None
    assert loaded.encrypted_password != CUSTOMER_PASSWORD
    assert loaded.encrypted_password == ciphertext
    assert "password" not in DataSourceConnection.__table__.columns


def test_stored_ciphertext_decrypts_to_original_password(
    db_session: Session,
) -> None:
    ciphertext = encrypt_secret(CUSTOMER_PASSWORD)
    connection = _seed_connection(db_session, encrypted_password=ciphertext)
    db_session.expire_all()

    loaded = db_session.scalar(
        select(DataSourceConnection).where(DataSourceConnection.id == connection.id)
    )
    assert loaded is not None
    assert decrypt_secret(loaded.encrypted_password) == CUSTOMER_PASSWORD


def test_connector_config_is_built_from_decrypted_password(
    db_session: Session,
) -> None:
    ciphertext = encrypt_secret(CUSTOMER_PASSWORD)
    connection = _seed_connection(db_session, encrypted_password=ciphertext)

    config = connector_config_from_connection(connection)

    assert isinstance(config, ConnectorConfig)
    assert config.credential == CUSTOMER_PASSWORD
    assert config.host == connection.host
    assert config.port == connection.port
    assert config.database_name == connection.database_name
    assert config.username == connection.username
    assert config.ssl_mode == connection.ssl_mode
    assert CUSTOMER_PASSWORD not in repr(config)
    assert "credential" not in repr(config)
    assert connection.encrypted_password != CUSTOMER_PASSWORD


def test_decrypted_password_is_not_written_back_to_the_model(
    db_session: Session,
) -> None:
    ciphertext = encrypt_secret(CUSTOMER_PASSWORD)
    connection = _seed_connection(db_session, encrypted_password=ciphertext)
    config = connector_config_from_connection(connection)
    db_session.flush()
    db_session.refresh(connection)

    assert config.credential == CUSTOMER_PASSWORD
    assert connection.encrypted_password == ciphertext
    assert connection.encrypted_password != config.credential


def test_api_schemas_never_expose_credentials(db_session: Session) -> None:
    ciphertext = encrypt_secret(CUSTOMER_PASSWORD)
    connection = _seed_connection(db_session, encrypted_password=ciphertext)

    source_payload = DataSourceRead.model_validate(connection.data_source).model_dump()
    connection_payload = DataSourceConnectionRead.model_validate(
        connection
    ).model_dump()

    for payload in (source_payload, connection_payload):
        assert "password" not in payload
        assert "encrypted_password" not in payload
        assert CUSTOMER_PASSWORD not in str(payload)
        assert ciphertext not in str(payload)


def test_credentials_are_not_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("credential-security-test")
    logger.addFilter(RedactingFilter())

    with caplog.at_level(logging.DEBUG):
        ciphertext = encrypt_secret(CUSTOMER_PASSWORD, key=KEY_A)
        recovered = decrypt_secret(ciphertext, key=KEY_A)
        logger.info(
            "encrypted_password="
            + ciphertext
            + " password="
            + CUSTOMER_PASSWORD
            + " DATA_SOURCE_ENCRYPTION_KEY="
            + settings.DATA_SOURCE_ENCRYPTION_KEY
        )

    assert recovered == CUSTOMER_PASSWORD
    text = caplog.text
    assert CUSTOMER_PASSWORD not in text
    assert ciphertext not in text
    assert settings.DATA_SOURCE_ENCRYPTION_KEY not in text
    assert "encrypted_password=[REDACTED]" in text
    assert "password=[REDACTED]" in text
    assert "DATA_SOURCE_ENCRYPTION_KEY=[REDACTED]" in text


def test_redaction_covers_encryption_secrets() -> None:
    key = settings.DATA_SOURCE_ENCRYPTION_KEY
    raw = (
        f"DATA_SOURCE_ENCRYPTION_KEY={key} "
        f"decrypted_password={CUSTOMER_PASSWORD} "
        f"encrypted_password=v1:nonce:payload"
    )

    redacted = redact_secret(raw)

    assert key not in redacted
    assert CUSTOMER_PASSWORD not in redacted
    assert "v1:nonce:payload" not in redacted
    assert "DATA_SOURCE_ENCRYPTION_KEY=[REDACTED]" in redacted
    assert "decrypted_password=[REDACTED]" in redacted
    assert "encrypted_password=[REDACTED]" in redacted


def test_connector_package_does_not_import_credential_service() -> None:
    from app.connectors import postgresql as postgresql_module

    assert not hasattr(postgresql_module, "encrypt_secret")
    assert not hasattr(postgresql_module, "decrypt_secret")
    assert "credentials" not in postgresql_module.__name__


def test_encrypt_secret_does_not_log_by_itself(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG):
        ciphertext = encrypt_secret(CUSTOMER_PASSWORD, key=KEY_A)
        decrypt_secret(ciphertext, key=KEY_A)

    combined = " ".join(record.getMessage() for record in caplog.records)
    assert CUSTOMER_PASSWORD not in combined
    assert ciphertext not in combined
