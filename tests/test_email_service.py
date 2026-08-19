import importlib
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from app.core.config import Settings, settings
from app.services.email_service import EmailService


def test_console_provider_queues_without_smtp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "EMAIL_PROVIDER", "console")
    monkeypatch.setattr(settings, "EMAIL_FROM", "noreply@analyticcastle.local")
    monkeypatch.setattr(settings, "FRONTEND_URL", "http://localhost:3000")
    service = EmailService()

    service.send_verification_email("user@example.com", "raw-token")

    assert len(service.outbox) == 1
    assert "token=raw-token" in service.outbox[0].body
    assert service.outbox[0].to == "user@example.com"


def test_smtp_provider_sends_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "EMAIL_PROVIDER", "smtp")
    monkeypatch.setattr(settings, "EMAIL_FROM", "noreply@example.com")
    monkeypatch.setattr(settings, "EMAIL_FROM_NAME", "AnalyticCastle")
    monkeypatch.setattr(settings, "SMTP_HOST", "smtp.example.com")
    monkeypatch.setattr(settings, "SMTP_PORT", 587)
    monkeypatch.setattr(settings, "SMTP_USERNAME", "smtp-user")
    monkeypatch.setattr(settings, "SMTP_PASSWORD", "smtp-pass")
    monkeypatch.setattr(settings, "SMTP_USE_TLS", True)
    monkeypatch.setattr(settings, "SMTP_TIMEOUT_SECONDS", 10.0)

    smtp_client = MagicMock()
    smtp_cm = MagicMock()
    smtp_cm.__enter__.return_value = smtp_client
    smtp_cm.__exit__.return_value = False
    smtp_ctor = MagicMock(return_value=smtp_cm)
    email_module = importlib.import_module("app.services.email_service")
    monkeypatch.setattr(email_module.smtplib, "SMTP", smtp_ctor)

    service = EmailService()
    service.send_password_reset_email("user@example.com", "reset-token")

    assert len(service.outbox) == 1
    smtp_ctor.assert_called_once_with("smtp.example.com", 587, timeout=10.0)
    smtp_client.starttls.assert_called_once()
    smtp_client.login.assert_called_once_with("smtp-user", "smtp-pass")
    smtp_client.send_message.assert_called_once()
    sent = smtp_client.send_message.call_args.args[0]
    assert sent["To"] == "user@example.com"
    assert "AnalyticCastle <noreply@example.com>" == sent["From"]
    assert "token=reset-token" in sent.get_content()


def test_smtp_provider_requires_smtp_settings() -> None:
    with pytest.raises(ValidationError):
        Settings(
            APP_ENV="local",
            DATABASE_URL=settings.DATABASE_URL,
            JWT_SECRET_KEY=settings.JWT_SECRET_KEY,
            EMAIL_PROVIDER="smtp",
            SMTP_HOST="",
            SMTP_USERNAME="",
            SMTP_PASSWORD="",
        )


def test_settings_strip_quoted_smtp_password() -> None:
    configured = Settings(
        APP_ENV="local",
        DATABASE_URL=settings.DATABASE_URL,
        JWT_SECRET_KEY=settings.JWT_SECRET_KEY,
        EMAIL_PROVIDER="smtp",
        SMTP_HOST="smtp.example.com",
        SMTP_USERNAME="user@example.com",
        SMTP_PASSWORD='"quoted-pass"',
    )

    assert configured.SMTP_PASSWORD == "quoted-pass"
