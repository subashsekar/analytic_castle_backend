from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage as SmtpEmailMessage

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmailMessage:
    to: str
    subject: str
    body: str
    sender: str


class EmailService:
    """Delivers auth emails via console logging or SMTP."""

    def __init__(self) -> None:
        self.outbox: list[EmailMessage] = []

    def send_verification_email(self, to: str, token: str) -> None:
        link = f"{settings.FRONTEND_URL.rstrip('/')}/verify-email?token={token}"
        expire_minutes = settings.EMAIL_VERIFICATION_EXPIRE_MINUTES
        self._deliver(
            EmailMessage(
                to=to,
                subject=f"Verify your {settings.APP_NAME} email",
                body=(
                    f"Please verify your {settings.APP_NAME} email address.\n\n"
                    f"Open this link to verify your account:\n{link}\n\n"
                    f"This link expires in {expire_minutes} minutes."
                ),
                sender=settings.EMAIL_FROM,
            )
        )

    def send_password_reset_email(self, to: str, token: str) -> None:
        link = f"{settings.FRONTEND_URL.rstrip('/')}/reset-password?token={token}"
        expire_minutes = settings.PASSWORD_RESET_EXPIRE_MINUTES
        self._deliver(
            EmailMessage(
                to=to,
                subject=f"Reset your {settings.APP_NAME} password",
                body=(
                    f"We received a request to reset your {settings.APP_NAME} password.\n\n"
                    f"Open this link to choose a new password:\n{link}\n\n"
                    f"This link expires in {expire_minutes} minutes.\n"
                    "If you did not request this, you can ignore this email."
                ),
                sender=settings.EMAIL_FROM,
            )
        )

    def _from_header(self) -> str:
        name = settings.EMAIL_FROM_NAME.strip()
        address = settings.EMAIL_FROM.strip()
        if name:
            return f"{name} <{address}>"
        return address

    def _deliver(self, message: EmailMessage) -> None:
        self.outbox.append(message)
        if settings.EMAIL_PROVIDER == "smtp":
            self._send_smtp(message)
            logger.info("Email sent via SMTP to %s (%s)", message.to, message.subject)
            return
        logger.info("Email queued for %s (%s)", message.to, message.subject)

    def _send_smtp(self, message: EmailMessage) -> None:
        email = SmtpEmailMessage()
        email["Subject"] = message.subject
        email["From"] = self._from_header()
        email["To"] = message.to
        email.set_content(message.body)

        with smtplib.SMTP(
            settings.SMTP_HOST,
            settings.SMTP_PORT,
            timeout=settings.SMTP_TIMEOUT_SECONDS,
        ) as client:
            if settings.SMTP_USE_TLS:
                client.starttls()
            client.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            client.send_message(email)


email_service = EmailService()
