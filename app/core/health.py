from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.db.session import engine

logger = logging.getLogger(__name__)


def check_database(*, timeout_seconds: float | None = None) -> bool:
    """Return True when a lightweight ``SELECT 1`` succeeds within the timeout."""
    timeout = (
        settings.HEALTH_CHECK_TIMEOUT_SECONDS
        if timeout_seconds is None
        else timeout_seconds
    )
    timeout_ms = str(max(1, int(timeout * 1000)))
    try:
        with engine.connect() as connection:
            connection.execute(
                text("SELECT set_config('statement_timeout', :timeout, true)"),
                {"timeout": timeout_ms},
            )
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        logger.warning(
            "Database health check failed error_type=%s",
            type(exc).__name__,
        )
        return False
    return True
