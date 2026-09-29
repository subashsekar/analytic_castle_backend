import logging
import re
import sys
import traceback

from app.core.request_id import get_request_id

_REDACTED = "[REDACTED]"
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)\b(encrypted_password|decrypted_password|password|passwd|password_hash|"
    r"current_password|new_password|credential|"
    r"refresh_token|access_token|reset_token|verification_token|token|"
    r"authorization|secret|jwt_secret_key|data_source_encryption_key|"
    r"llm_api_key|openrouter_api_key|openai_api_key|api_key|"
    r"database_url|connection_string|connection_uri)\b\s*[:=]\s*([^\s,;&]+)"
)
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-+=/]+")
_JWT = re.compile(r"eyJ[A-Za-z0-9_\-]+=*\.[A-Za-z0-9_\-]+=*\.[A-Za-z0-9_\-]+=*")


def redact_secret(value: str) -> str:
    redacted = _BEARER.sub(f"Bearer {_REDACTED}", value)
    redacted = _JWT.sub(_REDACTED, redacted)
    return _SENSITIVE_ASSIGNMENT.sub(
        lambda match: f"{match.group(1)}={_REDACTED}",
        redacted,
    )


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_secret(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    key: redact_secret(value) if isinstance(value, str) else value
                    for key, value in record.args.items()
                }
            else:
                record.args = tuple(
                    redact_secret(arg) if isinstance(arg, str) else arg
                    for arg in record.args
                )
        if record.exc_info:
            if record.exc_text is None:
                record.exc_text = "".join(traceback.format_exception(*record.exc_info))
            record.exc_text = redact_secret(record.exc_text)
        elif record.exc_text:
            record.exc_text = redact_secret(record.exc_text)
        return True


def configure_logging(*, debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=(
            "%(asctime)s %(levelname)s [%(name)s] request_id=%(request_id)s %(message)s"
        ),
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    context = RequestContextFilter()
    redactor = RedactingFilter()
    root = logging.getLogger()
    root.addFilter(context)
    root.addFilter(redactor)
    for handler in root.handlers:
        handler.addFilter(context)
        handler.addFilter(redactor)
