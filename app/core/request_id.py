from __future__ import annotations

import re
import uuid
from contextvars import ContextVar, Token

REQUEST_ID_HEADER = "X-Request-ID"
MISSING_REQUEST_ID = "-"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default=MISSING_REQUEST_ID)


def get_request_id() -> str:
    return _request_id_ctx.get()


def bind_request_id(request_id: str) -> Token[str]:
    return _request_id_ctx.set(request_id)


def reset_request_id(token: Token[str]) -> None:
    _request_id_ctx.reset(token)


def new_request_id() -> str:
    return str(uuid.uuid4())


def normalize_request_id(candidate: str | None) -> str:
    if candidate is None:
        return new_request_id()
    value = candidate.strip()
    if not _VALID_REQUEST_ID.fullmatch(value):
        return new_request_id()
    return value
