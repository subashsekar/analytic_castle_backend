"""Prompt system error hierarchy with stable machine-readable codes."""

from __future__ import annotations

from enum import Enum

from app.core.logging import redact_secret


class PromptErrorCode(str, Enum):
    PROMPT_VALIDATION_ERROR = "PROMPT_VALIDATION_ERROR"
    PROMPT_RENDER_ERROR = "PROMPT_RENDER_ERROR"
    PROMPT_NOT_FOUND = "PROMPT_NOT_FOUND"
    PROMPT_VERSION_ERROR = "PROMPT_VERSION_ERROR"


def _sanitize(message: str) -> str:
    return redact_secret(message)


class PromptError(Exception):
    """Base error for the prompt infrastructure layer."""

    code: PromptErrorCode = PromptErrorCode.PROMPT_VALIDATION_ERROR

    def __init__(
        self,
        message: str = "Prompt operation failed",
        *,
        code: PromptErrorCode | None = None,
    ) -> None:
        super().__init__(_sanitize(message))
        if code is not None:
            self.code = code


class PromptValidationError(PromptError):
    code = PromptErrorCode.PROMPT_VALIDATION_ERROR

    def __init__(self, message: str = "Prompt template is invalid") -> None:
        super().__init__(message, code=self.code)


class PromptRenderError(PromptError):
    code = PromptErrorCode.PROMPT_RENDER_ERROR

    def __init__(self, message: str = "Prompt rendering failed") -> None:
        super().__init__(message, code=self.code)


class PromptNotFoundError(PromptError):
    code = PromptErrorCode.PROMPT_NOT_FOUND

    def __init__(self, message: str = "Prompt was not found") -> None:
        super().__init__(message, code=self.code)


class PromptVersionError(PromptError):
    code = PromptErrorCode.PROMPT_VERSION_ERROR

    def __init__(self, message: str = "Prompt version is invalid") -> None:
        super().__init__(message, code=self.code)
