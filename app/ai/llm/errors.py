"""LLM infrastructure error hierarchy with stable machine-readable codes."""

from __future__ import annotations

from enum import Enum

from app.core.logging import redact_secret


class LLMErrorCode(str, Enum):
    LLM_CONFIGURATION_ERROR = "LLM_CONFIGURATION_ERROR"
    LLM_AUTHENTICATION_ERROR = "LLM_AUTHENTICATION_ERROR"
    LLM_RATE_LIMITED = "LLM_RATE_LIMITED"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_PROVIDER_ERROR = "LLM_PROVIDER_ERROR"
    LLM_INVALID_REQUEST = "LLM_INVALID_REQUEST"
    LLM_INTERNAL_ERROR = "LLM_INTERNAL_ERROR"
    LLM_RESPONSE_VALIDATION_ERROR = "LLM_RESPONSE_VALIDATION_ERROR"


def _sanitize(message: str) -> str:
    return redact_secret(message)


class LLMError(Exception):
    """Base error for the LLM infrastructure layer."""

    code: LLMErrorCode = LLMErrorCode.LLM_INTERNAL_ERROR

    def __init__(
        self,
        message: str = "LLM request failed",
        *,
        code: LLMErrorCode | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(_sanitize(message))
        if code is not None:
            self.code = code
        self.request_id = request_id
        self.retry_after = retry_after


class LLMConfigurationError(LLMError):
    code = LLMErrorCode.LLM_CONFIGURATION_ERROR

    def __init__(self, message: str = "LLM is not configured") -> None:
        super().__init__(message, code=self.code)


class LLMAuthenticationError(LLMError):
    code = LLMErrorCode.LLM_AUTHENTICATION_ERROR

    def __init__(self, message: str = "LLM authentication failed") -> None:
        super().__init__(message, code=self.code)


class LLMRateLimitError(LLMError):
    code = LLMErrorCode.LLM_RATE_LIMITED

    def __init__(
        self,
        message: str = "LLM rate limit exceeded",
        *,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, code=self.code, retry_after=retry_after)


class LLMTimeoutError(LLMError):
    code = LLMErrorCode.LLM_TIMEOUT

    def __init__(self, message: str = "LLM request timed out") -> None:
        super().__init__(message, code=self.code)


class LLMProviderError(LLMError):
    code = LLMErrorCode.LLM_PROVIDER_ERROR

    def __init__(self, message: str = "LLM provider request failed") -> None:
        super().__init__(message, code=self.code)


class LLMInvalidRequestError(LLMError):
    code = LLMErrorCode.LLM_INVALID_REQUEST

    def __init__(self, message: str = "LLM request is invalid") -> None:
        super().__init__(message, code=self.code)


class LLMInternalError(LLMError):
    code = LLMErrorCode.LLM_INTERNAL_ERROR

    def __init__(self, message: str = "LLM internal error") -> None:
        super().__init__(message, code=self.code)


class LLMResponseValidationError(LLMError):
    code = LLMErrorCode.LLM_RESPONSE_VALIDATION_ERROR

    def __init__(
        self,
        message: str = "LLM provider returned an invalid response",
        *,
        raw_content: str | None = None,
    ) -> None:
        super().__init__(message, code=self.code)
        self.raw_content = raw_content
