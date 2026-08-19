from __future__ import annotations

from app.core.logging import redact_secret


def sanitize_ai_message(message: str) -> str:
    """Strip secrets from AI error messages before they reach logs or clients."""
    return redact_secret(message)


class AIError(Exception):
    """Base error for AI analyst failures.

    Messages are sanitized so API keys and other secrets never reach clients.
    """

    def __init__(self, message: str = "AI request failed") -> None:
        super().__init__(sanitize_ai_message(message))


class AIConfigurationError(AIError):
    """The LLM provider is missing required configuration."""

    def __init__(self, message: str = "AI provider is not configured") -> None:
        super().__init__(message)


class AIProviderError(AIError):
    """The LLM provider request failed."""

    def __init__(self, message: str = "AI provider request failed") -> None:
        super().__init__(message)


class AIProviderTimeoutError(AIProviderError):
    """The LLM provider did not respond in time."""

    def __init__(self, message: str = "AI provider timed out") -> None:
        super().__init__(message)


class AIProviderRateLimitError(AIProviderError):
    """The LLM provider rejected the request because of rate limiting."""

    def __init__(self, message: str = "AI provider rate limit exceeded") -> None:
        super().__init__(message)


class AIProviderAuthenticationError(AIProviderError):
    """The LLM provider rejected the configured credentials."""

    def __init__(self, message: str = "AI provider authentication failed") -> None:
        super().__init__(message)


class AIResponseValidationError(AIError):
    """The LLM response was missing, oversized, or malformed."""

    def __init__(
        self,
        message: str = "AI provider returned an invalid response",
        *,
        raw_content: str | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_content = raw_content


class AIRequestValidationError(AIError):
    """The AI request exceeded configured limits or was incomplete."""

    def __init__(self, message: str = "AI request is invalid") -> None:
        super().__init__(message)


class AIContextError(AIError):
    """The AI request context is missing or does not match the data source."""

    def __init__(self, message: str = "AI request context is invalid") -> None:
        super().__init__(message)
