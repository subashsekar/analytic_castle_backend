"""Anomaly detection agent error hierarchy with stable machine-readable codes."""

from __future__ import annotations

from enum import Enum

from app.core.logging import redact_secret


class AnomalyDetectionErrorCode(str, Enum):
    ANOMALY_DETECTION_CONFIGURATION_ERROR = "ANOMALY_DETECTION_CONFIGURATION_ERROR"
    ANOMALY_DETECTION_AUTHORIZATION_ERROR = "ANOMALY_DETECTION_AUTHORIZATION_ERROR"
    ANOMALY_DETECTION_VALIDATION_ERROR = "ANOMALY_DETECTION_VALIDATION_ERROR"
    ANOMALY_DETECTION_LLM_ERROR = "ANOMALY_DETECTION_LLM_ERROR"
    ANOMALY_DETECTION_INTERNAL_ERROR = "ANOMALY_DETECTION_INTERNAL_ERROR"


def _sanitize(message: str) -> str:
    return redact_secret(message)


class AnomalyDetectionError(Exception):
    """Base error for the anomaly detection agent."""

    code: AnomalyDetectionErrorCode = AnomalyDetectionErrorCode.ANOMALY_DETECTION_INTERNAL_ERROR

    def __init__(
        self,
        message: str = "Anomaly detection request failed",
        *,
        code: AnomalyDetectionErrorCode | None = None,
    ) -> None:
        super().__init__(_sanitize(message))
        if code is not None:
            self.code = code


class AnomalyDetectionConfigurationError(AnomalyDetectionError):
    code = AnomalyDetectionErrorCode.ANOMALY_DETECTION_CONFIGURATION_ERROR

    def __init__(self, message: str = "Anomaly detection is not configured") -> None:
        super().__init__(message, code=self.code)


class AnomalyDetectionAuthorizationError(AnomalyDetectionError):
    code = AnomalyDetectionErrorCode.ANOMALY_DETECTION_AUTHORIZATION_ERROR

    def __init__(self, message: str = "Anomaly detection authorization failed") -> None:
        super().__init__(message, code=self.code)


class AnomalyDetectionValidationError(AnomalyDetectionError):
    code = AnomalyDetectionErrorCode.ANOMALY_DETECTION_VALIDATION_ERROR

    def __init__(self, message: str = "Anomaly detection output validation failed") -> None:
        super().__init__(message, code=self.code)


class AnomalyDetectionLLMError(AnomalyDetectionError):
    code = AnomalyDetectionErrorCode.ANOMALY_DETECTION_LLM_ERROR

    def __init__(self, message: str = "Anomaly detection LLM request failed") -> None:
        super().__init__(message, code=self.code)
