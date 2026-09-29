"""Anomaly Detection Agent for AnalyticCastle."""

from app.ai.anomaly_detection.detection import scan_for_anomalies, scan_prompt_payload
from app.ai.anomaly_detection.errors import (
    AnomalyDetectionError,
    AnomalyDetectionErrorCode,
    AnomalyDetectionConfigurationError,
    AnomalyDetectionAuthorizationError,
    AnomalyDetectionValidationError,
    AnomalyDetectionLLMError,
)
from app.ai.anomaly_detection.models import (
    Anomaly,
    AnomalyAnalysisResult,
    AnomalyScan,
    AnomalySeverity,
    AnomalyType,
    ColumnThreshold,
    LLMAnomalyDetectionOutput,
)
from app.ai.anomaly_detection.prompts import build_anomaly_detection_prompt_registry
from app.ai.anomaly_detection.service import AnomalyDetectionAgent

__all__ = [
    "AnomalyDetectionAgent",
    "AnomalyAnalysisResult",
    "Anomaly",
    "AnomalyScan",
    "AnomalySeverity",
    "AnomalyType",
    "ColumnThreshold",
    "LLMAnomalyDetectionOutput",
    "AnomalyDetectionError",
    "AnomalyDetectionErrorCode",
    "AnomalyDetectionConfigurationError",
    "AnomalyDetectionAuthorizationError",
    "AnomalyDetectionValidationError",
    "AnomalyDetectionLLMError",
    "build_anomaly_detection_prompt_registry",
    "scan_for_anomalies",
    "scan_prompt_payload",
]
