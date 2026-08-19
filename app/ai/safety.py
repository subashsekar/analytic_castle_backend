"""Input safety checks for intent detection.

These heuristics are a defense in depth around the LLM. They classify clearly
unsupported write, credential, and instruction-override requests before any
planning occurs. They do not parse LLM output.
"""

from __future__ import annotations

import re

from app.ai.intent_types import AIConfidence, AIIntent, AIIntentType

UNSUPPORTED_WRITE = "write_operation"
UNSUPPORTED_CREDENTIAL = "credential_request"
UNSUPPORTED_INJECTION = "prompt_injection"
UNSUPPORTED_SIDE_EFFECT = "side_effect"

_POLICY_MODEL = "policy"

_WRITE_PATTERNS = (
    re.compile(
        r"\b(delete|remove|wipe|destroy)\s+(all\s+)?(the\s+)?"
        r"(customers?|orders?|users?|rows?|records?|data|everything|tables?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(drop|truncate)\b.{0,40}\b(table|database|schema|index|view)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\brun\s+drop\b", re.IGNORECASE),
    re.compile(r"\binsert\s+into\b", re.IGNORECASE),
    re.compile(r"\bupdate\s+\w+\s+set\b", re.IGNORECASE),
    re.compile(
        r"\bupdate\b.{0,80}\b(salary|password|email|status|records?|customers?|orders?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\balter\s+(table|database|schema|user|role)\b", re.IGNORECASE),
    re.compile(
        r"\b(create|grant|revoke)\s+(table|database|index|user|role|schema)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(grant|revoke)\b.{0,40}\b(on|to|from)\b", re.IGNORECASE),
    re.compile(
        r"\b(change|modify)\s+(the\s+)?(database|schema|table)\b",
        re.IGNORECASE,
    ),
)

_CREDENTIAL_PATTERNS = (
    re.compile(
        r"\b(give|tell|return|show|reveal|send|share)\b.{0,60}\b("
        r"password|passwd|api[_ -]?key|secret|credential|connection string|"
        r"database url|encryption key)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(password|api[_ -]?key|secret|credential)s?\b.{0,30}\b"
        r"(for|of)\s+(the\s+)?(database|db|connection|provider|llm)\b",
        re.IGNORECASE,
    ),
)

_INJECTION_PATTERNS = (
    re.compile(
        r"\bignore\b.{0,40}\b(previous|prior|above|all)\b.{0,40}\b"
        r"(instructions?|rules?|prompt)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(disregard|override)\b.{0,40}\b(system\s+)?(prompt|instructions?|rules?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\byou are now\b", re.IGNORECASE),
    re.compile(
        r"\breveal\b.{0,40}\b(system\s+)?(prompt|instructions?)\b",
        re.IGNORECASE,
    ),
)

_SIDE_EFFECT_PATTERNS = (
    re.compile(r"\bsend\s+(an?\s+)?email\b", re.IGNORECASE),
    re.compile(r"\bemail\s+(the\s+)?customers?\b", re.IGNORECASE),
)


def classify_unsupported(message: str) -> str | None:
    """Return a short unsupported reason, or None if the request may be analytical."""
    text = message.strip()
    if not text:
        return None
    for pattern in _WRITE_PATTERNS:
        if pattern.search(text):
            return UNSUPPORTED_WRITE
    for pattern in _CREDENTIAL_PATTERNS:
        if pattern.search(text):
            return UNSUPPORTED_CREDENTIAL
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return UNSUPPORTED_INJECTION
    for pattern in _SIDE_EFFECT_PATTERNS:
        if pattern.search(text):
            return UNSUPPORTED_SIDE_EFFECT
    return None


def unsupported_intent(reason: str) -> AIIntent:
    return AIIntent(
        intent=AIIntentType.UNSUPPORTED,
        confidence=AIConfidence.HIGH,
        requires_data_access=False,
        requires_metadata=False,
        requires_relationships=False,
        requires_clarification=False,
        unsupported_reason=reason,
    )


def policy_model_name() -> str:
    return _POLICY_MODEL
