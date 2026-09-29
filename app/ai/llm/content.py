"""Normalize LLM message content and JSON payloads across providers/models."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*([\s\S]*?)\s*```",
    re.MULTILINE,
)


def extract_message_content(message: dict[str, Any]) -> str | None:
    """Return assistant text from heterogeneous chat-completion message shapes."""
    content = message.get("content")
    text = _normalize_content_value(content)
    if text:
        return text

    parsed = message.get("parsed")
    if isinstance(parsed, dict):
        try:
            encoded = json.dumps(parsed, ensure_ascii=False)
        except (TypeError, ValueError):
            encoded = None
        if encoded and encoded.strip():
            return encoded

    for key in ("reasoning", "reasoning_content"):
        reasoning = message.get(key)
        if isinstance(reasoning, str) and reasoning.strip():
            extracted = extract_json_object_text(reasoning)
            if extracted is not None:
                return extracted

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str) and arguments.strip():
                return arguments
            if isinstance(arguments, dict):
                try:
                    return json.dumps(arguments, ensure_ascii=False)
                except (TypeError, ValueError):
                    continue
    return None


def extract_json_object(raw_content: str) -> dict[str, Any]:
    """Parse a JSON object from raw model text, tolerating fences and prose."""
    candidates = _json_candidate_strings(raw_content)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(payload, dict):
            return payload
        last_error = ValueError("JSON payload must be an object")
    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("Expecting value", raw_content, 0)


def extract_json_object_text(raw_content: str) -> str | None:
    """Return the first parseable JSON object as a string, or ``None``."""
    try:
        payload = extract_json_object(raw_content)
    except (json.JSONDecodeError, ValueError):
        return None
    try:
        return json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError):
        return None


def looks_like_reasoning_truncation(
    choice: dict[str, Any],
    message: dict[str, Any],
    usage: object,
) -> bool:
    finish_reason = choice.get("finish_reason")
    native_finish = choice.get("native_finish_reason")
    truncated = finish_reason in {"length", "max_tokens", "max_output_tokens"} or (
        isinstance(native_finish, str)
        and native_finish in {"length", "max_tokens", "max_output_tokens"}
    )
    has_reasoning = any(
        isinstance(message.get(key), str) and bool(str(message.get(key)).strip())
        for key in ("reasoning", "reasoning_content")
    )
    reasoning_tokens = 0
    if isinstance(usage, dict):
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict):
            raw = details.get("reasoning_tokens")
            if isinstance(raw, int):
                reasoning_tokens = raw
    return truncated and (has_reasoning or reasoning_tokens > 0)


def _normalize_content_value(content: object) -> str | None:
    if isinstance(content, str):
        stripped = content.strip()
        return content if stripped else None
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)
                continue
            if item.get("type") == "output_text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        joined = "\n".join(parts).strip()
        return joined or None
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str) and text.strip():
            return text
    return None


def _json_candidate_strings(raw_content: str) -> list[str]:
    text = raw_content.strip()
    if not text:
        return []
    candidates: list[str] = [text]
    for match in _FENCE_RE.finditer(text):
        fenced = match.group(1).strip()
        if fenced:
            candidates.append(fenced)
    extracted = _extract_balanced_object(text)
    if extracted is not None:
        candidates.append(extracted)
    # Preserve order, drop duplicates.
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def _extract_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
