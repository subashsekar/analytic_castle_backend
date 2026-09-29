"""Typed prompt variable models with sensitive-field metadata."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict


class PromptVariables(BaseModel):
    """Base class for typed template variables."""

    model_config = ConfigDict(extra="forbid")

    # Subclasses should override with ClassVar frozensets of field names.
    _sensitive_fields: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def sensitive_fields(cls) -> frozenset[str]:
        """Field names that must never appear in logs or repr output."""
        declared = cls.__dict__.get("_sensitive_fields", frozenset())
        if not isinstance(declared, frozenset):
            declared = frozenset(declared) if declared else frozenset()
        return declared

    def __repr__(self) -> str:
        sensitive = self.sensitive_fields()
        parts: list[str] = []
        for name in self.__class__.model_fields:
            if name in sensitive:
                parts.append(f"{name}='[REDACTED]'")
            else:
                parts.append(f"{name}={getattr(self, name)!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def log_context(self) -> dict[str, object]:
        """Safe log fields: sensitive values are redacted to counts only."""
        sensitive = self.sensitive_fields()
        context: dict[str, object] = {}
        for name, value in self.model_dump().items():
            if name in sensitive:
                text = "" if value is None else str(value)
                context[f"{name}_char_count"] = len(text)
                context[f"{name}_redacted"] = True
            else:
                context[name] = value
        return context


def variables_to_render_map(model: PromptVariables) -> dict[str, str]:
    """Convert validated variables into string values for template rendering."""
    rendered: dict[str, str] = {}
    for name, value in model.model_dump().items():
        rendered[name] = _stringify_value(value)
    return rendered


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "\n".join(_stringify_value(item) for item in value)
    return str(value)
