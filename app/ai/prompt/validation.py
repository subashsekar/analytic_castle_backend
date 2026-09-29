"""Prompt template validation helpers."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app.ai.prompt.errors import PromptValidationError

if TYPE_CHECKING:
    from app.ai.prompt.variables import PromptVariables

_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def extract_placeholders(template: str) -> frozenset[str]:
    return frozenset(_PLACEHOLDER.findall(template))


def validate_template(
    template: str,
    variables_type: type[PromptVariables],
    *,
    require_all_fields: bool = True,
) -> None:
    """Validate that template placeholders match the typed variable model."""
    if not template.strip():
        raise PromptValidationError("Prompt template must not be empty")

    if _has_unbalanced_braces(template):
        raise PromptValidationError("Template contains malformed placeholders")

    placeholders = extract_placeholders(template)
    field_names = frozenset(variables_type.model_fields.keys())

    unknown = placeholders - field_names
    if unknown:
        names = ", ".join(sorted(unknown))
        raise PromptValidationError(f"Template references unknown variables: {names}")

    if require_all_fields:
        unused = field_names - placeholders
        if unused:
            names = ", ".join(sorted(unused))
            raise PromptValidationError(
                f"Template is missing placeholders for variables: {names}"
            )

    stripped = _PLACEHOLDER.sub("", template)
    if "{" in stripped or "}" in stripped:
        raise PromptValidationError("Template contains malformed placeholders")


def _has_unbalanced_braces(template: str) -> bool:
    depth = 0
    for char in template:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return True
    return depth != 0
