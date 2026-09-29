"""Safe prompt rendering without format-string injection."""

from __future__ import annotations

from collections.abc import Mapping

from app.ai.prompt.errors import PromptRenderError
from app.ai.prompt.models import PromptTemplate, PromptVersion, RenderedPrompt
from app.ai.prompt.validation import extract_placeholders
from app.ai.prompt.variables import PromptVariables, variables_to_render_map


def render_template(
    template: PromptTemplate,
    variables: PromptVariables,
) -> RenderedPrompt:
    """Render a versioned template using validated typed variables."""
    values = variables_to_render_map(variables)
    content = render_template_text(template.template, values)
    return RenderedPrompt(
        prompt_id=template.prompt_id,
        version=template.version,
        role=template.role,
        content=content,
    )


def render_template_text(template: str, variables: Mapping[str, str]) -> str:
    """Replace `{name}` placeholders with explicit values only."""
    placeholders = extract_placeholders(template)
    missing = sorted(name for name in placeholders if name not in variables)
    if missing:
        raise PromptRenderError(
            f"Missing values for template variables: {', '.join(missing)}"
        )

    rendered = template
    for name in sorted(placeholders, key=len, reverse=True):
        rendered = rendered.replace("{" + name + "}", variables[name])

    if not rendered.strip():
        raise PromptRenderError("Rendered prompt content must not be empty")

    return rendered


def render_system_prompt(
    *,
    prompt_id: str,
    version: PromptVersion,
    content: str,
) -> RenderedPrompt:
    if not content.strip():
        raise PromptRenderError("System prompt content must not be empty")
    return RenderedPrompt(
        prompt_id=prompt_id,
        version=version,
        role="system",
        content=content,
    )
