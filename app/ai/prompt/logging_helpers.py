"""Safe observability helpers that never emit prompt contents."""

from __future__ import annotations

from app.ai.prompt.models import PromptBundle, RenderedPrompt


def rendered_prompt_log_context(rendered: RenderedPrompt) -> dict[str, object]:
    return {
        "prompt_id": rendered.prompt_id,
        "prompt_version": rendered.version.value,
        "prompt_role": rendered.role,
        "prompt_char_count": rendered.char_count,
    }


def prompt_bundle_log_context(bundle: PromptBundle) -> dict[str, object]:
    context: dict[str, object] = {
        "prompt_bundle_version": bundle.version.value,
        "has_structured_output": bundle.structured_output is not None,
    }
    for label, rendered in (
        ("system", bundle.system),
        ("user", bundle.user),
        ("assistant", bundle.assistant),
    ):
        if rendered is not None:
            context[f"{label}_prompt_id"] = rendered.prompt_id
            context[f"{label}_prompt_version"] = rendered.version.value
            context[f"{label}_prompt_char_count"] = rendered.char_count
    return context
