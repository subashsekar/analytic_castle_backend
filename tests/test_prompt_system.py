from __future__ import annotations

import logging
from typing import ClassVar

import pytest
from pydantic import BaseModel

from app.ai.llm.models import LLMMessage
from app.ai.prompt import (
    PromptBundle,
    PromptNotFoundError,
    PromptRegistry,
    PromptRenderError,
    PromptTemplate,
    PromptValidationError,
    PromptVariables,
    PromptVersion,
    PromptVersionError,
    SystemPrompt,
    bundle_to_llm_messages,
    extract_placeholders,
    json_object_output,
    prompt_bundle_log_context,
    render_template_text,
    rendered_prompt_log_context,
    response_schema,
    structured_output_from_model,
    validate_template,
    variables_to_render_map,
)
from app.core.logging import RedactingFilter, redact_secret

SECRET_VALUE = "sk-test-secret-prompt-key"
CUSTOMER_PASSWORD = "CustomerDbPassword!@# 42"


class DemoUserVariables(PromptVariables):
    _sensitive_fields: ClassVar[frozenset[str]] = frozenset({"secret_note"})

    workspace_name: str
    message: str
    secret_note: str = ""


class DemoResponseModel(BaseModel):
    answer: str
    confidence: str


def _registry() -> PromptRegistry:
    registry = PromptRegistry()
    registry.register_system(
        SystemPrompt(
            prompt_id="demo.system",
            version=PromptVersion(value="v1"),
            content="You are a helpful analyst.",
        )
    )
    registry.register_system(
        SystemPrompt(
            prompt_id="demo.system",
            version=PromptVersion(value="v2"),
            content="You are a careful analyst.",
        )
    )
    registry.register_template(
        PromptTemplate(
            prompt_id="demo.user",
            version=PromptVersion(value="v1"),
            template=(
                "Workspace: {workspace_name}\n"
                "Secret note: {secret_note}\n"
                "Request: {message}"
            ),
        ),
        DemoUserVariables,
    )
    return registry


def test_extract_placeholders_finds_named_fields() -> None:
    placeholders = extract_placeholders("Hello {name}, use {tool}.")
    assert placeholders == frozenset({"name", "tool"})


def test_validate_template_rejects_unknown_placeholders() -> None:
    with pytest.raises(PromptValidationError, match="unknown variables"):
        validate_template("Hello {missing}", DemoUserVariables)


def test_validate_template_rejects_missing_placeholders() -> None:
    with pytest.raises(PromptValidationError, match="missing placeholders"):
        validate_template("Only {workspace_name}", DemoUserVariables)


def test_validate_template_rejects_malformed_placeholders() -> None:
    with pytest.raises(PromptValidationError, match="malformed"):
        validate_template("Broken {workspace_name", DemoUserVariables)


def test_render_template_text_substitutes_values() -> None:
    rendered = render_template_text(
        "Workspace: {workspace_name}\nRequest: {message}",
        {
            "workspace_name": "Acme",
            "message": "Show revenue",
            "secret_note": SECRET_VALUE,
        },
    )
    assert "Workspace: Acme" in rendered
    assert "Show revenue" in rendered


def test_render_template_text_rejects_missing_values() -> None:
    with pytest.raises(PromptRenderError, match="Missing values"):
        render_template_text("{workspace_name}", {"message": "x"})


def test_render_template_text_is_not_vulnerable_to_format_injection() -> None:
    rendered = render_template_text(
        "Value: {message}",
        {
            "message": "{secret_note}",
            "workspace_name": "Acme",
            "secret_note": "ignored",
        },
    )
    assert rendered == "Value: {secret_note}"


def test_registry_resolves_latest_version() -> None:
    registry = _registry()
    system = registry.get_system("demo.system")
    assert system.version.value == "v2"


def test_registry_can_render_specific_version() -> None:
    registry = _registry()
    rendered = registry.render_system(
        "demo.system",
        version=PromptVersion(value="v1"),
    )
    assert "helpful analyst" in rendered.content


def test_registry_rejects_wrong_variable_type() -> None:
    registry = _registry()

    class OtherVariables(PromptVariables):
        workspace_name: str
        message: str
        secret_note: str = ""

    with pytest.raises(PromptVersionError, match="DemoUserVariables"):
        registry.render_template(
            "demo.user",
            OtherVariables(
                workspace_name="Acme",
                message="hello",
                secret_note=SECRET_VALUE,
            ),
        )


def test_registry_lists_registered_versions() -> None:
    registry = _registry()
    assert registry.list_versions("demo.system") == ("v1", "v2")


def test_registry_builds_bundle_for_llm_messages() -> None:
    registry = _registry()
    bundle = registry.build_bundle(
        bundle_version=PromptVersion(value="bundle_v1"),
        system_prompt_id="demo.system",
        user_template_id="demo.user",
        user_variables=DemoUserVariables(
            workspace_name="Acme",
            message="Count orders",
            secret_note=SECRET_VALUE,
        ),
        structured_output=structured_output_from_model(DemoResponseModel),
    )
    messages = bundle_to_llm_messages(bundle)
    assert messages == [
        LLMMessage(role="system", content=bundle.system.content),  # type: ignore[union-attr]
        LLMMessage(role="user", content=bundle.user.content),  # type: ignore[union-attr]
    ]
    assert bundle.structured_output is not None
    assert bundle.structured_output.response_format["type"] == "json_schema"


def test_structured_output_from_model_builds_json_schema() -> None:
    config = structured_output_from_model(DemoResponseModel)
    schema = config.response_format["json_schema"]["schema"]
    assert "answer" in schema["properties"]
    assert response_schema(DemoResponseModel)["title"] == "DemoResponseModel"


def test_structured_output_is_openai_strict_compatible() -> None:
    class OptionalFieldsModel(BaseModel):
        label: str
        confidence: str = "MEDIUM"
        note: str | None = None

    config = structured_output_from_model(OptionalFieldsModel)
    schema = config.response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert "default" not in schema["properties"]["confidence"]
    assert config.response_format["json_schema"]["strict"] is True


def test_json_object_output_uses_default_response_format() -> None:
    config = json_object_output()
    assert config.response_format == {"type": "json_object"}


def test_rendered_prompt_repr_hides_content() -> None:
    registry = _registry()
    rendered = registry.render_template(
        "demo.user",
        DemoUserVariables(
            workspace_name="Acme",
            message="hello",
            secret_note=SECRET_VALUE,
        ),
    )
    text = repr(rendered)
    assert SECRET_VALUE not in text
    assert CUSTOMER_PASSWORD not in text
    assert "char_count=" in text


def test_prompt_log_context_never_includes_content() -> None:
    registry = _registry()
    bundle = registry.build_bundle(
        bundle_version=PromptVersion(value="bundle_v1"),
        system_prompt_id="demo.system",
        user_template_id="demo.user",
        user_variables=DemoUserVariables(
            workspace_name="Acme",
            message=CUSTOMER_PASSWORD,
            secret_note=SECRET_VALUE,
        ),
    )
    system_context = rendered_prompt_log_context(bundle.system)  # type: ignore[arg-type]
    bundle_context = prompt_bundle_log_context(bundle)
    assert CUSTOMER_PASSWORD not in str(system_context)
    assert SECRET_VALUE not in str(bundle_context)
    assert bundle_context["user_prompt_char_count"] > 0


def test_variables_to_render_map_stringifies_values() -> None:
    rendered = variables_to_render_map(
        DemoUserVariables(
            workspace_name="Acme",
            message="hello",
            secret_note=SECRET_VALUE,
        )
    )
    assert rendered["workspace_name"] == "Acme"
    assert rendered["secret_note"] == SECRET_VALUE


def test_prompt_errors_redact_secrets() -> None:
    error = PromptRenderError(f"failed with api_key={SECRET_VALUE}")
    assert SECRET_VALUE not in str(error)
    assert "api_key=[REDACTED]" in str(error)


def test_prompt_logging_does_not_leak_sensitive_variables(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("tests.prompt.security")
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter())
    logger.addHandler(handler)
    logger.propagate = False

    registry = _registry()
    rendered = registry.render_template(
        "demo.user",
        DemoUserVariables(
            workspace_name="Acme",
            message="hello",
            secret_note=SECRET_VALUE,
        ),
    )
    logger.info("prompt rendered %s", rendered_prompt_log_context(rendered))

    output = caplog.text + redact_secret(str(rendered_prompt_log_context(rendered)))
    assert SECRET_VALUE not in output
    assert CUSTOMER_PASSWORD not in output
    logger.removeHandler(handler)


def test_missing_prompt_raises_not_found() -> None:
    registry = PromptRegistry()
    with pytest.raises(PromptNotFoundError):
        registry.render_system("missing.prompt")


def test_prompt_bundle_requires_at_least_one_message() -> None:
    with pytest.raises(ValueError, match="at least one message"):
        PromptBundle(version=PromptVersion(value="v1"))
