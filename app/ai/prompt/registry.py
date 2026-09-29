"""Versioned prompt registry for system prompts and templates."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.ai.llm.models import StructuredOutputConfig
from app.ai.prompt.errors import PromptNotFoundError, PromptVersionError
from app.ai.prompt.models import (
    PromptBundle,
    PromptTemplate,
    PromptVersion,
    RenderedPrompt,
    SystemPrompt,
)
from app.ai.prompt.render import render_system_prompt, render_template
from app.ai.prompt.validation import validate_template
from app.ai.prompt.variables import PromptVariables


@dataclass
class PromptRegistry:
    """In-memory registry of versioned prompts for future agents."""

    _system_prompts: dict[tuple[str, str], SystemPrompt] = field(default_factory=dict)
    _templates: dict[tuple[str, str], tuple[PromptTemplate, type[PromptVariables]]] = (
        field(default_factory=dict)
    )
    _latest_versions: dict[str, str] = field(default_factory=dict)

    def register_system(self, prompt: SystemPrompt) -> None:
        key = (prompt.prompt_id, prompt.version.value)
        self._system_prompts[key] = prompt
        self._track_latest(prompt.prompt_id, prompt.version)

    def register_template(
        self,
        template: PromptTemplate,
        variables_type: type[PromptVariables],
        *,
        validate: bool = True,
    ) -> None:
        if validate:
            validate_template(template.template, variables_type)
        key = (template.prompt_id, template.version.value)
        self._templates[key] = (template, variables_type)
        self._track_latest(template.prompt_id, template.version)

    def get_system(
        self,
        prompt_id: str,
        *,
        version: PromptVersion | None = None,
    ) -> SystemPrompt:
        resolved = self._resolve_version(prompt_id, version)
        key = (prompt_id, resolved.value)
        prompt = self._system_prompts.get(key)
        if prompt is None:
            raise PromptNotFoundError(
                f"System prompt {prompt_id!r} version {resolved.value!r} was not found"
            )
        return prompt

    def get_template(
        self,
        prompt_id: str,
        *,
        version: PromptVersion | None = None,
    ) -> PromptTemplate:
        resolved = self._resolve_version(prompt_id, version)
        entry = self._templates.get((prompt_id, resolved.value))
        if entry is None:
            raise PromptNotFoundError(
                f"Prompt template {prompt_id!r} version {resolved.value!r} was not found"
            )
        return entry[0]

    def render_system(
        self,
        prompt_id: str,
        *,
        version: PromptVersion | None = None,
    ) -> RenderedPrompt:
        prompt = self.get_system(prompt_id, version=version)
        return render_system_prompt(
            prompt_id=prompt.prompt_id,
            version=prompt.version,
            content=prompt.content,
        )

    def render_template(
        self,
        prompt_id: str,
        variables: PromptVariables,
        *,
        version: PromptVersion | None = None,
    ) -> RenderedPrompt:
        resolved = self._resolve_version(prompt_id, version)
        entry = self._templates.get((prompt_id, resolved.value))
        if entry is None:
            raise PromptNotFoundError(
                f"Prompt template {prompt_id!r} version {resolved.value!r} was not found"
            )
        template, variables_type = entry
        if not isinstance(variables, variables_type):
            raise PromptVersionError(
                f"Variables for {prompt_id!r} must be {variables_type.__name__}"
            )
        return render_template(template, variables)

    def build_bundle(
        self,
        *,
        bundle_version: PromptVersion,
        system_prompt_id: str | None = None,
        user_template_id: str | None = None,
        user_variables: PromptVariables | None = None,
        system_version: PromptVersion | None = None,
        user_version: PromptVersion | None = None,
        structured_output: StructuredOutputConfig | None = None,
    ) -> PromptBundle:
        system = (
            self.render_system(system_prompt_id, version=system_version)
            if system_prompt_id
            else None
        )
        user = (
            self.render_template(
                user_template_id,
                user_variables,
                version=user_version,
            )
            if user_template_id and user_variables is not None
            else None
        )
        return PromptBundle(
            version=bundle_version,
            system=system,
            user=user,
            structured_output=structured_output,
        )

    def list_versions(self, prompt_id: str) -> tuple[str, ...]:
        versions = {
            version
            for registered_id, version in (
                *self._system_prompts.keys(),
                *self._templates.keys(),
            )
            if registered_id == prompt_id
        }
        return tuple(sorted(versions))

    def _resolve_version(
        self,
        prompt_id: str,
        version: PromptVersion | None,
    ) -> PromptVersion:
        if version is not None:
            return version
        latest = self._latest_versions.get(prompt_id)
        if latest is None:
            raise PromptNotFoundError(f"Prompt {prompt_id!r} was not found")
        return PromptVersion(value=latest)

    def _track_latest(self, prompt_id: str, version: PromptVersion) -> None:
        current = self._latest_versions.get(prompt_id)
        if current is None or version.value > current:
            self._latest_versions[prompt_id] = version.value
