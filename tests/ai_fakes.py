from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic import BaseModel, ValidationError

from app.ai.exceptions import AIResponseValidationError
from app.ai.intent_types import (
    AIConfidence,
    AIIntentType,
    LLMIntentDetection,
)
from app.ai.providers.base import LLMGeneration, LLMMessage, StructuredGeneration
from app.ai.types import AIAnalysisResult, TokenUsage

TSchema = type[BaseModel]


def default_intent_detection() -> LLMIntentDetection:
    return LLMIntentDetection(
        intent=AIIntentType.SCHEMA_QUESTION,
        subject="catalog",
        confidence=AIConfidence.HIGH,
        requires_data_access=False,
        requires_metadata=True,
        requires_clarification=False,
    )


class FakeLLMProvider:
    """Test double. Not a production LLM provider."""

    name = "fake"

    def __init__(self) -> None:
        self.messages: list[Sequence[LLMMessage]] = []
        self.error: Exception | None = None
        self.invalid_content: str | None = None
        self.structured_payload: BaseModel | dict[str, object] | None = None
        self.intent = default_intent_detection()
        self.analysis = AIAnalysisResult(
            answer="I can only discuss the authorized data source from the provided context.",
            intent="question",
            requires_data_access=False,
            metadata_context=[],
            warnings=[],
        )
        self.model = "fake-model"
        self.usage = TokenUsage(
            input_tokens=11,
            output_tokens=7,
            total_tokens=18,
        )
        self.generate_calls = 0
        self.structured_calls = 0

    async def generate(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMGeneration:
        del temperature, max_output_tokens
        self.generate_calls += 1
        self.messages.append(messages)
        if self.error is not None:
            raise self.error
        content = self.invalid_content
        if content is None:
            payload = self.structured_payload or self.intent
            if isinstance(payload, BaseModel):
                content = payload.model_dump_json()
            else:
                content = json.dumps(payload)
        return LLMGeneration(content=content, model=self.model, usage=self.usage)

    async def generate_structured(
        self,
        messages: Sequence[LLMMessage],
        schema: TSchema,
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> StructuredGeneration[BaseModel]:
        self.structured_calls += 1
        generation = await self.generate(
            messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        if self.invalid_content is not None:
            raise AIResponseValidationError(raw_content=self.invalid_content)
        payload: object
        if self.structured_payload is not None:
            payload = self.structured_payload
        elif schema is AIAnalysisResult:
            payload = self.analysis
        else:
            payload = self.intent
        try:
            if isinstance(payload, schema):
                result: BaseModel = payload
            elif isinstance(payload, BaseModel):
                result = schema.model_validate(payload.model_dump())
            elif isinstance(payload, dict):
                result = schema.model_validate(payload)
            else:
                raise AIResponseValidationError(raw_content=generation.content)
        except ValidationError as exc:
            raise AIResponseValidationError(raw_content=generation.content) from exc
        return StructuredGeneration(
            result=result,
            model=generation.model,
            usage=generation.usage,
            finish_reason=generation.finish_reason,
        )
