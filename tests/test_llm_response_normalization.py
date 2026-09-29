from __future__ import annotations

import json

import httpx
import pytest

from app.ai.llm import (
    AsyncLLMClient,
    LLMClientConfig,
    LLMMessage,
    LLMRequest,
    StructuredOutputConfig,
)
from app.ai.llm.content import (
    extract_json_object,
    extract_message_content,
)
from app.ai.planner_agent.validation import parse_llm_plan
from app.ai.supervisor.validation import parse_llm_classification
from tests.conftest import run_async

FAKE_API_KEY = "sk-fake-openrouter-key-not-real"


def test_extract_json_object_from_fenced_and_prose() -> None:
    raw = 'Sure.\n```json\n{"category":"ANALYTICAL_QUERY","confidence":"HIGH"}\n```\n'
    assert extract_json_object(raw)["category"] == "ANALYTICAL_QUERY"


def test_extract_message_content_from_parts_and_parsed() -> None:
    assert (
        extract_message_content(
            {"content": [{"type": "text", "text": '{"ok":true}'}]}
        )
        == '{"ok":true}'
    )
    assert json.loads(
        extract_message_content({"content": None, "parsed": {"ok": True}}) or ""
    ) == {"ok": True}


def test_extract_message_content_from_reasoning_json() -> None:
    content = extract_message_content(
        {
            "content": None,
            "reasoning": 'thinking...\n{"category":"SCHEMA_QUESTION","confidence":"MEDIUM"}',
        }
    )
    assert content is not None
    assert extract_json_object(content)["category"] == "SCHEMA_QUESTION"


def test_parsers_accept_messy_model_text() -> None:
    classification = parse_llm_classification(
        'Here you go:\n```json\n{"category":"GENERAL","confidence":"LOW",'
        '"requires_data_access":false,"requires_clarification":false,'
        '"clarification_question":null,"reason":"chat"}\n```'
    )
    assert classification.category.value == "GENERAL"

    plan = parse_llm_plan(
        'Result = {"intent":"AGGREGATION","operations":["AGGREGATE"],'
        '"required_capabilities":["AGGREGATION"],"required_data":[],'
        '"action_steps":[],"missing_information":[],'
        '"requires_clarification":false,"clarification_question":null,'
        '"unsupported":false,"unsupported_reason":null}'
    )
    assert plan.intent.value == "AGGREGATION"


def test_client_downgrades_invalid_structured_format() -> None:
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        response_format = body.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            return httpx.Response(400, json={"error": {"message": "bad schema"}})
        return httpx.Response(
            200,
            json={
                "id": "gen-1",
                "model": "openrouter/auto",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"hello":"world"}',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    client = AsyncLLMClient(
        LLMClientConfig(
            api_key=FAKE_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            model="openrouter/auto",
            timeout=5.0,
            temperature=0.2,
            max_tokens=128,
            max_retries=0,
            retry_base_backoff=0.01,
            retry_max_backoff=0.01,
        ),
        transport=httpx.MockTransport(handler),
    )
    response = run_async(
        client.complete(
            LLMRequest(
                model="openrouter/auto",
                messages=[LLMMessage(role="user", content="hi")],
                structured_output=StructuredOutputConfig(
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "demo",
                            "schema": {"type": "object"},
                            "strict": True,
                        },
                    }
                ),
            )
        )
    )
    assert response.content == '{"hello":"world"}'
    assert len(calls) == 2
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[1]["response_format"]["type"] == "json_object"


def test_client_retries_truncated_structured_output() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        max_tokens = int(body["max_tokens"])
        calls.append(max_tokens)
        if max_tokens < 2000:
            return httpx.Response(
                200,
                json={
                    "id": "gen-trunc",
                    "model": "deepseek/deepseek-v4-flash-0731",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": '{"intent":"RANKING","operation":',
                            },
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": max_tokens,
                        "total_tokens": 100 + max_tokens,
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "gen-ok",
                "model": "deepseek/deepseek-v4-flash-0731",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"intent":"RANKING","confidence":"HIGH"}',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            },
        )

    client = AsyncLLMClient(
        LLMClientConfig(
            api_key=FAKE_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            model="openrouter/auto",
            timeout=5.0,
            temperature=0.2,
            max_tokens=1024,
            max_retries=0,
            retry_base_backoff=0.01,
            retry_max_backoff=0.01,
        ),
        transport=httpx.MockTransport(handler),
    )
    response = run_async(
        client.complete(
            LLMRequest(
                model="openrouter/auto",
                messages=[LLMMessage(role="user", content="rank jobs")],
                structured_output=StructuredOutputConfig(),
            )
        )
    )
    assert '"intent":"RANKING"' in response.content
    assert calls[0] == 1024
    assert calls[1] > 1024

    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        max_tokens = int(body["max_tokens"])
        calls.append(max_tokens)
        if max_tokens < 1000:
            return httpx.Response(
                200,
                json={
                    "id": "gen-2",
                    "model": "openai/gpt-5-nano",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning": "still thinking",
                            },
                            "finish_reason": "length",
                            "native_finish_reason": "max_output_tokens",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": max_tokens,
                        "total_tokens": 10 + max_tokens,
                        "completion_tokens_details": {
                            "reasoning_tokens": max_tokens
                        },
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "gen-3",
                "model": "openai/gpt-5-nano",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"ok":true}',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    client = AsyncLLMClient(
        LLMClientConfig(
            api_key=FAKE_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-5-nano",
            timeout=5.0,
            temperature=0.2,
            max_tokens=256,
            max_retries=0,
            retry_base_backoff=0.01,
            retry_max_backoff=0.01,
        ),
        transport=httpx.MockTransport(handler),
    )
    response = run_async(
        client.complete(
            LLMRequest(
                model="openai/gpt-5-nano",
                messages=[LLMMessage(role="user", content="hi")],
            )
        )
    )
    assert response.content == '{"ok":true}'
    assert calls == [256, 1280]
