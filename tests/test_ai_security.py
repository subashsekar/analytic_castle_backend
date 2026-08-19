from __future__ import annotations

import ast
from pathlib import Path

from app.ai.prompts import (
    INTENT_SYSTEM_PROMPT_V1,
    SYSTEM_PROMPT_V1,
    USER_PROMPT_TEMPLATE_V1,
)
from app.ai.providers.config import LLMProviderConfig
from app.core.logging import redact_secret

_AI_DIR = Path(__file__).resolve().parents[1] / "app" / "ai"
_ROUTE_PATH = Path(__file__).resolve().parents[1] / "app" / "api" / "routes" / "ai.py"
_FORBIDDEN_IMPORTS = {
    "psycopg",
    "asyncpg",
    "app.connectors.postgresql",
    "app.services.sample_data",
    "app.services.credentials",
    "mcp",
    "langchain",
    "chromadb",
}
SECRET_KEY = "sk-test-secret-llm-key-do-not-log"
CUSTOMER_PASSWORD = "CustomerDbPassword!@# 42"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_ai_modules_do_not_import_query_or_credential_stacks() -> None:
    files = list(_AI_DIR.rglob("*.py")) + [_ROUTE_PATH]
    for path in files:
        imported = _imported_modules(path)
        assert not imported.intersection(_FORBIDDEN_IMPORTS), path
        source = path.read_text(encoding="utf-8")
        assert "execute_query" not in source or path.name == "exceptions.py"
        assert "generate_sql" not in source


def test_prompts_forbid_sql_execution_and_secret_exposure() -> None:
    combined = SYSTEM_PROMPT_V1 + INTENT_SYSTEM_PROMPT_V1 + USER_PROMPT_TEMPLATE_V1
    assert "Do not generate or execute SQL" in combined
    assert "Never invent" in combined
    assert "untrusted" in combined.lower()
    assert "UNSUPPORTED" in INTENT_SYSTEM_PROMPT_V1
    assert "secrets" in combined.lower()
    assert CUSTOMER_PASSWORD not in combined
    assert SECRET_KEY not in combined


def test_provider_config_never_reprs_api_key() -> None:
    config = LLMProviderConfig(
        provider="openai",
        api_key=SECRET_KEY,
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        timeout_seconds=30.0,
        temperature=0.2,
        max_output_tokens=256,
    )
    rendered = repr(config)
    assert SECRET_KEY not in rendered
    assert config.api_key == SECRET_KEY


def test_redactor_covers_llm_api_key() -> None:
    assert (
        redact_secret(f"LLM_API_KEY={SECRET_KEY} api_key={SECRET_KEY}")
        == "LLM_API_KEY=[REDACTED] api_key=[REDACTED]"
    )
