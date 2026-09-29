from __future__ import annotations

import ast
from pathlib import Path

from app.ai.prompts import (
    INTENT_SYSTEM_PROMPT_V1,
    SYSTEM_PROMPT_V1,
    USER_PROMPT_TEMPLATE_V1,
)
from app.ai.providers.config import LLMProviderConfig
from app.ai.sql_generation.prompts import (
    SQL_GENERATION_SYSTEM_PROMPT_ID,
    build_sql_generation_prompt_registry,
)
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
_SQL_GENERATION_DIR = _AI_DIR / "sql_generation"
_SQL_VALIDATION_DIR = _AI_DIR / "sql_validation"
_SQL_VALIDATION_ALLOWED_CONNECTORS = {
    "app.connectors.readonly_sql",
    "app.connectors.exceptions",
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


def _allows_generate_sql(path: Path) -> bool:
    try:
        path.relative_to(_SQL_GENERATION_DIR)
    except ValueError:
        return False
    return True


def _is_sql_validation(path: Path) -> bool:
    try:
        path.relative_to(_SQL_VALIDATION_DIR)
    except ValueError:
        return False
    return True


def test_ai_modules_do_not_import_query_or_credential_stacks() -> None:
    files = list(_AI_DIR.rglob("*.py")) + [_ROUTE_PATH]
    for path in files:
        imported = _imported_modules(path)
        assert not imported.intersection(_FORBIDDEN_IMPORTS), path
        source = path.read_text(encoding="utf-8")
        assert "execute_query" not in source or path.name == "exceptions.py"
        if "generate_sql" in source:
            assert _allows_generate_sql(path), path
        if _allows_generate_sql(path):
            assert "execute_query" not in source
            assert "app.mcp" not in source
            assert "app.connectors" not in source
        if _is_sql_validation(path):
            assert "execute_query" not in source
            assert "app.mcp" not in source
            connector_imports = {
                name for name in imported if name.startswith("app.connectors")
            }
            assert connector_imports <= _SQL_VALIDATION_ALLOWED_CONNECTORS, path


def test_sql_generation_prompt_forbids_execution_and_invention() -> None:
    registry = build_sql_generation_prompt_registry()
    system = registry.get_system(SQL_GENERATION_SYSTEM_PROMPT_ID)
    content = system.content.lower()
    assert "do not execute sql" in content
    assert "do not invent" in content
    assert "untrusted" in content
    assert "credentials" in content
    assert "bypass schema" in content or "ignore any user instructions" in content
    assert CUSTOMER_PASSWORD not in system.content
    assert SECRET_KEY not in system.content


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
