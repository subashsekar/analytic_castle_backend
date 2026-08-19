import sys

import pytest
from fastapi.testclient import TestClient


def test_app_starts(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["title"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only event loop policy")
def test_windows_uses_selector_event_loop_policy() -> None:
    import asyncio

    import app.main  # noqa: F401

    policy = asyncio.get_event_loop_policy()
    assert isinstance(policy, asyncio.WindowsSelectorEventLoopPolicy)


def test_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "database": "healthy"}


def test_production_app_does_not_expose_authorization_probe_routes(
    client: TestClient,
) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert all("/_authz" not in path for path in paths)
