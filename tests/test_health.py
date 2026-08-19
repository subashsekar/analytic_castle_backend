from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from app.core import health as health_module
from app.core.config import settings


def test_liveness_does_not_require_the_database(client: TestClient) -> None:
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
    assert "database" not in response.json()


def test_readiness_and_health_report_database_status(client: TestClient) -> None:
    ready = client.get("/health/ready")
    overall = client.get("/health")

    assert ready.status_code == 200
    assert ready.json() == {"status": "healthy", "database": "healthy"}
    assert overall.status_code == 200
    assert overall.json() == {"status": "healthy", "database": "healthy"}


def test_readiness_fails_cleanly_when_database_is_unavailable(
    client: TestClient,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(health_module, "check_database", lambda: False)

    ready = client.get("/health/ready")
    overall = client.get("/health")

    assert ready.status_code == 503
    assert ready.json() == {"status": "unhealthy", "database": "unavailable"}
    assert overall.status_code == 503
    assert overall.json() == {"status": "unhealthy", "database": "unavailable"}
    for response in (ready, overall):
        text = response.text.lower()
        assert "postgresql" not in text
        assert "database_url" not in text
        assert settings.DATABASE_URL not in response.text
        assert "password" not in text
        assert "traceback" not in text


def test_liveness_stays_up_when_database_check_fails(
    client: TestClient,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(health_module, "check_database", lambda: False)

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_database_health_check_hides_connection_errors(
    monkeypatch: MonkeyPatch,
) -> None:
    class FakeError(health_module.SQLAlchemyError):
        def __str__(self) -> str:
            return f"could not connect {settings.DATABASE_URL}"

    def boom() -> object:
        raise FakeError()

    monkeypatch.setattr(health_module.engine, "connect", boom)

    assert health_module.check_database() is False
