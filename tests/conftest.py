import os

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg://analyticcastle:analyticcastle@127.0.0.1:5432/analyticcastle",
)
os.environ.setdefault("APP_ENV", "test")

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
