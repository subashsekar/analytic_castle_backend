from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core import health as health_checks

router = APIRouter()


@router.get("/")
def root() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/live")
def liveness() -> dict[str, str]:
    return {"status": "healthy"}


def _readiness_response() -> JSONResponse:
    if health_checks.check_database():
        return JSONResponse(
            status_code=200,
            content={"status": "healthy", "database": "healthy"},
        )
    return JSONResponse(
        status_code=503,
        content={"status": "unhealthy", "database": "unavailable"},
    )


@router.get("/health/ready")
def readiness() -> JSONResponse:
    return _readiness_response()


@router.get("/health")
def health() -> JSONResponse:
    return _readiness_response()
