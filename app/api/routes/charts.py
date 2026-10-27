"""Visualization (Phase 9.3) endpoints.

These endpoints translate already-executed SQL results and/or chart
specifications into chart-ready payloads without LLM calls or re-querying.
"""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.ai.result_facts import build_result_facts
from app.ai.sql_execution.models import SQLExecutionResult
from app.ai.visualization_agent import build_chart_spec
from app.api.deps import require_data_source_permission, require_rate_limit
from app.core.authorization import DataSourceAccess
from app.core.request_id import get_request_id, new_request_id
from app.enums import WorkspacePermission
from app.schemas.charts import (
    ChartReadyDataRequest,
    ChartReadyDataResponse,
    ChartSpecificationRequest,
    ChartSpecificationResponse,
    ChartSpec,
)

router = APIRouter(prefix="/api/v1/workspaces/{workspace_id}/charts", tags=["charts"])

_AUTH_ERRORS: dict[int | str, dict[str, Any]] = {
    status.HTTP_401_UNAUTHORIZED: {"description": "Not authenticated"},
    status.HTTP_403_FORBIDDEN: {"description": "Not authorized"},
}

_MAX_API_ROWS = 10_000
_MAX_CHART_INPUT_ROWS = 2_000  # keep in sync with visualization_agent's spec builder

INVALID_REQUEST_DETAIL = "Invalid request"
DATA_SOURCE_NOT_FOUND_DETAIL = "Data source not found"
WORKSPACE_NOT_FOUND_DETAIL = "Workspace not found"


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

def _workspace_not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=WORKSPACE_NOT_FOUND_DETAIL,
    )


def _request_id_for_current_request() -> str:
    request_id = get_request_id()
    if not request_id or request_id == "-":
        request_id = new_request_id()
    return request_id


@router.post(
    "/data-sources/{data_source_id}/specifications",
    response_model=ChartSpecificationResponse,
    response_model_exclude_none=True,
    summary="Build a chart specification from query results",
    description=(
        "Phase 9.1/9.2 integration: builds a deterministic chart spec and "
        "validates it against the server chart schema. No SQL re-execution and "
        "no LLM calls are performed."
    ),
    responses={
        **_AUTH_ERRORS,
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid chart specification request"},
        status.HTTP_404_NOT_FOUND: {"description": DATA_SOURCE_NOT_FOUND_DETAIL},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Validation error"},
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "Too many requests"},
    },
    dependencies=[Depends(require_rate_limit("ai-chat"))],
)
def build_chart_specifications(
    workspace_id: UUID,
    data_source_id: UUID,
    payload: ChartSpecificationRequest,
    _access: Annotated[
        DataSourceAccess,
        Depends(
            require_data_source_permission(
                WorkspacePermission.DATA_SOURCE_READ,
                allow_super_admin=True,
            )
        ),
    ],
) -> ChartSpecificationResponse:
    if _access.workspace.id != workspace_id:
        # Avoid leaking the existence of other workspaces.
        raise _workspace_not_found()

    rows = payload.query_result.rows
    if rows is None:
        raise _bad_request("query_result.rows must be present")
    if len(rows) > _MAX_API_ROWS:
        raise _bad_request(
            f"query_result.rows exceeds maximum payload size ({_MAX_API_ROWS})"
        )

    query_result: SQLExecutionResult = payload.query_result
    if len(query_result.rows) > _MAX_CHART_INPUT_ROWS:
        query_result = query_result.model_copy(
            update={
                "rows": query_result.rows[:_MAX_CHART_INPUT_ROWS],
                "truncated": True,
            }
        )

    try:
        facts = build_result_facts(query_result, message=payload.message)
        spec = build_chart_spec(
            message=payload.message, query_result=query_result, facts=facts
        )
    except Exception as exc:  # noqa: BLE001
        raise _bad_request(str(exc) or INVALID_REQUEST_DETAIL) from None

    if spec is None:
        raise _bad_request(INVALID_REQUEST_DETAIL)

    # Ensure we always return a ChartSpec-shaped payload (avoid response-model
    # validation surprises turning into 5xx responses).
    try:
        ChartSpec.model_validate(spec)
    except Exception:  # noqa: BLE001
        raise _bad_request(INVALID_REQUEST_DETAIL) from None

    return ChartSpecificationResponse(
        request_id=_request_id_for_current_request(),
        workspace_id=workspace_id,
        data_source_id=data_source_id,
        chart_spec=spec,  # response model validates this structure.
    )


@router.post(
    "/data-sources/{data_source_id}/ready-data",
    response_model=ChartReadyDataResponse,
    response_model_exclude_none=True,
    summary="Convert a chart specification into chart-ready data",
    description=(
        "Phase 9.3 'retrieve' endpoint: validates the incoming chart spec and "
        "returns chart-ready data without re-querying or calling any LLM."
    ),
    responses={
        **_AUTH_ERRORS,
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid chart specification"},
        status.HTTP_404_NOT_FOUND: {"description": DATA_SOURCE_NOT_FOUND_DETAIL},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Validation error"},
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "Too many requests"},
    },
    dependencies=[Depends(require_rate_limit("ai-chat"))],
)
def build_chart_ready_data(
    workspace_id: UUID,
    data_source_id: UUID,
    payload: ChartReadyDataRequest,
    _access: Annotated[
        DataSourceAccess,
        Depends(
            require_data_source_permission(
                WorkspacePermission.DATA_SOURCE_READ,
                allow_super_admin=True,
            )
        ),
    ],
) -> ChartReadyDataResponse:
    if _access.workspace.id != workspace_id:
        raise _workspace_not_found()
    return ChartReadyDataResponse(
        request_id=_request_id_for_current_request(),
        workspace_id=workspace_id,
        data_source_id=data_source_id,
        chart_ready_data=payload.chart_spec,
    )

