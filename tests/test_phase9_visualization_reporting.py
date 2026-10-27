from __future__ import annotations

import uuid

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import create_access_token
from app.db.models import DataSource, Organization, Workspace, WorkspaceMember
from app.enums import DataSourceType, UserRole, WorkspaceRole

from tests.conftest import _create_user


def _auth_headers(user_id: uuid.UUID) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user_id)}"}


def _seed_data_source(db: Session, *, workspace: Workspace, created_by: uuid.UUID) -> DataSource:
    ds = DataSource(
        workspace_id=workspace.id,
        name=f"DS {uuid.uuid4().hex[:6]}",
        type=DataSourceType.POSTGRESQL,
        created_by=created_by,
    )
    db.add(ds)
    db.flush()
    return ds


def _seed_second_user_in_workspace(
    db: Session,
    *,
    workspace: Workspace,
    organization: Organization,
    role: UserRole = UserRole.USER,
) -> WorkspaceMember:
    user = _create_user(db, role=role, first_name="Other", last_name="User")
    member = WorkspaceMember(
        workspace_id=workspace.id,
        user_id=user.id,
        role=WorkspaceRole.OWNER,
    )
    db.add(member)
    db.flush()
    return member


def test_charts_ready_data_workspace_isolation_and_validation(
    client: TestClient,
    db_session: Session,
    organization: Organization,
    workspace: Workspace,
    workspace_member: WorkspaceMember,
    auth_headers: dict[str, str],
) -> None:
    data_source = _seed_data_source(db_session, workspace=workspace, created_by=workspace_member.user_id)

    # Happy-path: valid chart spec should be accepted and echoed back.
    valid_payload = {
        "chart_spec": {
            "type": "line",
            "title": None,
            "x_axis": "period",
            "y_axis": "amount",
            "x": ["2024-01-01"],
            "y": [1.5],
            "truncated": False,
        }
    }
    resp = client.post(
        f"/api/v1/workspaces/{workspace.id}/charts/data-sources/{data_source.id}/ready-data",
        json=valid_payload,
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["workspace_id"] == str(workspace.id)
    assert body["data_source_id"] == str(data_source.id)
    assert body["chart_ready_data"]["type"] == "line"

    # Invalid chart spec should fail pydantic validation (422).
    invalid_payload = {
        "chart_spec": {
            "type": "line",
            "x": ["2024-01-01"],
            "y": ["not-a-number"],
        }
    }
    resp = client.post(
        f"/api/v1/workspaces/{workspace.id}/charts/data-sources/{data_source.id}/ready-data",
        json=invalid_payload,
        headers=auth_headers,
    )
    assert resp.status_code == 422

    # Workspace mismatch should not leak existence: endpoint returns 404.
    other_workspace = Workspace(
        organization_id=organization.id,
        name="Other WS",
        slug=f"other-ws-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(other_workspace)
    db_session.flush()
    resp = client.post(
        f"/api/v1/workspaces/{other_workspace.id}/charts/data-sources/{data_source.id}/ready-data",
        json=valid_payload,
        headers=auth_headers,
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Workspace not found"


def test_report_creation_empty_content_rejected(
    client: TestClient,
    db_session: Session,
    workspace: Workspace,
    workspace_member: WorkspaceMember,
    auth_headers: dict[str, str],
) -> None:
    data_source = _seed_data_source(
        db_session, workspace=workspace, created_by=workspace_member.user_id
    )

    empty_payload = {
        "question": "What is happening?",
        "summary": "No details",
        "key_findings": [],
        "chart_specifications": [],
        "supporting_data": {},
    }
    resp = client.post(
        f"/api/v1/workspaces/{workspace.id}/data-sources/{data_source.id}/reports",
        json=empty_payload,
        headers=auth_headers,
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Report content is empty"


def test_report_exports_csv_formula_injection_and_ownership_isolation(
    client: TestClient,
    db_session: Session,
    organization: Organization,
    workspace: Workspace,
    workspace_member: WorkspaceMember,
    auth_headers: dict[str, str],
) -> None:
    data_source = _seed_data_source(
        db_session, workspace=workspace, created_by=workspace_member.user_id
    )

    # Create a report for user #1 with formula-like content in key_findings.
    create_payload = {
        "question": "Q1",
        "summary": "S1",
        "key_findings": ["\t=SUM(1,2)"],
        "chart_specifications": [
            {
                "type": "line",
                "title": None,
                "x_axis": "period",
                "y_axis": "value",
                "x": ["2024-01-01", "2024-02-01"],
                "y": [1.0, 2.0],
                "truncated": False,
            }
        ],
        "supporting_data": {"some": "data"},
    }
    resp = client.post(
        f"/api/v1/workspaces/{workspace.id}/data-sources/{data_source.id}/reports",
        json=create_payload,
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.text
    report_id = uuid.UUID(resp.json()["id"])

    # CSV export should neutralize CSV injection.
    export_resp = client.get(
        f"/api/v1/workspaces/{workspace.id}/reports/{report_id}/export/csv",
        headers=auth_headers,
    )
    assert export_resp.status_code == 200
    csv_text = export_resp.text
    assert "'\t=SUM(1,2)" in csv_text
    assert export_resp.headers["content-type"].startswith("text/csv")
    assert "attachment" in export_resp.headers["content-disposition"]

    # Ownership isolation: user #2 should not be able to export user #1’s report.
    other_member = _seed_second_user_in_workspace(
        db_session,
        workspace=workspace,
        organization=organization,
    )
    other_headers = _auth_headers(other_member.user_id)
    export_resp = client.get(
        f"/api/v1/workspaces/{workspace.id}/reports/{report_id}/export/csv",
        headers=other_headers,
    )
    assert export_resp.status_code == 404

    # Chart CSV export should flatten the stored ChartSpec.
    charts_csv_resp = client.get(
        f"/api/v1/workspaces/{workspace.id}/reports/{report_id}/export/charts/csv",
        headers=auth_headers,
    )
    assert charts_csv_resp.status_code == 200
    charts_csv = charts_csv_resp.text
    assert "chart_type" in charts_csv
    assert "line" in charts_csv
    assert ",2024-01-01," in charts_csv

    # PDF export should return a PDF payload.
    pdf_resp = client.get(
        f"/api/v1/workspaces/{workspace.id}/reports/{report_id}/export/pdf",
        headers=auth_headers,
    )
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"].startswith("application/pdf")
    assert pdf_resp.content.startswith(b"%PDF")


def test_report_creation_whitespace_only_question_and_summary_rejected(
    client: TestClient,
    db_session: Session,
    workspace: Workspace,
    workspace_member: WorkspaceMember,
    auth_headers: dict[str, str],
) -> None:
    data_source = _seed_data_source(
        db_session, workspace=workspace, created_by=workspace_member.user_id
    )

    payload = {
        "question": "   ",
        "summary": "\n\t  ",
        "key_findings": ["ok"],
        "chart_specifications": [],
        "supporting_data": {"some": "data"},
    }
    resp = client.post(
        f"/api/v1/workspaces/{workspace.id}/data-sources/{data_source.id}/reports",
        json=payload,
        headers=auth_headers,
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Report content is empty"

