from fastapi.testclient import TestClient

from app.db.models import User, WorkspaceMember
from app.enums import UserRole, WorkspaceRole


def test_role_fixtures_are_explicit(
    test_user: User,
    admin_user: User,
    super_admin_user: User,
) -> None:
    assert test_user.role == UserRole.USER
    assert admin_user.role == UserRole.ADMIN
    assert super_admin_user.role == UserRole.SUPER_ADMIN
    assert test_user.id != admin_user.id


def test_workspace_member_fixture_can_call_the_api(
    client: TestClient,
    workspace_member: WorkspaceMember,
    auth_headers: dict[str, str],
) -> None:
    response = client.get(
        f"/api/v1/workspaces/{workspace_member.workspace_id}",
        headers=auth_headers,
    )

    assert workspace_member.role == WorkspaceRole.OWNER
    assert response.status_code == 200
    assert response.json()["id"] == str(workspace_member.workspace_id)


def test_login_tokens_fixture_issues_usable_tokens(
    client: TestClient,
    login_tokens: dict[str, str],
) -> None:
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {login_tokens['access_token']}"},
    )

    assert response.status_code == 200
    assert login_tokens["refresh_token"]
