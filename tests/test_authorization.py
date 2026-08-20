import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import (
    require_organization_role,
    require_roles,
    require_workspace_role,
)
from app.core.config import settings
from app.core.security import TOKEN_TYPE_ACCESS, create_access_token, hash_password
from app.db.models import (
    Organization,
    User,
    UserRole,
    Workspace,
    WorkspaceMember,
    WorkspaceRole,
)
from app.enums import WorkspacePermission

AUTHZ_PREFIX = "/api/v1/_authz"
VALID_PASSWORD = "SecurePassword123!"


def _auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _headers_for(user: User) -> dict[str, str]:
    return _auth_header(create_access_token(user.id))


def _create_user(db_session: Session, *, role: UserRole = UserRole.USER) -> User:
    user = User(
        first_name="Test",
        last_name="User",
        email=f"user-{uuid.uuid4().hex[:8]}@example.com",
        password_hash=hash_password(VALID_PASSWORD),
        role=role,
        is_verified=True,
        is_active=True,
    )
    db_session.add(user)
    db_session.flush()
    return user


def _create_workspace(db_session: Session) -> Workspace:
    organization = Organization(
        name="Org",
        slug=f"org-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(organization)
    db_session.flush()
    workspace = Workspace(
        organization_id=organization.id,
        name="Workspace",
        slug=f"ws-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(workspace)
    db_session.flush()
    return workspace


def _add_member(
    db_session: Session,
    workspace: Workspace,
    user: User,
    role: WorkspaceRole,
) -> WorkspaceMember:
    member = WorkspaceMember(
        workspace_id=workspace.id,
        user_id=user.id,
        role=role,
    )
    db_session.add(member)
    db_session.flush()
    return member


def test_require_roles_rejects_empty_roles() -> None:
    with pytest.raises(ValueError, match="require_roles requires at least one role"):
        require_roles()


def test_require_workspace_role_rejects_empty_roles() -> None:
    with pytest.raises(
        ValueError, match="require_workspace_role requires at least one role"
    ):
        require_workspace_role()


def test_require_organization_role_rejects_empty_roles() -> None:
    with pytest.raises(
        ValueError, match="require_organization_role requires at least one role"
    ):
        require_organization_role()


def test_super_admin_is_allowed_on_admin_route(
    authz_client: TestClient,
    db_session: Session,
) -> None:
    user = _create_user(db_session, role=UserRole.SUPER_ADMIN)

    response = authz_client.get(
        f"{AUTHZ_PREFIX}/admin-only", headers=_headers_for(user)
    )

    assert response.status_code == 200
    assert response.json()["role"] == UserRole.SUPER_ADMIN.value


def test_admin_is_allowed_on_admin_route(
    authz_client: TestClient,
    db_session: Session,
) -> None:
    user = _create_user(db_session, role=UserRole.ADMIN)

    response = authz_client.get(
        f"{AUTHZ_PREFIX}/admin-only", headers=_headers_for(user)
    )

    assert response.status_code == 200
    assert response.json()["role"] == UserRole.ADMIN.value


def test_user_is_denied_on_admin_route(
    authz_client: TestClient,
    db_session: Session,
) -> None:
    user = _create_user(db_session, role=UserRole.USER)

    response = authz_client.get(
        f"{AUTHZ_PREFIX}/admin-only", headers=_headers_for(user)
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Not authorized"


def test_user_and_admin_are_denied_on_super_admin_route(
    authz_client: TestClient,
    db_session: Session,
) -> None:
    regular = _create_user(db_session, role=UserRole.USER)
    admin = _create_user(db_session, role=UserRole.ADMIN)
    super_admin = _create_user(db_session, role=UserRole.SUPER_ADMIN)

    assert (
        authz_client.get(
            f"{AUTHZ_PREFIX}/super-admin-only", headers=_headers_for(regular)
        ).status_code
        == 403
    )
    assert (
        authz_client.get(
            f"{AUTHZ_PREFIX}/super-admin-only", headers=_headers_for(admin)
        ).status_code
        == 403
    )
    assert (
        authz_client.get(
            f"{AUTHZ_PREFIX}/super-admin-only", headers=_headers_for(super_admin)
        ).status_code
        == 200
    )


def test_all_global_roles_can_access_user_level_route(
    authz_client: TestClient,
    db_session: Session,
) -> None:
    for role in (UserRole.USER, UserRole.ADMIN, UserRole.SUPER_ADMIN):
        user = _create_user(db_session, role=role)
        response = authz_client.get(
            f"{AUTHZ_PREFIX}/authenticated", headers=_headers_for(user)
        )
        assert response.status_code == 200, role


def test_missing_token_is_unauthorized(authz_client: TestClient) -> None:
    response = authz_client.get(f"{AUTHZ_PREFIX}/admin-only")

    assert response.status_code == 401


def test_invalid_token_is_unauthorized(authz_client: TestClient) -> None:
    response = authz_client.get(
        f"{AUTHZ_PREFIX}/admin-only",
        headers=_auth_header("not-a-jwt"),
    )

    assert response.status_code == 401


def test_valid_user_with_insufficient_role_is_forbidden(
    authz_client: TestClient,
    db_session: Session,
) -> None:
    user = _create_user(db_session, role=UserRole.USER)

    response = authz_client.get(
        f"{AUTHZ_PREFIX}/admin-only", headers=_headers_for(user)
    )

    assert response.status_code == 403


def test_role_change_is_enforced_from_database(
    authz_client: TestClient,
    db_session: Session,
) -> None:
    user = _create_user(db_session, role=UserRole.ADMIN)
    headers = _headers_for(user)

    assert (
        authz_client.get(f"{AUTHZ_PREFIX}/admin-only", headers=headers).status_code
        == 200
    )

    user.role = UserRole.USER
    db_session.flush()

    response = authz_client.get(f"{AUTHZ_PREFIX}/admin-only", headers=headers)
    assert response.status_code == 403


def test_jwt_role_claim_is_ignored(
    authz_client: TestClient,
    db_session: Session,
) -> None:
    user = _create_user(db_session, role=UserRole.USER)
    now = datetime.now(UTC)
    forged = jwt.encode(
        {
            "sub": str(user.id),
            "type": TOKEN_TYPE_ACCESS,
            "role": UserRole.SUPER_ADMIN.value,
            "iat": now,
            "exp": now + timedelta(minutes=15),
        },
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    response = authz_client.get(
        f"{AUTHZ_PREFIX}/admin-only",
        headers=_auth_header(forged),
    )

    assert response.status_code == 403


def test_request_body_role_cannot_elevate_privileges(
    authz_client: TestClient,
    db_session: Session,
) -> None:
    user = _create_user(db_session, role=UserRole.USER)

    response = authz_client.post(
        f"{AUTHZ_PREFIX}/admin-only",
        headers=_headers_for(user),
        json={"role": UserRole.ADMIN.value},
    )

    assert response.status_code == 403


def test_workspace_owner_can_perform_owner_and_admin_operations(
    authz_client: TestClient,
    db_session: Session,
) -> None:
    user = _create_user(db_session)
    workspace = _create_workspace(db_session)
    _add_member(db_session, workspace, user, WorkspaceRole.OWNER)
    headers = _headers_for(user)

    assert (
        authz_client.get(
            f"{AUTHZ_PREFIX}/workspaces/{workspace.id}/owner", headers=headers
        ).status_code
        == 200
    )
    assert (
        authz_client.get(
            f"{AUTHZ_PREFIX}/workspaces/{workspace.id}/admin", headers=headers
        ).status_code
        == 200
    )
    assert (
        authz_client.get(
            f"{AUTHZ_PREFIX}/workspaces/{workspace.id}/permissions/delete",
            headers=headers,
        ).status_code
        == 200
    )


def test_workspace_admin_can_administer_but_not_act_as_owner(
    authz_client: TestClient,
    db_session: Session,
) -> None:
    user = _create_user(db_session)
    workspace = _create_workspace(db_session)
    _add_member(db_session, workspace, user, WorkspaceRole.ADMIN)
    headers = _headers_for(user)

    assert (
        authz_client.get(
            f"{AUTHZ_PREFIX}/workspaces/{workspace.id}/admin", headers=headers
        ).status_code
        == 200
    )
    assert (
        authz_client.get(
            f"{AUTHZ_PREFIX}/workspaces/{workspace.id}/permissions/members",
            headers=headers,
        ).status_code
        == 200
    )
    assert (
        authz_client.get(
            f"{AUTHZ_PREFIX}/workspaces/{workspace.id}/owner", headers=headers
        ).status_code
        == 403
    )
    assert (
        authz_client.get(
            f"{AUTHZ_PREFIX}/workspaces/{workspace.id}/permissions/delete",
            headers=headers,
        ).status_code
        == 403
    )


def test_workspace_member_cannot_perform_admin_operations(
    authz_client: TestClient,
    db_session: Session,
) -> None:
    user = _create_user(db_session)
    workspace = _create_workspace(db_session)
    _add_member(db_session, workspace, user, WorkspaceRole.MEMBER)
    headers = _headers_for(user)

    assert (
        authz_client.get(
            f"{AUTHZ_PREFIX}/workspaces/{workspace.id}", headers=headers
        ).status_code
        == 200
    )
    assert (
        authz_client.get(
            f"{AUTHZ_PREFIX}/workspaces/{workspace.id}/admin", headers=headers
        ).status_code
        == 403
    )
    assert (
        authz_client.get(
            f"{AUTHZ_PREFIX}/workspaces/{workspace.id}/permissions/members",
            headers=headers,
        ).status_code
        == 403
    )


def test_user_can_access_own_workspace_but_not_another(
    authz_client: TestClient,
    db_session: Session,
) -> None:
    user = _create_user(db_session)
    workspace_a = _create_workspace(db_session)
    workspace_b = _create_workspace(db_session)
    _add_member(db_session, workspace_a, user, WorkspaceRole.MEMBER)
    headers = _headers_for(user)

    allowed = authz_client.get(
        f"{AUTHZ_PREFIX}/workspaces/{workspace_a.id}", headers=headers
    )
    denied = authz_client.get(
        f"{AUTHZ_PREFIX}/workspaces/{workspace_b.id}", headers=headers
    )

    assert allowed.status_code == 200
    assert allowed.json()["workspace_id"] == str(workspace_a.id)
    assert denied.status_code == 403
    assert denied.json()["detail"] == "Not authorized"


def test_idor_path_swap_does_not_authorize_other_workspace(
    authz_client: TestClient,
    db_session: Session,
) -> None:
    user = _create_user(db_session)
    workspace_a = _create_workspace(db_session)
    workspace_b = _create_workspace(db_session)
    _add_member(db_session, workspace_a, user, WorkspaceRole.OWNER)
    headers = _headers_for(user)

    response = authz_client.get(
        f"{AUTHZ_PREFIX}/workspaces/{workspace_b.id}/probe/{workspace_a.id}",
        headers=headers,
    )

    assert response.status_code == 403


def test_global_admin_does_not_bypass_workspace_isolation(
    authz_client: TestClient,
    db_session: Session,
) -> None:
    admin = _create_user(db_session, role=UserRole.ADMIN)
    workspace = _create_workspace(db_session)

    response = authz_client.get(
        f"{AUTHZ_PREFIX}/workspaces/{workspace.id}",
        headers=_headers_for(admin),
    )

    assert response.status_code == 403


def test_super_admin_does_not_bypass_workspace_checks_by_default(
    authz_client: TestClient,
    db_session: Session,
) -> None:
    super_admin = _create_user(db_session, role=UserRole.SUPER_ADMIN)
    workspace = _create_workspace(db_session)
    headers = _headers_for(super_admin)

    assert (
        authz_client.get(
            f"{AUTHZ_PREFIX}/workspaces/{workspace.id}", headers=headers
        ).status_code
        == 403
    )
    assert (
        authz_client.get(
            f"{AUTHZ_PREFIX}/workspaces/{workspace.id}/admin", headers=headers
        ).status_code
        == 403
    )


def test_super_admin_can_bypass_when_explicitly_allowed(
    authz_client: TestClient,
    db_session: Session,
) -> None:
    super_admin = _create_user(db_session, role=UserRole.SUPER_ADMIN)
    workspace = _create_workspace(db_session)

    response = authz_client.get(
        f"{AUTHZ_PREFIX}/workspaces/{workspace.id}/admin-with-super",
        headers=_headers_for(super_admin),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["workspace_id"] == str(workspace.id)
    assert body["via_super_admin"] is True
    assert body["role"] is None


def test_super_admin_bypass_still_requires_existing_workspace(
    authz_client: TestClient,
    db_session: Session,
) -> None:
    super_admin = _create_user(db_session, role=UserRole.SUPER_ADMIN)

    response = authz_client.get(
        f"{AUTHZ_PREFIX}/workspaces/{uuid.uuid4()}/admin-with-super",
        headers=_headers_for(super_admin),
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Workspace not found"


def test_non_member_is_denied(authz_client: TestClient, db_session: Session) -> None:
    user = _create_user(db_session)
    workspace = _create_workspace(db_session)

    response = authz_client.get(
        f"{AUTHZ_PREFIX}/workspaces/{workspace.id}",
        headers=_headers_for(user),
    )

    assert response.status_code == 403


def test_removed_membership_is_denied(
    authz_client: TestClient,
    db_session: Session,
) -> None:
    user = _create_user(db_session)
    workspace = _create_workspace(db_session)
    member = _add_member(db_session, workspace, user, WorkspaceRole.ADMIN)
    headers = _headers_for(user)

    assert (
        authz_client.get(
            f"{AUTHZ_PREFIX}/workspaces/{workspace.id}/admin", headers=headers
        ).status_code
        == 200
    )

    db_session.delete(member)
    db_session.flush()

    response = authz_client.get(
        f"{AUTHZ_PREFIX}/workspaces/{workspace.id}/admin", headers=headers
    )
    assert response.status_code == 403


def test_invalid_workspace_id_is_unprocessable(
    authz_client: TestClient, db_session: Session
) -> None:
    user = _create_user(db_session)

    response = authz_client.get(
        f"{AUTHZ_PREFIX}/workspaces/not-a-uuid",
        headers=_headers_for(user),
    )

    assert response.status_code == 422


def test_unknown_workspace_is_not_found(
    authz_client: TestClient,
    db_session: Session,
) -> None:
    user = _create_user(db_session)

    response = authz_client.get(
        f"{AUTHZ_PREFIX}/workspaces/{uuid.uuid4()}",
        headers=_headers_for(user),
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Workspace not found"


def test_authorization_errors_do_not_reveal_membership_details(
    authz_client: TestClient,
    db_session: Session,
) -> None:
    outsider = _create_user(db_session)
    member = _create_user(db_session)
    workspace = _create_workspace(db_session)
    _add_member(db_session, workspace, member, WorkspaceRole.OWNER)

    response = authz_client.get(
        f"{AUTHZ_PREFIX}/workspaces/{workspace.id}/admin",
        headers=_headers_for(outsider),
    )

    assert response.status_code == 403
    detail = response.json()["detail"]
    assert detail == "Not authorized"
    assert member.email not in response.text
    assert "OWNER" not in response.text


def test_workspace_authorization_requires_authentication(
    authz_client: TestClient,
    db_session: Session,
) -> None:
    workspace = _create_workspace(db_session)

    missing = authz_client.get(f"{AUTHZ_PREFIX}/workspaces/{workspace.id}")
    invalid = authz_client.get(
        f"{AUTHZ_PREFIX}/workspaces/{workspace.id}",
        headers=_auth_header("not-a-jwt"),
    )

    assert missing.status_code == 401
    assert invalid.status_code == 401


def test_require_roles_dependency_rejects_insufficient_user(
    db_session: Session,
) -> None:
    user = _create_user(db_session, role=UserRole.USER)
    dependency = require_roles(UserRole.ADMIN)

    with pytest.raises(HTTPException) as exc_info:
        dependency(current_user=user)

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Not authorized"


def test_require_roles_dependency_accepts_higher_role(db_session: Session) -> None:
    user = _create_user(db_session, role=UserRole.SUPER_ADMIN)
    dependency = require_roles(UserRole.ADMIN)

    assert dependency(current_user=user) is user


def test_workspace_permission_enum_matches_documented_set() -> None:
    assert {permission.value for permission in WorkspacePermission} == {
        "workspace:read",
        "workspace:update",
        "workspace:delete",
        "member:read",
        "member:add",
        "member:update",
        "member:remove",
        "data_source:read",
        "data_source:query",
        "data_source:create",
        "data_source:update",
        "data_source:delete",
        "data_source:test",
    }
