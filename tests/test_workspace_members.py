import uuid

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import create_access_token, hash_password
from app.db.models import User, UserRole, Workspace, WorkspaceMember, WorkspaceRole

ORG_PREFIX = "/api/v1/organizations"
WS_PREFIX = "/api/v1/workspaces"
VALID_PASSWORD = "SecurePassword123!"


def _auth_header(user: User) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user.id)}"}


def _create_user(
    db_session: Session,
    *,
    role: UserRole = UserRole.USER,
    first_name: str = "Test",
    last_name: str = "User",
) -> User:
    user = User(
        first_name=first_name,
        last_name=last_name,
        email=f"user-{uuid.uuid4().hex[:8]}@example.com",
        password_hash=hash_password(VALID_PASSWORD),
        role=role,
        is_verified=True,
        is_active=True,
    )
    db_session.add(user)
    db_session.flush()
    return user


def _seed_workspace(
    client: TestClient,
    db_session: Session,
    owner: User,
) -> Workspace:
    created = client.post(ORG_PREFIX, headers=_auth_header(owner), json={"name": "Org"})
    assert created.status_code == 201
    workspace = db_session.scalar(
        select(Workspace).where(
            Workspace.organization_id == uuid.UUID(created.json()["id"])
        )
    )
    assert workspace is not None
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


def test_list_members_returns_safe_user_fields(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session, first_name="Ada", last_name="Lovelace")
    workspace = _seed_workspace(client, db_session, owner)

    response = client.get(
        f"{WS_PREFIX}/{workspace.id}/members", headers=_auth_header(owner)
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    member = body[0]
    assert member["user_id"] == str(owner.id)
    assert member["first_name"] == "Ada"
    assert member["last_name"] == "Lovelace"
    assert member["email"] == owner.email
    assert member["role"] == WorkspaceRole.OWNER.value
    assert "password_hash" not in member
    assert "password" not in member
    assert "access_token" not in member
    assert owner.password_hash not in response.text


def test_add_member_and_reject_duplicates_and_invalid_users(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    target = _create_user(db_session, first_name="Grace", last_name="Hopper")
    workspace = _seed_workspace(client, db_session, owner)

    created = client.post(
        f"{WS_PREFIX}/{workspace.id}/members",
        headers=_auth_header(owner),
        json={"user_id": str(target.id), "role": WorkspaceRole.MEMBER.value},
    )
    duplicate = client.post(
        f"{WS_PREFIX}/{workspace.id}/members",
        headers=_auth_header(owner),
        json={"user_id": str(target.id), "role": WorkspaceRole.MEMBER.value},
    )
    missing_user = client.post(
        f"{WS_PREFIX}/{workspace.id}/members",
        headers=_auth_header(owner),
        json={"user_id": str(uuid.uuid4()), "role": WorkspaceRole.MEMBER.value},
    )
    missing_workspace = client.post(
        f"{WS_PREFIX}/{uuid.uuid4()}/members",
        headers=_auth_header(owner),
        json={"user_id": str(target.id), "role": WorkspaceRole.MEMBER.value},
    )

    assert created.status_code == 201
    assert created.json()["email"] == target.email
    assert created.json()["role"] == WorkspaceRole.MEMBER.value
    assert duplicate.status_code == 409
    assert missing_user.status_code == 404
    assert missing_workspace.status_code == 404


def test_member_cannot_perform_admin_member_operations(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    member_user = _create_user(db_session)
    other = _create_user(db_session)
    workspace = _seed_workspace(client, db_session, owner)
    _add_member(db_session, workspace, member_user, WorkspaceRole.MEMBER)

    add = client.post(
        f"{WS_PREFIX}/{workspace.id}/members",
        headers=_auth_header(member_user),
        json={"user_id": str(other.id), "role": WorkspaceRole.MEMBER.value},
    )
    promote_self = client.patch(
        f"{WS_PREFIX}/{workspace.id}/members/{member_user.id}",
        headers=_auth_header(member_user),
        json={"role": WorkspaceRole.ADMIN.value},
    )
    assign_owner = client.post(
        f"{WS_PREFIX}/{workspace.id}/members",
        headers=_auth_header(member_user),
        json={"user_id": str(other.id), "role": WorkspaceRole.OWNER.value},
    )
    listing = client.get(
        f"{WS_PREFIX}/{workspace.id}/members", headers=_auth_header(member_user)
    )

    assert add.status_code == 403
    assert promote_self.status_code == 403
    assert assign_owner.status_code == 403
    assert listing.status_code == 200


def test_admin_cannot_assign_owner_or_admin_or_modify_owner(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    admin_user = _create_user(db_session)
    target = _create_user(db_session)
    workspace = _seed_workspace(client, db_session, owner)
    _add_member(db_session, workspace, admin_user, WorkspaceRole.ADMIN)

    assign_owner = client.post(
        f"{WS_PREFIX}/{workspace.id}/members",
        headers=_auth_header(admin_user),
        json={"user_id": str(target.id), "role": WorkspaceRole.OWNER.value},
    )
    assign_admin = client.post(
        f"{WS_PREFIX}/{workspace.id}/members",
        headers=_auth_header(admin_user),
        json={"user_id": str(target.id), "role": WorkspaceRole.ADMIN.value},
    )
    add_member = client.post(
        f"{WS_PREFIX}/{workspace.id}/members",
        headers=_auth_header(admin_user),
        json={"user_id": str(target.id), "role": WorkspaceRole.MEMBER.value},
    )
    demote_owner = client.patch(
        f"{WS_PREFIX}/{workspace.id}/members/{owner.id}",
        headers=_auth_header(admin_user),
        json={"role": WorkspaceRole.MEMBER.value},
    )
    remove_owner = client.delete(
        f"{WS_PREFIX}/{workspace.id}/members/{owner.id}",
        headers=_auth_header(admin_user),
    )

    assert assign_owner.status_code == 400
    assert assign_admin.status_code == 403
    assert add_member.status_code == 201
    assert demote_owner.status_code == 403
    assert remove_owner.status_code == 403


def test_owner_can_update_and_remove_member(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    target = _create_user(db_session)
    workspace = _seed_workspace(client, db_session, owner)
    _add_member(db_session, workspace, target, WorkspaceRole.MEMBER)

    updated = client.patch(
        f"{WS_PREFIX}/{workspace.id}/members/{target.id}",
        headers=_auth_header(owner),
        json={"role": WorkspaceRole.ADMIN.value},
    )
    owner_self = client.patch(
        f"{WS_PREFIX}/{workspace.id}/members/{owner.id}",
        headers=_auth_header(owner),
        json={"role": WorkspaceRole.ADMIN.value},
    )
    transfer = client.patch(
        f"{WS_PREFIX}/{workspace.id}/members/{target.id}",
        headers=_auth_header(owner),
        json={"role": WorkspaceRole.OWNER.value},
    )
    removed = client.delete(
        f"{WS_PREFIX}/{workspace.id}/members/{target.id}",
        headers=_auth_header(owner),
    )

    assert updated.status_code == 200
    assert updated.json()["role"] == WorkspaceRole.ADMIN.value
    assert owner_self.status_code == 400
    assert transfer.status_code == 400
    assert removed.status_code == 200
    assert (
        db_session.scalar(
            select(WorkspaceMember).where(
                WorkspaceMember.workspace_id == workspace.id,
                WorkspaceMember.user_id == target.id,
            )
        )
        is None
    )


def test_member_can_leave_but_owner_cannot(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    member_user = _create_user(db_session)
    workspace = _seed_workspace(client, db_session, owner)
    _add_member(db_session, workspace, member_user, WorkspaceRole.MEMBER)

    leave = client.delete(
        f"{WS_PREFIX}/{workspace.id}/members/{member_user.id}",
        headers=_auth_header(member_user),
    )
    owner_leave = client.delete(
        f"{WS_PREFIX}/{workspace.id}/members/{owner.id}",
        headers=_auth_header(owner),
    )

    assert leave.status_code == 200
    assert owner_leave.status_code == 400


def test_cross_tenant_member_operations_are_denied(
    client: TestClient,
    db_session: Session,
) -> None:
    owner_a = _create_user(db_session)
    owner_b = _create_user(db_session)
    outsider = _create_user(db_session)
    _seed_workspace(client, db_session, owner_a)
    workspace_b = _seed_workspace(client, db_session, owner_b)

    list_b = client.get(
        f"{WS_PREFIX}/{workspace_b.id}/members", headers=_auth_header(owner_a)
    )
    add_to_b = client.post(
        f"{WS_PREFIX}/{workspace_b.id}/members",
        headers=_auth_header(owner_a),
        json={"user_id": str(outsider.id), "role": WorkspaceRole.MEMBER.value},
    )
    update_b = client.patch(
        f"{WS_PREFIX}/{workspace_b.id}/members/{owner_b.id}",
        headers=_auth_header(owner_a),
        json={"role": WorkspaceRole.MEMBER.value},
    )

    assert list_b.status_code == 403
    assert add_to_b.status_code == 403
    assert update_b.status_code == 403


def test_arbitrary_user_id_cannot_grant_access(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    outsider = _create_user(db_session)
    workspace = _seed_workspace(client, db_session, owner)

    response = client.patch(
        f"{WS_PREFIX}/{workspace.id}/members/{outsider.id}",
        headers=_auth_header(outsider),
        json={"role": WorkspaceRole.OWNER.value},
    )

    assert response.status_code == 403
    assert (
        db_session.scalar(
            select(WorkspaceMember).where(
                WorkspaceMember.workspace_id == workspace.id,
                WorkspaceMember.user_id == outsider.id,
            )
        )
        is None
    )
