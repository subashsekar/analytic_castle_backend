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


def _create_organization(client: TestClient, user: User, name: str = "Org") -> dict:
    response = client.post(ORG_PREFIX, headers=_auth_header(user), json={"name": name})
    assert response.status_code == 201
    return response.json()


def test_create_workspace_assigns_owner_and_generates_slug(
    client: TestClient,
    db_session: Session,
) -> None:
    user = _create_user(db_session)
    organization = _create_organization(client, user, "Acme")

    response = client.post(
        WS_PREFIX,
        headers=_auth_header(user),
        json={"organization_id": organization["id"], "name": "Sales Analytics"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Sales Analytics"
    assert body["slug"] == "sales-analytics"
    assert body["organization_id"] == organization["id"]

    member = db_session.scalar(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == uuid.UUID(body["id"]),
            WorkspaceMember.user_id == user.id,
        )
    )
    assert member is not None
    assert member.role == WorkspaceRole.OWNER


def test_create_workspace_rejects_invalid_and_foreign_organizations(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    outsider = _create_user(db_session)
    organization = _create_organization(client, owner, "Private Org")

    missing = client.post(
        WS_PREFIX,
        headers=_auth_header(owner),
        json={"organization_id": str(uuid.uuid4()), "name": "Analytics"},
    )
    foreign = client.post(
        WS_PREFIX,
        headers=_auth_header(outsider),
        json={"organization_id": organization["id"], "name": "Analytics"},
    )
    extra = client.post(
        WS_PREFIX,
        headers=_auth_header(owner),
        json={
            "organization_id": organization["id"],
            "name": "Analytics",
            "owner_id": str(outsider.id),
        },
    )
    unauthenticated = client.post(
        WS_PREFIX,
        json={"organization_id": organization["id"], "name": "Analytics"},
    )

    assert missing.status_code == 404
    assert foreign.status_code == 404
    assert extra.status_code == 422
    assert unauthenticated.status_code == 401


def test_create_workspace_handles_duplicate_slugs_within_organization(
    client: TestClient,
    db_session: Session,
) -> None:
    user = _create_user(db_session)
    first_org = _create_organization(client, user, "First Org")
    second_org = _create_organization(client, user, "Second Org")

    first = client.post(
        WS_PREFIX,
        headers=_auth_header(user),
        json={"organization_id": first_org["id"], "name": "Analytics"},
    )
    duplicate = client.post(
        WS_PREFIX,
        headers=_auth_header(user),
        json={"organization_id": first_org["id"], "name": "Analytics"},
    )
    other_org = client.post(
        WS_PREFIX,
        headers=_auth_header(user),
        json={"organization_id": second_org["id"], "name": "Analytics"},
    )

    assert first.json()["slug"] == "analytics"
    assert duplicate.json()["slug"] == "analytics-2"
    assert other_org.json()["slug"] == "analytics"
    assert other_org.json()["organization_id"] == second_org["id"]


def test_workspace_member_cannot_create_another_workspace(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    member_user = _create_user(db_session)
    organization = _create_organization(client, owner, "Org")
    workspace = db_session.scalar(
        select(Workspace).where(
            Workspace.organization_id == uuid.UUID(organization["id"])
        )
    )
    assert workspace is not None
    db_session.add(
        WorkspaceMember(
            workspace_id=workspace.id,
            user_id=member_user.id,
            role=WorkspaceRole.MEMBER,
        )
    )
    db_session.flush()

    response = client.post(
        WS_PREFIX,
        headers=_auth_header(member_user),
        json={"organization_id": organization["id"], "name": "New Space"},
    )

    assert response.status_code == 403


def test_list_workspaces_filters_and_hides_unrelated(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    outsider = _create_user(db_session)
    first = _create_organization(client, owner, "Org One")
    second = _create_organization(client, owner, "Org Two")
    created = client.post(
        WS_PREFIX,
        headers=_auth_header(owner),
        json={"organization_id": first["id"], "name": "Extra"},
    )
    assert created.status_code == 201

    all_mine = client.get(WS_PREFIX, headers=_auth_header(owner))
    filtered = client.get(
        WS_PREFIX,
        headers=_auth_header(owner),
        params={"organization_id": first["id"]},
    )
    other_org = client.get(
        WS_PREFIX,
        headers=_auth_header(owner),
        params={"organization_id": second["id"]},
    )
    outsider_list = client.get(WS_PREFIX, headers=_auth_header(outsider))

    assert all_mine.status_code == 200
    assert {item["name"] for item in all_mine.json()} >= {"Org One", "Org Two", "Extra"}
    assert {item["organization_id"] for item in filtered.json()} == {first["id"]}
    assert {item["name"] for item in other_org.json()} == {"Org Two"}
    assert outsider_list.json() == []


def test_get_update_delete_workspace_enforces_rbac(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    admin_user = _create_user(db_session)
    member_user = _create_user(db_session)
    outsider = _create_user(db_session)
    organization = _create_organization(client, owner, "Org")
    workspace_id = db_session.scalar(
        select(Workspace.id).where(
            Workspace.organization_id == uuid.UUID(organization["id"])
        )
    )
    assert workspace_id is not None
    db_session.add_all(
        [
            WorkspaceMember(
                workspace_id=workspace_id,
                user_id=admin_user.id,
                role=WorkspaceRole.ADMIN,
            ),
            WorkspaceMember(
                workspace_id=workspace_id,
                user_id=member_user.id,
                role=WorkspaceRole.MEMBER,
            ),
        ]
    )
    db_session.flush()

    owner_get = client.get(f"{WS_PREFIX}/{workspace_id}", headers=_auth_header(owner))
    outsider_get = client.get(
        f"{WS_PREFIX}/{workspace_id}", headers=_auth_header(outsider)
    )
    missing = client.get(f"{WS_PREFIX}/{uuid.uuid4()}", headers=_auth_header(owner))

    assert owner_get.status_code == 200
    assert outsider_get.status_code == 403
    assert missing.status_code == 404

    admin_update = client.patch(
        f"{WS_PREFIX}/{workspace_id}",
        headers=_auth_header(admin_user),
        json={"name": "Renamed"},
    )
    member_update = client.patch(
        f"{WS_PREFIX}/{workspace_id}",
        headers=_auth_header(member_user),
        json={"name": "Denied"},
    )
    forbidden_org_change = client.patch(
        f"{WS_PREFIX}/{workspace_id}",
        headers=_auth_header(owner),
        json={"organization_id": str(uuid.uuid4())},
    )

    assert admin_update.status_code == 200
    assert admin_update.json()["name"] == "Renamed"
    assert admin_update.json()["organization_id"] == organization["id"]
    assert member_update.status_code == 403
    assert forbidden_org_change.status_code == 422

    extra = client.post(
        WS_PREFIX,
        headers=_auth_header(owner),
        json={"organization_id": organization["id"], "name": "Extra"},
    )
    assert extra.status_code == 201

    member_delete = client.delete(
        f"{WS_PREFIX}/{workspace_id}", headers=_auth_header(member_user)
    )
    admin_delete = client.delete(
        f"{WS_PREFIX}/{workspace_id}", headers=_auth_header(admin_user)
    )
    owner_delete = client.delete(
        f"{WS_PREFIX}/{workspace_id}", headers=_auth_header(owner)
    )

    assert member_delete.status_code == 403
    assert admin_delete.status_code == 403
    assert owner_delete.status_code == 200
    assert db_session.get(Workspace, workspace_id) is None
    assert (
        db_session.scalar(
            select(WorkspaceMember).where(WorkspaceMember.workspace_id == workspace_id)
        )
        is None
    )


def test_super_admin_can_access_workspace_without_membership(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    super_admin = _create_user(db_session, role=UserRole.SUPER_ADMIN)
    organization = _create_organization(client, owner, "Org")
    workspace_id = db_session.scalar(
        select(Workspace.id).where(
            Workspace.organization_id == uuid.UUID(organization["id"])
        )
    )

    response = client.get(
        f"{WS_PREFIX}/{workspace_id}", headers=_auth_header(super_admin)
    )
    listing = client.get(WS_PREFIX, headers=_auth_header(super_admin))

    assert response.status_code == 200
    assert any(item["id"] == str(workspace_id) for item in listing.json())


def test_cannot_delete_last_workspace_in_organization(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    organization = _create_organization(client, owner, "Org")
    workspace_id = db_session.scalar(
        select(Workspace.id).where(
            Workspace.organization_id == uuid.UUID(organization["id"])
        )
    )
    assert workspace_id is not None

    response = client.delete(f"{WS_PREFIX}/{workspace_id}", headers=_auth_header(owner))

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Cannot delete the last workspace in an organization"
    )
    assert db_session.get(Workspace, workspace_id) is not None
    listed = client.get(ORG_PREFIX, headers=_auth_header(owner))
    assert any(item["id"] == organization["id"] for item in listed.json())
