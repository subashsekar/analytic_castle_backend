import uuid

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import create_access_token, hash_password
from app.db.models import (
    Organization,
    User,
    UserRole,
    Workspace,
    WorkspaceMember,
    WorkspaceRole,
)

ORG_PREFIX = "/api/v1/organizations"
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


def _create_org_via_api(client: TestClient, user: User, name: str = "Castle Labs"):
    return client.post(ORG_PREFIX, headers=_auth_header(user), json={"name": name})


def test_create_organization_makes_creator_workspace_owner(
    client: TestClient,
    db_session: Session,
) -> None:
    user = _create_user(db_session)

    response = _create_org_via_api(client, user, "Sales Analytics")

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Sales Analytics"
    assert body["slug"] == "sales-analytics"
    assert "password_hash" not in body

    organization_id = uuid.UUID(body["id"])
    workspace = db_session.scalar(
        select(Workspace).where(Workspace.organization_id == organization_id)
    )
    assert workspace is not None
    assert workspace.name == "Sales Analytics"
    member = db_session.scalar(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == workspace.id,
            WorkspaceMember.user_id == user.id,
        )
    )
    assert member is not None
    assert member.role == WorkspaceRole.OWNER
    assert user.role == UserRole.USER


def test_create_organization_rejects_unauthenticated_and_invalid_names(
    client: TestClient,
    db_session: Session,
) -> None:
    user = _create_user(db_session)

    assert client.post(ORG_PREFIX, json={"name": "Acme"}).status_code == 401
    assert (
        client.post(
            ORG_PREFIX, headers=_auth_header(user), json={"name": "   "}
        ).status_code
        == 422
    )
    assert (
        client.post(
            ORG_PREFIX, headers=_auth_header(user), json={"name": "@@@"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            ORG_PREFIX,
            headers=_auth_header(user),
            json={"name": "Acme", "role": "SUPER_ADMIN"},
        ).status_code
        == 422
    )


def test_create_organization_avoids_duplicate_slugs(
    client: TestClient,
    db_session: Session,
) -> None:
    user = _create_user(db_session)

    first = _create_org_via_api(client, user, "Acme")
    second = _create_org_via_api(client, user, "Acme")

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["slug"] == "acme"
    assert second.json()["slug"] == "acme-2"


def test_organization_routes_require_authentication(client: TestClient) -> None:
    organization_id = uuid.uuid4()

    assert client.get(ORG_PREFIX).status_code == 401
    assert client.get(f"{ORG_PREFIX}/{organization_id}").status_code == 401
    assert (
        client.patch(f"{ORG_PREFIX}/{organization_id}", json={"name": "X"}).status_code
        == 401
    )
    assert client.delete(f"{ORG_PREFIX}/{organization_id}").status_code == 401


def test_list_organizations_returns_only_accessible(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    outsider = _create_user(db_session)
    created = _create_org_via_api(client, owner, "Owner Org")
    assert created.status_code == 201

    owner_list = client.get(ORG_PREFIX, headers=_auth_header(owner))
    outsider_list = client.get(ORG_PREFIX, headers=_auth_header(outsider))

    assert owner_list.status_code == 200
    assert [item["name"] for item in owner_list.json()] == ["Owner Org"]
    assert outsider_list.status_code == 200
    assert outsider_list.json() == []


def test_super_admin_can_list_all_organizations(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    super_admin = _create_user(db_session, role=UserRole.SUPER_ADMIN)
    global_admin = _create_user(db_session, role=UserRole.ADMIN)
    created = _create_org_via_api(client, owner, "Hidden Org")
    assert created.status_code == 201

    super_list = client.get(ORG_PREFIX, headers=_auth_header(super_admin))
    admin_list = client.get(ORG_PREFIX, headers=_auth_header(global_admin))

    assert any(item["name"] == "Hidden Org" for item in super_list.json())
    assert admin_list.json() == []


def test_get_update_delete_organization_enforces_access(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    outsider = _create_user(db_session)
    created = _create_org_via_api(client, owner, "Castle")
    organization_id = created.json()["id"]

    allowed = client.get(f"{ORG_PREFIX}/{organization_id}", headers=_auth_header(owner))
    denied = client.get(
        f"{ORG_PREFIX}/{organization_id}", headers=_auth_header(outsider)
    )
    missing = client.get(f"{ORG_PREFIX}/{uuid.uuid4()}", headers=_auth_header(owner))

    assert allowed.status_code == 200
    assert allowed.json()["name"] == "Castle"
    assert denied.status_code == 404
    assert denied.json()["detail"] == "Organization not found"
    assert missing.status_code == 404

    updated = client.patch(
        f"{ORG_PREFIX}/{organization_id}",
        headers=_auth_header(owner),
        json={"name": "Castle Updated"},
    )
    assert updated.status_code == 200
    assert updated.json()["name"] == "Castle Updated"
    assert updated.json()["slug"] == "castle-updated"

    forbidden_update = client.patch(
        f"{ORG_PREFIX}/{organization_id}",
        headers=_auth_header(outsider),
        json={"name": "Hacked"},
    )
    assert forbidden_update.status_code == 404

    extra_field = client.patch(
        f"{ORG_PREFIX}/{organization_id}",
        headers=_auth_header(owner),
        json={"name": "Castle Updated", "owner_id": str(outsider.id)},
    )
    assert extra_field.status_code == 422

    deleted = client.delete(
        f"{ORG_PREFIX}/{organization_id}", headers=_auth_header(owner)
    )
    assert deleted.status_code == 200
    assert deleted.json()["detail"] == "Organization deleted"
    assert db_session.get(Organization, uuid.UUID(organization_id)) is None
    assert (
        db_session.scalar(
            select(Workspace).where(
                Workspace.organization_id == uuid.UUID(organization_id)
            )
        )
        is None
    )


def test_workspace_member_cannot_update_or_delete_organization(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    member_user = _create_user(db_session)
    created = _create_org_via_api(client, owner, "Restricted Org")
    organization_id = uuid.UUID(created.json()["id"])
    workspace = db_session.scalar(
        select(Workspace).where(Workspace.organization_id == organization_id)
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

    update = client.patch(
        f"{ORG_PREFIX}/{organization_id}",
        headers=_auth_header(member_user),
        json={"name": "Nope"},
    )
    delete = client.delete(
        f"{ORG_PREFIX}/{organization_id}", headers=_auth_header(member_user)
    )

    assert update.status_code == 403
    assert delete.status_code == 403


def test_workspace_admin_cannot_control_organization_by_creating_workspace(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    admin_user = _create_user(db_session)
    created = _create_org_via_api(client, owner, "Org")
    organization_id = uuid.UUID(created.json()["id"])
    workspace = db_session.scalar(
        select(Workspace).where(Workspace.organization_id == organization_id)
    )
    assert workspace is not None
    db_session.add(
        WorkspaceMember(
            workspace_id=workspace.id,
            user_id=admin_user.id,
            role=WorkspaceRole.ADMIN,
        )
    )
    db_session.flush()

    second = client.post(
        "/api/v1/workspaces",
        headers=_auth_header(admin_user),
        json={"organization_id": str(organization_id), "name": "Second"},
    )
    assert second.status_code == 201
    second_member = db_session.scalar(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == uuid.UUID(second.json()["id"]),
            WorkspaceMember.user_id == admin_user.id,
        )
    )
    assert second_member is not None
    assert second_member.role == WorkspaceRole.OWNER

    update = client.patch(
        f"{ORG_PREFIX}/{organization_id}",
        headers=_auth_header(admin_user),
        json={"name": "Hijacked"},
    )
    delete = client.delete(
        f"{ORG_PREFIX}/{organization_id}", headers=_auth_header(admin_user)
    )

    assert update.status_code == 403
    assert delete.status_code == 403
    assert db_session.get(Organization, organization_id) is not None
    assert (
        client.get(
            f"{ORG_PREFIX}/{organization_id}", headers=_auth_header(owner)
        ).json()["name"]
        == "Org"
    )

    owner_delete = client.delete(
        f"{ORG_PREFIX}/{organization_id}", headers=_auth_header(owner)
    )
    assert owner_delete.status_code == 200
    assert db_session.get(Organization, organization_id) is None


def test_delete_organization_does_not_remove_unrelated_records(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    first = _create_org_via_api(client, owner, "Keep Me")
    second = _create_org_via_api(client, owner, "Delete Me")
    keep_id = uuid.UUID(first.json()["id"])
    delete_id = uuid.UUID(second.json()["id"])

    response = client.delete(f"{ORG_PREFIX}/{delete_id}", headers=_auth_header(owner))

    assert response.status_code == 200
    assert db_session.get(Organization, keep_id) is not None
    assert db_session.get(Organization, delete_id) is None
    assert (
        db_session.scalar(select(Workspace).where(Workspace.organization_id == keep_id))
        is not None
    )
