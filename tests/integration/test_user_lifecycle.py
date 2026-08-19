import uuid

from fastapi.testclient import TestClient

from app.enums import WorkspaceRole
from tests.conftest import VALID_TEST_PASSWORD

AUTH_PREFIX = "/api/v1/auth"
ORG_PREFIX = "/api/v1/organizations"
WS_PREFIX = "/api/v1/workspaces"


def _register(
    client: TestClient, **overrides: object
) -> tuple[dict[str, object], object]:
    payload: dict[str, object] = {
        "first_name": "Flow",
        "last_name": "User",
        "email": f"flow-{uuid.uuid4().hex[:8]}@example.com",
        "password": VALID_TEST_PASSWORD,
    }
    payload.update(overrides)
    return payload, client.post(f"{AUTH_PREFIX}/register", json=payload)


def test_register_login_me_organization_workspace_member_and_authorization(
    client: TestClient,
) -> None:
    owner_payload, register_response = _register(client, first_name="Owner")
    assert register_response.status_code == 201
    owner_id = register_response.json()["id"]

    member_payload, member_register = _register(client, first_name="Member")
    outsider_payload, outsider_register = _register(client, first_name="Outsider")
    assert member_register.status_code == 201
    assert outsider_register.status_code == 201
    member_id = member_register.json()["id"]

    login = client.post(
        f"{AUTH_PREFIX}/login",
        json={"email": owner_payload["email"], "password": VALID_TEST_PASSWORD},
    )
    assert login.status_code == 200
    owner_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    me = client.get(f"{AUTH_PREFIX}/me", headers=owner_headers)
    assert me.status_code == 200
    assert me.json()["id"] == owner_id
    assert me.json()["email"] == owner_payload["email"]

    organization = client.post(
        ORG_PREFIX, headers=owner_headers, json={"name": "Flow Org"}
    )
    assert organization.status_code == 201
    organization_id = organization.json()["id"]

    workspace = client.post(
        WS_PREFIX,
        headers=owner_headers,
        json={"organization_id": organization_id, "name": "Flow Workspace"},
    )
    assert workspace.status_code == 201
    workspace_id = workspace.json()["id"]

    added = client.post(
        f"{WS_PREFIX}/{workspace_id}/members",
        headers=owner_headers,
        json={"user_id": member_id, "role": WorkspaceRole.MEMBER.value},
    )
    assert added.status_code == 201
    assert added.json()["user_id"] == member_id
    assert added.json()["role"] == WorkspaceRole.MEMBER.value

    member_login = client.post(
        f"{AUTH_PREFIX}/login",
        json={"email": member_payload["email"], "password": VALID_TEST_PASSWORD},
    )
    outsider_login = client.post(
        f"{AUTH_PREFIX}/login",
        json={"email": outsider_payload["email"], "password": VALID_TEST_PASSWORD},
    )
    member_headers = {"Authorization": f"Bearer {member_login.json()['access_token']}"}
    outsider_headers = {
        "Authorization": f"Bearer {outsider_login.json()['access_token']}"
    }

    member_can_read = client.get(f"{WS_PREFIX}/{workspace_id}", headers=member_headers)
    outsider_workspace = client.get(
        f"{WS_PREFIX}/{workspace_id}", headers=outsider_headers
    )
    outsider_org = client.get(
        f"{ORG_PREFIX}/{organization_id}", headers=outsider_headers
    )
    member_cannot_add = client.post(
        f"{WS_PREFIX}/{workspace_id}/members",
        headers=member_headers,
        json={
            "user_id": outsider_register.json()["id"],
            "role": WorkspaceRole.MEMBER.value,
        },
    )

    assert member_can_read.status_code == 200
    assert outsider_workspace.status_code == 403
    assert outsider_org.status_code == 404
    assert member_cannot_add.status_code == 403
