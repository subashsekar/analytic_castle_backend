from __future__ import annotations

import uuid

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.ai.memory import ConversationMemoryService, CreateConversationParams
from app.core.security import create_access_token, hash_password
from app.db.models import User, UserRole, WorkspaceMember, WorkspaceRole

PREFIX = "/api/v1/workspaces"
ORG_PREFIX = "/api/v1/organizations"
WS_PREFIX = "/api/v1/workspaces"
VALID_PASSWORD = "SecurePassword123!"


def _auth_header(user: User) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user.id)}"}


def _create_user(db_session: Session) -> User:
    user = User(
        first_name="Test",
        last_name="User",
        email=f"user-{uuid.uuid4().hex[:8]}@example.com",
        password_hash=hash_password(VALID_PASSWORD),
        role=UserRole.USER,
        is_verified=True,
        is_active=True,
    )
    db_session.add(user)
    db_session.flush()
    return user


def _create_organization(client: TestClient, user: User) -> dict:
    response = client.post(
        ORG_PREFIX,
        headers=_auth_header(user),
        json={"name": "Org"},
    )
    assert response.status_code == 201
    return response.json()


def _founding_workspace(client: TestClient, user: User, organization_id: str) -> dict:
    response = client.get(
        WS_PREFIX,
        headers=_auth_header(user),
        params={"organization_id": organization_id},
    )
    assert response.status_code == 200
    return response.json()[0]


def test_conversation_api_create_list_and_messages(
    client: TestClient,
    db_session: Session,
) -> None:
    user = _create_user(db_session)
    organization = _create_organization(client, user)
    workspace = _founding_workspace(client, user, organization["id"])

    create_response = client.post(
        f"{PREFIX}/{workspace['id']}/conversations",
        headers=_auth_header(user),
        json={"initial_message": "Hello"},
    )
    assert create_response.status_code == 201
    conversation = create_response.json()
    conversation_id = conversation["conversation_id"]

    list_response = client.get(
        f"{PREFIX}/{workspace['id']}/conversations",
        headers=_auth_header(user),
    )
    assert list_response.status_code == 200
    assert list_response.json()["total"] == 1

    append_response = client.post(
        f"{PREFIX}/{workspace['id']}/conversations/{conversation_id}/messages",
        headers=_auth_header(user),
        json={
            "role": "user",
            "content": "Hi there",
            "expected_context_version": conversation["conversation_version"],
        },
    )
    assert append_response.status_code == 200
    assert append_response.json()["message_count"] == 2

    messages_response = client.get(
        f"{PREFIX}/{workspace['id']}/conversations/{conversation_id}/messages",
        headers=_auth_header(user),
    )
    assert messages_response.status_code == 200
    assert messages_response.json()["total"] == 2


def test_conversation_api_idor_hidden(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    intruder = _create_user(db_session)
    organization = _create_organization(client, owner)
    workspace = _founding_workspace(client, owner, organization["id"])
    db_session.add(
        WorkspaceMember(
            workspace_id=workspace["id"],
            user_id=intruder.id,
            role=WorkspaceRole.MEMBER,
        )
    )
    db_session.flush()

    service = ConversationMemoryService(db_session)
    record = service.create_conversation(
        CreateConversationParams(
            user_id=owner.id,
            workspace_id=uuid.UUID(workspace["id"]),
            organization_id=uuid.UUID(organization["id"]),
        )
    )
    db_session.commit()

    response = client.get(
        f"{PREFIX}/{workspace['id']}/conversations/{record.conversation_id}",
        headers=_auth_header(intruder),
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found"
