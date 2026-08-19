import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import (
    EmailVerificationToken,
    Organization,
    PasswordResetToken,
    User,
    UserRole,
    Workspace,
    WorkspaceMember,
    WorkspaceRole,
)
from app.schemas import OrganizationRead, UserRead, WorkspaceMemberRead, WorkspaceRead


def _user(**overrides: object) -> User:
    values = {
        "first_name": "Ada",
        "last_name": "Lovelace",
        "email": f"ada-{uuid.uuid4()}@example.com",
        "password_hash": "hashed-password",
        "role": UserRole.USER,
    }
    values.update(overrides)
    return User(**values)


def _organization(**overrides: object) -> Organization:
    values = {
        "name": "AnalyticCastle",
        "slug": f"analyticcastle-{uuid.uuid4().hex[:8]}",
    }
    values.update(overrides)
    return Organization(**values)


def test_user_can_be_created(db_session: Session) -> None:
    user = _user(email="ada@example.com")
    db_session.add(user)
    db_session.flush()

    assert user.id is not None
    assert user.email == "ada@example.com"
    assert user.role == UserRole.USER
    assert user.is_verified is False
    assert user.is_active is True
    assert user.created_at is not None
    assert user.updated_at is not None


def test_user_email_must_be_unique(db_session: Session) -> None:
    db_session.add(_user(email="duplicate@example.com"))
    db_session.flush()

    db_session.add(_user(email="duplicate@example.com", first_name="Grace"))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_user_email_uniqueness_is_case_insensitive(db_session: Session) -> None:
    user = _user(email="User@example.com")
    db_session.add(user)
    db_session.flush()

    assert user.email == "user@example.com"

    db_session.add(_user(email="USER@example.com", first_name="Grace"))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_database_rejects_mixed_case_duplicate_email(db_session: Session) -> None:
    db_session.add(_user(email="ada@example.com"))
    db_session.flush()

    with pytest.raises(IntegrityError):
        db_session.execute(
            text(
                "INSERT INTO users "
                "(id, first_name, last_name, email, password_hash, role) "
                "VALUES (gen_random_uuid(), 'Grace', 'Hopper', "
                "'Ada@example.com', 'hashed-password', 'USER')"
            )
        )


def test_organization_can_be_created(db_session: Session) -> None:
    organization = _organization(name="Castle Labs", slug="castle-labs")
    db_session.add(organization)
    db_session.flush()

    assert organization.id is not None
    assert organization.name == "Castle Labs"
    assert organization.slug == "castle-labs"
    assert organization.created_at is not None
    assert organization.updated_at is not None


def test_organization_slug_must_be_unique(db_session: Session) -> None:
    db_session.add(_organization(slug="shared-slug"))
    db_session.flush()

    db_session.add(_organization(name="Other Org", slug="shared-slug"))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_workspace_belongs_to_organization(db_session: Session) -> None:
    organization = _organization()
    db_session.add(organization)
    db_session.flush()

    workspace = Workspace(
        organization_id=organization.id,
        name="Analytics",
        slug="analytics",
    )
    db_session.add(workspace)
    db_session.flush()
    db_session.refresh(organization)

    assert workspace.organization_id == organization.id
    assert workspace.organization.id == organization.id
    assert workspace in organization.workspaces


def test_workspace_invalid_organization_fk_is_rejected(db_session: Session) -> None:
    workspace = Workspace(
        organization_id=uuid.uuid4(),
        name="Orphan",
        slug="orphan",
    )
    db_session.add(workspace)

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_workspace_slug_is_unique_within_organization(db_session: Session) -> None:
    organization = _organization()
    db_session.add(organization)
    db_session.flush()

    db_session.add(
        Workspace(organization_id=organization.id, name="Analytics", slug="analytics")
    )
    db_session.flush()

    db_session.add(
        Workspace(organization_id=organization.id, name="Analytics 2", slug="analytics")
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_workspace_slug_can_repeat_in_another_organization(db_session: Session) -> None:
    organization = _organization()
    other_organization = _organization()
    db_session.add_all([organization, other_organization])
    db_session.flush()

    first = Workspace(
        organization_id=organization.id,
        name="Analytics",
        slug="analytics",
    )
    second = Workspace(
        organization_id=other_organization.id,
        name="Analytics",
        slug="analytics",
    )
    db_session.add_all([first, second])
    db_session.flush()

    assert first.id is not None
    assert second.id is not None
    assert first.slug == second.slug
    assert first.organization_id != second.organization_id


def test_user_can_join_workspace(db_session: Session) -> None:
    user = _user()
    organization = _organization()
    db_session.add_all([user, organization])
    db_session.flush()

    workspace = Workspace(
        organization_id=organization.id,
        name="Analytics",
        slug="analytics",
    )
    db_session.add(workspace)
    db_session.flush()

    member = WorkspaceMember(
        workspace_id=workspace.id,
        user_id=user.id,
        role=WorkspaceRole.MEMBER,
    )
    db_session.add(member)
    db_session.flush()

    assert member.id is not None
    assert member.workspace.id == workspace.id
    assert member.user.id == user.id
    assert member in workspace.members
    assert member in user.workspace_memberships


def test_duplicate_workspace_membership_is_rejected(db_session: Session) -> None:
    user = _user()
    organization = _organization()
    db_session.add_all([user, organization])
    db_session.flush()

    workspace = Workspace(
        organization_id=organization.id,
        name="Analytics",
        slug="analytics",
    )
    db_session.add(workspace)
    db_session.flush()

    db_session.add(
        WorkspaceMember(
            workspace_id=workspace.id,
            user_id=user.id,
            role=WorkspaceRole.MEMBER,
        )
    )
    db_session.flush()

    db_session.add(
        WorkspaceMember(
            workspace_id=workspace.id,
            user_id=user.id,
            role=WorkspaceRole.ADMIN,
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_workspace_member_invalid_foreign_keys_are_rejected(
    db_session: Session,
) -> None:
    member = WorkspaceMember(
        workspace_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        role=WorkspaceRole.OWNER,
    )
    db_session.add(member)

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_email_verification_token_belongs_to_user(db_session: Session) -> None:
    user = _user()
    db_session.add(user)
    db_session.flush()

    token = EmailVerificationToken(
        user_id=user.id,
        token_hash="a" * 64,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    db_session.add(token)
    db_session.flush()

    assert token.id is not None
    assert token.user.id == user.id
    assert token.used_at is None
    assert token in user.email_verification_tokens


def test_email_verification_token_invalid_user_fk_is_rejected(
    db_session: Session,
) -> None:
    token = EmailVerificationToken(
        user_id=uuid.uuid4(),
        token_hash="b" * 64,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    db_session.add(token)

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_password_reset_token_belongs_to_user(db_session: Session) -> None:
    user = _user()
    db_session.add(user)
    db_session.flush()

    token = PasswordResetToken(
        user_id=user.id,
        token_hash="c" * 64,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    db_session.add(token)
    db_session.flush()

    assert token.id is not None
    assert token.user.id == user.id
    assert token.used_at is None
    assert token in user.password_reset_tokens


def test_password_reset_token_invalid_user_fk_is_rejected(db_session: Session) -> None:
    token = PasswordResetToken(
        user_id=uuid.uuid4(),
        token_hash="d" * 64,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    db_session.add(token)

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_user_read_schema_excludes_password_hash(db_session: Session) -> None:
    user = _user()
    db_session.add(user)
    db_session.flush()

    payload = UserRead.model_validate(user)

    assert "password_hash" not in UserRead.model_fields
    assert "password_hash" not in payload.model_dump()
    assert set(UserRead.model_fields) == {
        "id",
        "first_name",
        "last_name",
        "email",
        "role",
        "is_verified",
        "is_active",
        "created_at",
        "updated_at",
    }


def test_read_schemas_map_from_models(db_session: Session) -> None:
    user = _user()
    organization = _organization()
    db_session.add_all([user, organization])
    db_session.flush()

    workspace = Workspace(
        organization_id=organization.id,
        name="Analytics",
        slug="analytics",
    )
    db_session.add(workspace)
    db_session.flush()

    member = WorkspaceMember(
        workspace_id=workspace.id,
        user_id=user.id,
        role=WorkspaceRole.OWNER,
    )
    db_session.add(member)
    db_session.flush()

    assert OrganizationRead.model_validate(organization).slug == organization.slug
    assert WorkspaceRead.model_validate(workspace).organization_id == organization.id
    member_payload = WorkspaceMemberRead.model_validate(member)
    assert member_payload.role == WorkspaceRole.OWNER
    assert member_payload.email == user.email
    assert member_payload.first_name == user.first_name
    assert "password_hash" not in member_payload.model_dump()
