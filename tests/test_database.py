import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.db.models import (
    EmailVerificationToken,
    Organization,
    PasswordResetToken,
    RefreshToken,
    User,
    UserRole,
    Workspace,
    WorkspaceMember,
    WorkspaceRole,
)
from app.db.session import SessionLocal, engine, get_db

VALID_PASSWORD = "SecurePassword123!"


def _user(**overrides: object) -> User:
    values: dict[str, object] = {
        "first_name": "Ada",
        "last_name": "Lovelace",
        "email": f"ada-{uuid.uuid4().hex[:8]}@example.com",
        "password_hash": hash_password(VALID_PASSWORD),
        "role": UserRole.USER,
    }
    values.update(overrides)
    return User(**values)


def test_uuid_primary_keys_and_timezone_timestamps(db_session: Session) -> None:
    user = _user()
    db_session.add(user)
    db_session.flush()

    assert isinstance(user.id, uuid.UUID)
    assert user.created_at.tzinfo is not None
    assert user.updated_at.tzinfo is not None
    assert user.created_at <= datetime.now(UTC)


def test_native_enum_rejects_unknown_role(db_session: Session) -> None:
    user = _user()
    db_session.add(user)
    db_session.flush()

    with pytest.raises(DBAPIError):
        db_session.execute(
            text("UPDATE users SET role = 'NOT_A_ROLE' WHERE id = :id"),
            {"id": user.id},
        )
        db_session.flush()


def test_uncommitted_work_is_discarded_on_rollback(db_session: Session) -> None:
    email = f"rollback-{uuid.uuid4().hex[:8]}@example.com"
    db_session.add(_user(email=email))
    db_session.flush()
    db_session.rollback()

    assert db_session.scalar(select(User).where(User.email == email)) is None


def test_get_db_rolls_back_uncommitted_work_on_exception(db_session: Session) -> None:
    email = f"session-safety-{uuid.uuid4().hex[:8]}@example.com"
    gen = get_db()
    session = next(gen)
    session.add(_user(email=email))
    session.flush()

    with pytest.raises(RuntimeError):
        gen.throw(RuntimeError("boom"))

    with SessionLocal() as verify:
        assert verify.scalar(select(User).where(User.email == email)) is None


def test_failed_requests_do_not_leak_checked_out_connections(
    client: TestClient,
) -> None:
    before = engine.pool.checkedout()

    assert client.get("/api/v1/auth/me").status_code == 401
    assert (
        client.post(
            "/api/v1/auth/login",
            json={"email": "not-an-email", "password": "x"},
        ).status_code
        == 422
    )

    assert engine.pool.checkedout() == before


def test_deleting_user_cascades_auth_tokens(db_session: Session) -> None:
    user = _user()
    db_session.add(user)
    db_session.flush()
    now = datetime.now(UTC)
    db_session.add_all(
        [
            RefreshToken(
                user_id=user.id,
                token_hash="a" * 64,
                expires_at=now,
            ),
            EmailVerificationToken(
                user_id=user.id,
                token_hash="b" * 64,
                expires_at=now,
            ),
            PasswordResetToken(
                user_id=user.id,
                token_hash="c" * 64,
                expires_at=now,
            ),
        ]
    )
    db_session.flush()
    user_id = user.id
    db_session.execute(delete(User).where(User.id == user_id))
    db_session.flush()

    assert db_session.scalar(select(User).where(User.id == user_id)) is None
    assert (
        db_session.scalar(select(RefreshToken).where(RefreshToken.user_id == user_id))
        is None
    )
    assert (
        db_session.scalar(
            select(EmailVerificationToken).where(
                EmailVerificationToken.user_id == user_id
            )
        )
        is None
    )
    assert (
        db_session.scalar(
            select(PasswordResetToken).where(PasswordResetToken.user_id == user_id)
        )
        is None
    )


def test_deleting_organization_does_not_orphan_or_silently_drop_workspaces(
    db_session: Session,
) -> None:
    organization = Organization(name="Org", slug=f"org-{uuid.uuid4().hex[:8]}")
    db_session.add(organization)
    db_session.flush()
    db_session.add(
        Workspace(
            organization_id=organization.id,
            name="Analytics",
            slug="analytics",
        )
    )
    db_session.flush()

    db_session.delete(organization)
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_workspace_member_unique_constraint(db_session: Session) -> None:
    user = _user()
    organization = Organization(name="Org", slug=f"org-{uuid.uuid4().hex[:8]}")
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
