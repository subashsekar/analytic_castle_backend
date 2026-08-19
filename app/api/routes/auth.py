import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import get_client_ip, get_current_user, require_rate_limit
from app.core.config import settings
from app.core.rate_limit import login_protection
from app.core.security import (
    DUMMY_PASSWORD_HASH,
    TOKEN_TYPE_REFRESH,
    InvalidTokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_secure_token,
    hash_password,
    hash_refresh_token,
    hash_token,
    tokens_match,
    verify_password,
)
from app.db.models import (
    EmailVerificationToken,
    PasswordResetToken,
    RefreshToken,
    User,
    UserRole,
)
from app.db.session import get_db
from app.schemas.auth import (
    ChangePasswordRequest,
    ForgotPasswordRequest,
    LoginRequest,
    MessageResponse,
    RefreshTokenRequest,
    RegisterRequest,
    ResendVerificationRequest,
    ResetPasswordRequest,
    TokenResponse,
    VerifyEmailRequest,
)
from app.schemas.user import UserRead
from app.services.email_service import email_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

FORGOT_PASSWORD_DETAIL = "If the account exists, a password reset link has been sent."
RESEND_VERIFICATION_DETAIL = (
    "If the account exists and is unverified, a verification email has been sent."
)
INVALID_VERIFICATION_TOKEN_DETAIL = "Invalid or expired verification token"
INVALID_RESET_TOKEN_DETAIL = "Invalid or expired reset token"


def _invalid_credentials() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid email or password",
    )


def _invalid_refresh_token() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid refresh token",
    )


def _invalid_verification_token() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=INVALID_VERIFICATION_TOKEN_DETAIL,
    )


def _invalid_reset_token() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=INVALID_RESET_TOKEN_DETAIL,
    )


def _too_many_requests(retry_after: int) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="Too many requests",
        headers={"Retry-After": str(retry_after)},
    )


def _issue_tokens(db: Session, user: User) -> TokenResponse:
    access_token = create_access_token(user.id)
    refresh_token, expires_at = create_refresh_token(user.id)
    db.add(
        RefreshToken(
            user_id=user.id,
            token_hash=hash_refresh_token(refresh_token),
            expires_at=expires_at,
        )
    )
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserRead.model_validate(user),
    )


def _load_refresh_session(db: Session, raw_token: str) -> tuple[RefreshToken, User]:
    try:
        payload = decode_token(raw_token, expected_type=TOKEN_TYPE_REFRESH)
        user_id = uuid.UUID(str(payload["sub"]))
    except (InvalidTokenError, ValueError):
        logger.info("Refresh token rejected")
        raise _invalid_refresh_token() from None

    stored = db.scalar(
        select(RefreshToken)
        .where(RefreshToken.token_hash == hash_refresh_token(raw_token))
        .with_for_update()
    )
    now = datetime.now(UTC)
    if stored is None or stored.revoked_at is not None or stored.expires_at <= now:
        logger.info("Refresh token rejected")
        raise _invalid_refresh_token()

    user = db.get(User, stored.user_id)
    if user is None or not user.is_active or user.id != user_id:
        logger.info("Refresh token rejected")
        raise _invalid_refresh_token()

    return stored, user


def _revoke_user_refresh_tokens(db: Session, user_id: uuid.UUID, now: datetime) -> None:
    tokens = db.scalars(
        select(RefreshToken).where(
            RefreshToken.user_id == user_id,
            RefreshToken.revoked_at.is_(None),
        )
    ).all()
    for token in tokens:
        token.revoked_at = now


def _invalidate_unused_tokens(
    tokens: Sequence[EmailVerificationToken] | Sequence[PasswordResetToken],
    now: datetime,
) -> None:
    for token in tokens:
        token.used_at = now


def _issue_email_verification_token(db: Session, user: User, now: datetime) -> str:
    existing = db.scalars(
        select(EmailVerificationToken).where(
            EmailVerificationToken.user_id == user.id,
            EmailVerificationToken.used_at.is_(None),
        )
    ).all()
    _invalidate_unused_tokens(existing, now)
    raw_token = generate_secure_token()
    db.add(
        EmailVerificationToken(
            user_id=user.id,
            token_hash=hash_token(raw_token),
            expires_at=now
            + timedelta(minutes=settings.EMAIL_VERIFICATION_EXPIRE_MINUTES),
        )
    )
    return raw_token


def _issue_password_reset_token(db: Session, user: User, now: datetime) -> str:
    existing = db.scalars(
        select(PasswordResetToken).where(
            PasswordResetToken.user_id == user.id,
            PasswordResetToken.used_at.is_(None),
        )
    ).all()
    _invalidate_unused_tokens(existing, now)
    raw_token = generate_secure_token()
    db.add(
        PasswordResetToken(
            user_id=user.id,
            token_hash=hash_token(raw_token),
            expires_at=now + timedelta(minutes=settings.PASSWORD_RESET_EXPIRE_MINUTES),
        )
    )
    return raw_token


def _send_verification_email(to: str, token: str) -> None:
    try:
        email_service.send_verification_email(to, token)
    except Exception:
        logger.exception("Failed to send verification email")


def _send_password_reset_email(to: str, token: str) -> None:
    try:
        email_service.send_password_reset_email(to, token)
    except Exception:
        logger.exception("Failed to send password reset email")


@router.post(
    "/register",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_rate_limit("register"))],
)
def register(
    payload: RegisterRequest,
    db: Annotated[Session, Depends(get_db)],
) -> User:
    existing = db.scalar(select(User).where(User.email == payload.email))
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )

    user = User(
        first_name=payload.first_name,
        last_name=payload.last_name,
        email=payload.email,
        password_hash=hash_password(payload.password),
        role=UserRole.USER,
        is_verified=False,
        is_active=True,
    )
    db.add(user)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        ) from None

    raw_token = _issue_email_verification_token(db, user, datetime.now(UTC))
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        ) from None
    db.refresh(user)
    _send_verification_email(user.email, raw_token)
    logger.info("User registered")
    return user


@router.post(
    "/login",
    response_model=TokenResponse,
    dependencies=[Depends(require_rate_limit("login"))],
)
def login(
    payload: LoginRequest,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> TokenResponse:
    client_ip = get_client_ip(request)
    if settings.RATE_LIMIT_ENABLED and not login_protection.allow(
        client_ip, payload.email
    ):
        logger.info("Authentication failed")
        raise _too_many_requests(login_protection.retry_after(client_ip, payload.email))

    user = db.scalar(select(User).where(User.email == payload.email))
    if user is None:
        verify_password(payload.password, DUMMY_PASSWORD_HASH)
        if settings.RATE_LIMIT_ENABLED:
            login_protection.record_failure(client_ip, payload.email)
        logger.info("Authentication failed")
        raise _invalid_credentials()
    if not verify_password(payload.password, user.password_hash) or not user.is_active:
        if settings.RATE_LIMIT_ENABLED:
            login_protection.record_failure(client_ip, payload.email)
        logger.info("Authentication failed")
        raise _invalid_credentials()

    login_protection.clear(client_ip, payload.email)
    response = _issue_tokens(db, user)
    db.commit()
    logger.info("User authenticated")
    return response


@router.get("/me", response_model=UserRead)
def read_current_user(current_user: Annotated[User, Depends(get_current_user)]) -> User:
    return current_user


@router.post(
    "/refresh",
    response_model=TokenResponse,
    dependencies=[Depends(require_rate_limit("refresh"))],
)
def refresh_tokens(
    payload: RefreshTokenRequest,
    db: Annotated[Session, Depends(get_db)],
) -> TokenResponse:
    stored, user = _load_refresh_session(db, payload.refresh_token)
    stored.revoked_at = datetime.now(UTC)
    response = _issue_tokens(db, user)
    db.commit()
    logger.info("Refresh token rotated")
    return response


@router.post("/logout", response_model=MessageResponse)
def logout(
    payload: RefreshTokenRequest,
    db: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    stored, _user = _load_refresh_session(db, payload.refresh_token)
    stored.revoked_at = datetime.now(UTC)
    db.commit()
    logger.info("Refresh token revoked")
    return MessageResponse(detail="Successfully logged out")


@router.post("/verify-email", response_model=MessageResponse)
def verify_email(
    payload: VerifyEmailRequest,
    db: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    token_hash = hash_token(payload.token)
    stored = db.scalar(
        select(EmailVerificationToken)
        .where(EmailVerificationToken.token_hash == token_hash)
        .with_for_update()
    )
    now = datetime.now(UTC)
    if (
        stored is None
        or not tokens_match(stored.token_hash, payload.token)
        or stored.used_at is not None
        or stored.expires_at <= now
    ):
        raise _invalid_verification_token()

    user = db.get(User, stored.user_id, with_for_update=True)
    if user is None or not user.is_active:
        raise _invalid_verification_token()

    stored.used_at = now
    if user.is_verified:
        db.commit()
        return MessageResponse(detail="Email is already verified")

    user.is_verified = True
    db.commit()
    logger.info("Email verified")
    return MessageResponse(detail="Email verified successfully")


@router.post(
    "/resend-verification",
    response_model=MessageResponse,
    dependencies=[Depends(require_rate_limit("resend-verification"))],
)
def resend_verification(
    payload: ResendVerificationRequest,
    db: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    user = db.scalar(select(User).where(User.email == payload.email))
    if user is None or not user.is_active or user.is_verified:
        logger.info("Verification email request completed")
        return MessageResponse(detail=RESEND_VERIFICATION_DETAIL)

    now = datetime.now(UTC)
    latest = db.scalar(
        select(EmailVerificationToken)
        .where(EmailVerificationToken.user_id == user.id)
        .order_by(EmailVerificationToken.created_at.desc())
        .limit(1)
    )
    if (
        latest is not None
        and (now - latest.created_at).total_seconds()
        < settings.EMAIL_VERIFICATION_RESEND_COOLDOWN_SECONDS
    ):
        logger.info("Verification email request completed")
        return MessageResponse(detail=RESEND_VERIFICATION_DETAIL)

    raw_token = _issue_email_verification_token(db, user, now)
    db.commit()
    _send_verification_email(user.email, raw_token)
    logger.info("Verification email request completed")
    return MessageResponse(detail=RESEND_VERIFICATION_DETAIL)


@router.post(
    "/verify-email/send",
    response_model=MessageResponse,
    dependencies=[Depends(require_rate_limit("resend-verification"))],
)
def send_verification_email(
    payload: ResendVerificationRequest,
    db: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    return resend_verification(payload, db)


@router.post(
    "/forgot-password",
    response_model=MessageResponse,
    dependencies=[Depends(require_rate_limit("forgot-password"))],
)
def forgot_password(
    payload: ForgotPasswordRequest,
    db: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    user = db.scalar(select(User).where(User.email == payload.email))
    if user is not None and user.is_active:
        raw_token = _issue_password_reset_token(db, user, datetime.now(UTC))
        db.commit()
        _send_password_reset_email(user.email, raw_token)
    logger.info("Password reset requested")
    return MessageResponse(detail=FORGOT_PASSWORD_DETAIL)


@router.post(
    "/reset-password",
    response_model=MessageResponse,
    dependencies=[Depends(require_rate_limit("reset-password"))],
)
def reset_password(
    payload: ResetPasswordRequest,
    db: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    token_hash = hash_token(payload.token)
    stored = db.scalar(
        select(PasswordResetToken)
        .where(PasswordResetToken.token_hash == token_hash)
        .with_for_update()
    )
    now = datetime.now(UTC)
    if (
        stored is None
        or not tokens_match(stored.token_hash, payload.token)
        or stored.used_at is not None
        or stored.expires_at <= now
    ):
        raise _invalid_reset_token()

    user = db.get(User, stored.user_id, with_for_update=True)
    if user is None or not user.is_active:
        raise _invalid_reset_token()

    user.password_hash = hash_password(payload.new_password)
    stored.used_at = now
    _revoke_user_refresh_tokens(db, user.id, now)
    db.commit()
    logger.info("Password reset completed")
    return MessageResponse(detail="Password has been reset successfully")


@router.post(
    "/change-password",
    response_model=MessageResponse,
    dependencies=[Depends(require_rate_limit("change-password"))],
)
def change_password(
    payload: ChangePasswordRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    if not verify_password(payload.current_password, current_user.password_hash):
        logger.info("Password change rejected")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect current password",
        )

    now = datetime.now(UTC)
    current_user.password_hash = hash_password(payload.new_password)
    _revoke_user_refresh_tokens(db, current_user.id, now)
    db.commit()
    logger.info("Password changed")
    return MessageResponse(detail="Password changed successfully")
