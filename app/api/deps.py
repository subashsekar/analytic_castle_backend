import uuid
from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, HTTPException, Path, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.authorization import (
    WORKSPACE_ROLE_RANK,
    DataSourceAccess,
    OrganizationAccess,
    WorkspaceAccess,
    is_super_admin,
    user_has_role,
    workspace_has_role,
    workspace_role_has_permission,
)
from app.core.config import settings
from app.core.rate_limit import check_endpoint_rate_limit
from app.core.security import TOKEN_TYPE_ACCESS, InvalidTokenError, decode_token
from app.db.models import DataSource, Organization, User, Workspace, WorkspaceMember
from app.db.session import get_db
from app.enums import UserRole, WorkspacePermission, WorkspaceRole

_bearer = HTTPBearer(auto_error=False)

NOT_AUTHORIZED_DETAIL = "Not authorized"
TOO_MANY_REQUESTS_DETAIL = "Too many requests"
ORGANIZATION_NOT_FOUND_DETAIL = "Organization not found"
WORKSPACE_NOT_FOUND_DETAIL = "Workspace not found"
DATA_SOURCE_NOT_FOUND_DETAIL = "Data source not found"


def _forbidden() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=NOT_AUTHORIZED_DETAIL,
    )


def _workspace_not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=WORKSPACE_NOT_FOUND_DETAIL,
    )


def _data_source_not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=DATA_SOURCE_NOT_FOUND_DETAIL,
    )


def _organization_not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=ORGANIZATION_NOT_FOUND_DETAIL,
    )


def get_client_ip(request: Request) -> str:
    if settings.TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            candidate = forwarded.split(",")[0].strip()
            if candidate:
                return candidate
    if request.client is not None and request.client.host:
        return request.client.host
    return "unknown"


def require_rate_limit(scope: str) -> Callable[..., None]:
    def _require_rate_limit(request: Request) -> None:
        if not settings.RATE_LIMIT_ENABLED:
            return
        result = check_endpoint_rate_limit(scope, get_client_ip(request))
        if not result.allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=TOO_MANY_REQUESTS_DETAIL,
                headers={"Retry-After": str(result.retry_after)},
            )

    return _require_rate_limit


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_token(credentials.credentials, expected_type=TOKEN_TYPE_ACCESS)
        user_id = uuid.UUID(str(payload["sub"]))
    except (InvalidTokenError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    # Authoritative role and status come from the database, never from the JWT.
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def require_roles(*roles: UserRole) -> Callable[..., User]:
    if not roles:
        raise ValueError("require_roles requires at least one role")
    required = roles

    def _require_roles(
        current_user: Annotated[User, Depends(get_current_user)],
    ) -> User:
        if not user_has_role(current_user.role, *required):
            raise _forbidden()
        return current_user

    return _require_roles


def _load_workspace(db: Session, workspace_id: uuid.UUID) -> Workspace:
    workspace = db.get(Workspace, workspace_id)
    if workspace is None:
        raise _workspace_not_found()
    return workspace


def _load_membership(
    db: Session,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
) -> WorkspaceMember | None:
    return db.scalar(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == user_id,
        )
    )


def get_workspace_member(
    workspace_id: Annotated[uuid.UUID, Path()],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> WorkspaceMember:
    _load_workspace(db, workspace_id)
    member = _load_membership(db, workspace_id, current_user.id)
    if member is None:
        raise _forbidden()
    return member


def load_workspace_access(
    db: Session,
    user: User,
    workspace_id: uuid.UUID,
    *,
    required_roles: tuple[WorkspaceRole, ...] | None = None,
    permission: WorkspacePermission | None = None,
    allow_super_admin: bool = False,
) -> WorkspaceAccess:
    workspace = _load_workspace(db, workspace_id)
    member = _load_membership(db, workspace_id, user.id)
    if allow_super_admin and is_super_admin(user.role):
        return WorkspaceAccess(
            user=user,
            workspace=workspace,
            member=member,
            via_super_admin=True,
        )
    if member is None:
        raise _forbidden()
    if required_roles and not workspace_has_role(member.role, *required_roles):
        raise _forbidden()
    if permission is not None and not workspace_role_has_permission(
        member.role, permission
    ):
        raise _forbidden()
    return WorkspaceAccess(user=user, workspace=workspace, member=member)


def require_workspace_role(
    *roles: WorkspaceRole,
    allow_super_admin: bool = False,
) -> Callable[..., WorkspaceAccess]:
    if not roles:
        raise ValueError("require_workspace_role requires at least one role")
    required = roles

    def _require_workspace_role(
        workspace_id: Annotated[uuid.UUID, Path()],
        current_user: Annotated[User, Depends(get_current_user)],
        db: Annotated[Session, Depends(get_db)],
    ) -> WorkspaceAccess:
        return load_workspace_access(
            db,
            current_user,
            workspace_id,
            required_roles=required,
            allow_super_admin=allow_super_admin,
        )

    return _require_workspace_role


def require_workspace_permission(
    permission: WorkspacePermission,
    *,
    allow_super_admin: bool = False,
) -> Callable[..., WorkspaceAccess]:
    def _require_workspace_permission(
        workspace_id: Annotated[uuid.UUID, Path()],
        current_user: Annotated[User, Depends(get_current_user)],
        db: Annotated[Session, Depends(get_db)],
    ) -> WorkspaceAccess:
        return load_workspace_access(
            db,
            current_user,
            workspace_id,
            permission=permission,
            allow_super_admin=allow_super_admin,
        )

    return _require_workspace_permission


def require_data_source_permission(
    permission: WorkspacePermission,
    *,
    allow_super_admin: bool = False,
) -> Callable[..., DataSourceAccess]:
    def _require_data_source_permission(
        data_source_id: Annotated[uuid.UUID, Path()],
        current_user: Annotated[User, Depends(get_current_user)],
        db: Annotated[Session, Depends(get_db)],
    ) -> DataSourceAccess:
        data_source = db.get(DataSource, data_source_id)
        if data_source is None:
            raise _data_source_not_found()
        access = load_workspace_access(
            db,
            current_user,
            data_source.workspace_id,
            permission=permission,
            allow_super_admin=allow_super_admin,
        )
        return DataSourceAccess(
            user=access.user,
            workspace=access.workspace,
            member=access.member,
            data_source=data_source,
            via_super_admin=access.via_super_admin,
        )

    return _require_data_source_permission


def _highest_organization_membership(
    db: Session,
    organization_id: uuid.UUID,
    user_id: uuid.UUID,
) -> WorkspaceMember | None:
    members = db.scalars(
        select(WorkspaceMember)
        .join(Workspace)
        .where(
            Workspace.organization_id == organization_id,
            WorkspaceMember.user_id == user_id,
        )
    ).all()
    if not members:
        return None
    return max(members, key=lambda member: WORKSPACE_ROLE_RANK[member.role])


def load_organization_access(
    db: Session,
    user: User,
    organization_id: uuid.UUID,
    *,
    required_roles: tuple[WorkspaceRole, ...] | None = None,
    allow_super_admin: bool = False,
) -> OrganizationAccess:
    organization = db.get(Organization, organization_id)
    if organization is None:
        raise _organization_not_found()

    member = _highest_organization_membership(db, organization_id, user.id)
    if allow_super_admin and is_super_admin(user.role):
        return OrganizationAccess(
            user=user,
            organization=organization,
            member=member,
            via_super_admin=True,
        )
    if member is None:
        raise _organization_not_found()
    if required_roles and not workspace_has_role(member.role, *required_roles):
        raise _forbidden()
    return OrganizationAccess(user=user, organization=organization, member=member)


def get_organization_access(
    organization_id: Annotated[uuid.UUID, Path()],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> OrganizationAccess:
    return load_organization_access(
        db,
        current_user,
        organization_id,
        allow_super_admin=True,
    )


def _founding_workspace(db: Session, organization_id: uuid.UUID) -> Workspace | None:
    return db.scalar(
        select(Workspace)
        .where(Workspace.organization_id == organization_id)
        .order_by(Workspace.created_at.asc(), Workspace.id.asc())
        .limit(1)
    )


def user_is_organization_owner(
    db: Session,
    organization_id: uuid.UUID,
    user_id: uuid.UUID,
) -> bool:
    founding = _founding_workspace(db, organization_id)
    if founding is None:
        return False
    member = _load_membership(db, founding.id, user_id)
    return member is not None and member.role == WorkspaceRole.OWNER


def require_organization_owner(
    *,
    allow_super_admin: bool = False,
) -> Callable[..., OrganizationAccess]:
    def _require_organization_owner(
        organization_id: Annotated[uuid.UUID, Path()],
        current_user: Annotated[User, Depends(get_current_user)],
        db: Annotated[Session, Depends(get_db)],
    ) -> OrganizationAccess:
        access = load_organization_access(
            db,
            current_user,
            organization_id,
            allow_super_admin=allow_super_admin,
        )
        if access.via_super_admin:
            return access
        if not user_is_organization_owner(db, organization_id, current_user.id):
            raise _forbidden()
        return access

    return _require_organization_owner


def require_organization_role(
    *roles: WorkspaceRole,
    allow_super_admin: bool = False,
) -> Callable[..., OrganizationAccess]:
    if not roles:
        raise ValueError("require_organization_role requires at least one role")
    required = roles

    def _require_organization_role(
        organization_id: Annotated[uuid.UUID, Path()],
        current_user: Annotated[User, Depends(get_current_user)],
        db: Annotated[Session, Depends(get_db)],
    ) -> OrganizationAccess:
        return load_organization_access(
            db,
            current_user,
            organization_id,
            required_roles=required,
            allow_super_admin=allow_super_admin,
        )

    return _require_organization_role
