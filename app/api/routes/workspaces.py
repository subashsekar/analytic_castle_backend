from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from app.api.deps import (
    get_current_user,
    load_organization_access,
    require_workspace_permission,
)
from app.core.authorization import (
    WorkspaceAccess,
    can_assign_workspace_role,
    can_change_member_role,
    can_remove_member,
    effective_workspace_role,
    is_super_admin,
)
from app.core.slugs import slugify, unique_slug
from app.db.models import User, Workspace, WorkspaceMember, WorkspaceRole
from app.db.session import get_db
from app.enums import WorkspacePermission
from app.schemas.auth import MessageResponse
from app.schemas.workspace import WorkspaceCreate, WorkspaceRead, WorkspaceUpdate
from app.schemas.workspace_member import (
    WorkspaceMemberCreate,
    WorkspaceMemberRead,
    WorkspaceMemberUpdate,
)

router = APIRouter(prefix="/api/v1/workspaces", tags=["workspaces"])

INVALID_SLUG_DETAIL = "Unable to generate a valid slug from the provided name"
DUPLICATE_WORKSPACE_SLUG_DETAIL = "A workspace with this slug already exists"
DUPLICATE_MEMBER_DETAIL = "User is already a member of this workspace"
OWNER_ASSIGNMENT_DETAIL = "Ownership cannot be assigned through member management"
OWNER_LEAVE_DETAIL = "Owner cannot leave the workspace"
LAST_WORKSPACE_DETAIL = "Cannot delete the last workspace in an organization"
SELF_ROLE_DETAIL = "You cannot change your own role"
USER_NOT_FOUND_DETAIL = "User not found"
MEMBER_NOT_FOUND_DETAIL = "Member not found"


def _invalid_slug() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail=INVALID_SLUG_DETAIL,
    )


def _forbidden() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Not authorized",
    )


def _slug_from_name(name: str) -> str:
    slug = slugify(name)
    if not slug:
        raise _invalid_slug()
    return slug


def _workspace_slug(
    db: Session,
    organization_id: UUID,
    name: str,
    *,
    exclude_id: UUID | None = None,
) -> str:
    base = _slug_from_name(name)
    filters = [Workspace.organization_id == organization_id]
    if exclude_id is not None:
        filters.append(Workspace.id != exclude_id)
    return unique_slug(db, Workspace.slug, base, *filters)


def _require_actor_role(access: WorkspaceAccess) -> WorkspaceRole:
    role = effective_workspace_role(access)
    if role is None:
        raise _forbidden()
    return role


def _load_member(
    db: Session,
    workspace_id: UUID,
    user_id: UUID,
) -> WorkspaceMember:
    member = db.scalar(
        select(WorkspaceMember)
        .options(selectinload(WorkspaceMember.user))
        .where(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == user_id,
        )
    )
    if member is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=MEMBER_NOT_FOUND_DETAIL,
        )
    return member


@router.post("", response_model=WorkspaceRead, status_code=status.HTTP_201_CREATED)
def create_workspace(
    payload: WorkspaceCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> Workspace:
    load_organization_access(
        db,
        current_user,
        payload.organization_id,
        required_roles=(WorkspaceRole.ADMIN,),
        allow_super_admin=True,
    )
    workspace = Workspace(
        organization_id=payload.organization_id,
        name=payload.name,
        slug=_workspace_slug(db, payload.organization_id, payload.name),
    )
    db.add(workspace)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=DUPLICATE_WORKSPACE_SLUG_DETAIL,
        ) from None

    db.add(
        WorkspaceMember(
            workspace_id=workspace.id,
            user_id=current_user.id,
            role=WorkspaceRole.OWNER,
        )
    )
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=DUPLICATE_WORKSPACE_SLUG_DETAIL,
        ) from None
    db.refresh(workspace)
    return workspace


@router.get("", response_model=list[WorkspaceRead])
def list_workspaces(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    organization_id: UUID | None = None,
) -> list[Workspace]:
    stmt = select(Workspace).order_by(Workspace.created_at.desc())
    if organization_id is not None:
        stmt = stmt.where(Workspace.organization_id == organization_id)
    if not is_super_admin(current_user.role):
        stmt = stmt.join(WorkspaceMember).where(
            WorkspaceMember.user_id == current_user.id
        )
    return list(db.scalars(stmt).unique().all())


@router.get("/{workspace_id}", response_model=WorkspaceRead)
def get_workspace(
    access: Annotated[
        WorkspaceAccess,
        Depends(
            require_workspace_permission(
                WorkspacePermission.WORKSPACE_READ,
                allow_super_admin=True,
            )
        ),
    ],
) -> Workspace:
    return access.workspace


@router.patch("/{workspace_id}", response_model=WorkspaceRead)
def update_workspace(
    payload: WorkspaceUpdate,
    access: Annotated[
        WorkspaceAccess,
        Depends(
            require_workspace_permission(
                WorkspacePermission.WORKSPACE_UPDATE,
                allow_super_admin=True,
            )
        ),
    ],
    db: Annotated[Session, Depends(get_db)],
) -> Workspace:
    workspace = access.workspace
    if payload.name is not None and payload.name != workspace.name:
        workspace.slug = _workspace_slug(
            db,
            workspace.organization_id,
            payload.name,
            exclude_id=workspace.id,
        )
        workspace.name = payload.name
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=DUPLICATE_WORKSPACE_SLUG_DETAIL,
        ) from None
    db.refresh(workspace)
    return workspace


@router.delete("/{workspace_id}", response_model=MessageResponse)
def delete_workspace(
    access: Annotated[
        WorkspaceAccess,
        Depends(
            require_workspace_permission(
                WorkspacePermission.WORKSPACE_DELETE,
                allow_super_admin=True,
            )
        ),
    ],
    db: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    workspace = access.workspace
    remaining = db.scalar(
        select(func.count())
        .select_from(Workspace)
        .where(
            Workspace.organization_id == workspace.organization_id,
            Workspace.id != workspace.id,
        )
    )
    if not remaining:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=LAST_WORKSPACE_DETAIL,
        )
    db.execute(
        delete(WorkspaceMember).where(WorkspaceMember.workspace_id == workspace.id)
    )
    db.delete(workspace)
    db.commit()
    return MessageResponse(detail="Workspace deleted")


@router.get("/{workspace_id}/members", response_model=list[WorkspaceMemberRead])
def list_members(
    access: Annotated[
        WorkspaceAccess,
        Depends(
            require_workspace_permission(
                WorkspacePermission.MEMBER_READ,
                allow_super_admin=True,
            )
        ),
    ],
    db: Annotated[Session, Depends(get_db)],
) -> list[WorkspaceMember]:
    return list(
        db.scalars(
            select(WorkspaceMember)
            .options(selectinload(WorkspaceMember.user))
            .where(WorkspaceMember.workspace_id == access.workspace.id)
            .order_by(WorkspaceMember.created_at.asc())
        ).all()
    )


@router.post(
    "/{workspace_id}/members",
    response_model=WorkspaceMemberRead,
    status_code=status.HTTP_201_CREATED,
)
def add_member(
    payload: WorkspaceMemberCreate,
    access: Annotated[
        WorkspaceAccess,
        Depends(
            require_workspace_permission(
                WorkspacePermission.MEMBER_ADD,
                allow_super_admin=True,
            )
        ),
    ],
    db: Annotated[Session, Depends(get_db)],
) -> WorkspaceMember:
    actor_role = _require_actor_role(access)
    if payload.role == WorkspaceRole.OWNER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=OWNER_ASSIGNMENT_DETAIL,
        )
    if not can_assign_workspace_role(actor_role, payload.role):
        raise _forbidden()

    user = db.get(User, payload.user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=USER_NOT_FOUND_DETAIL,
        )

    member = WorkspaceMember(
        workspace_id=access.workspace.id,
        user_id=user.id,
        role=payload.role,
    )
    db.add(member)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=DUPLICATE_MEMBER_DETAIL,
        ) from None
    db.commit()
    return _load_member(db, access.workspace.id, user.id)


@router.patch(
    "/{workspace_id}/members/{user_id}",
    response_model=WorkspaceMemberRead,
)
def update_member_role(
    user_id: UUID,
    payload: WorkspaceMemberUpdate,
    access: Annotated[
        WorkspaceAccess,
        Depends(
            require_workspace_permission(
                WorkspacePermission.MEMBER_UPDATE,
                allow_super_admin=True,
            )
        ),
    ],
    db: Annotated[Session, Depends(get_db)],
) -> WorkspaceMember:
    actor_role = _require_actor_role(access)
    member = _load_member(db, access.workspace.id, user_id)
    is_self = member.user_id == access.user.id

    if payload.role == WorkspaceRole.OWNER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=OWNER_ASSIGNMENT_DETAIL,
        )
    if is_self:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=SELF_ROLE_DETAIL,
        )
    if not can_change_member_role(
        actor_role,
        member.role,
        payload.role,
        is_self=is_self,
    ):
        raise _forbidden()

    member.role = payload.role
    db.commit()
    return _load_member(db, access.workspace.id, user_id)


@router.delete(
    "/{workspace_id}/members/{user_id}",
    response_model=MessageResponse,
)
def remove_member(
    user_id: UUID,
    access: Annotated[
        WorkspaceAccess,
        Depends(
            require_workspace_permission(
                WorkspacePermission.MEMBER_READ,
                allow_super_admin=True,
            )
        ),
    ],
    db: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    actor_role = _require_actor_role(access)
    member = _load_member(db, access.workspace.id, user_id)
    is_self = member.user_id == access.user.id

    if is_self and member.role == WorkspaceRole.OWNER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=OWNER_LEAVE_DETAIL,
        )
    if not can_remove_member(actor_role, member.role, is_self=is_self):
        raise _forbidden()

    db.delete(member)
    db.commit()
    return MessageResponse(detail="Member removed")
