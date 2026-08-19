from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import (
    get_current_user,
    get_organization_access,
    require_organization_owner,
)
from app.core.authorization import OrganizationAccess, is_super_admin
from app.core.slugs import slugify, unique_slug
from app.db.models import Organization, User, Workspace, WorkspaceMember, WorkspaceRole
from app.db.session import get_db
from app.schemas.auth import MessageResponse
from app.schemas.organization import (
    OrganizationCreate,
    OrganizationRead,
    OrganizationUpdate,
)

router = APIRouter(prefix="/api/v1/organizations", tags=["organizations"])

INVALID_SLUG_DETAIL = "Unable to generate a valid slug from the provided name"
DUPLICATE_ORGANIZATION_SLUG_DETAIL = "An organization with this slug already exists"


def _invalid_slug() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail=INVALID_SLUG_DETAIL,
    )


def _slug_from_name(name: str) -> str:
    slug = slugify(name)
    if not slug:
        raise _invalid_slug()
    return slug


def _organization_slug(
    db: Session, name: str, *, exclude_id: UUID | None = None
) -> str:
    base = _slug_from_name(name)
    filters = []
    if exclude_id is not None:
        filters.append(Organization.id != exclude_id)
    return unique_slug(db, Organization.slug, base, *filters)


@router.post("", response_model=OrganizationRead, status_code=status.HTTP_201_CREATED)
def create_organization(
    payload: OrganizationCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> Organization:
    organization = Organization(
        name=payload.name,
        slug=_organization_slug(db, payload.name),
    )
    db.add(organization)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=DUPLICATE_ORGANIZATION_SLUG_DETAIL,
        ) from None

    workspace = Workspace(
        organization_id=organization.id,
        name=payload.name,
        slug=unique_slug(
            db,
            Workspace.slug,
            organization.slug,
            Workspace.organization_id == organization.id,
        ),
    )
    db.add(workspace)
    db.flush()
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
            detail=DUPLICATE_ORGANIZATION_SLUG_DETAIL,
        ) from None
    db.refresh(organization)
    return organization


@router.get("", response_model=list[OrganizationRead])
def list_organizations(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> list[Organization]:
    stmt = select(Organization).order_by(Organization.created_at.desc())
    if not is_super_admin(current_user.role):
        stmt = (
            stmt.join(Workspace, Workspace.organization_id == Organization.id)
            .join(WorkspaceMember, WorkspaceMember.workspace_id == Workspace.id)
            .where(WorkspaceMember.user_id == current_user.id)
        )
    return list(db.scalars(stmt).unique().all())


@router.get("/{organization_id}", response_model=OrganizationRead)
def get_organization(
    access: Annotated[OrganizationAccess, Depends(get_organization_access)],
) -> Organization:
    return access.organization


@router.patch("/{organization_id}", response_model=OrganizationRead)
def update_organization(
    payload: OrganizationUpdate,
    access: Annotated[
        OrganizationAccess,
        Depends(require_organization_owner(allow_super_admin=True)),
    ],
    db: Annotated[Session, Depends(get_db)],
) -> Organization:
    organization = access.organization
    if payload.name is not None and payload.name != organization.name:
        organization.slug = _organization_slug(
            db, payload.name, exclude_id=organization.id
        )
        organization.name = payload.name
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=DUPLICATE_ORGANIZATION_SLUG_DETAIL,
        ) from None
    db.refresh(organization)
    return organization


@router.delete("/{organization_id}", response_model=MessageResponse)
def delete_organization(
    access: Annotated[
        OrganizationAccess,
        Depends(require_organization_owner(allow_super_admin=True)),
    ],
    db: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    organization = access.organization
    workspace_ids = select(Workspace.id).where(
        Workspace.organization_id == organization.id
    )
    db.execute(
        delete(WorkspaceMember).where(WorkspaceMember.workspace_id.in_(workspace_ids))
    )
    db.execute(delete(Workspace).where(Workspace.organization_id == organization.id))
    db.delete(organization)
    db.commit()
    return MessageResponse(detail="Organization deleted")
