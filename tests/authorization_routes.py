from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import (
    WorkspaceAccess,
    get_workspace_member,
    require_roles,
    require_workspace_permission,
    require_workspace_role,
)
from app.db.models import User, WorkspaceMember
from app.enums import UserRole, WorkspacePermission, WorkspaceRole

router = APIRouter(prefix="/api/v1/_authz", tags=["authz-test"])


class RoleProbe(BaseModel):
    role: str | None = None


@router.get("/admin-only")
def admin_only(
    current_user: Annotated[User, Depends(require_roles(UserRole.ADMIN))],
) -> dict[str, str]:
    return {"id": str(current_user.id), "role": current_user.role.value}


@router.post("/admin-only")
def admin_only_post(
    payload: RoleProbe,
    current_user: Annotated[User, Depends(require_roles(UserRole.ADMIN))],
) -> dict[str, str | None]:
    return {
        "id": str(current_user.id),
        "role": current_user.role.value,
        "body_role": payload.role,
    }


@router.get("/super-admin-only")
def super_admin_only(
    current_user: Annotated[User, Depends(require_roles(UserRole.SUPER_ADMIN))],
) -> dict[str, str]:
    return {"id": str(current_user.id), "role": current_user.role.value}


@router.get("/authenticated")
def any_authenticated_user(
    current_user: Annotated[User, Depends(require_roles(UserRole.USER))],
) -> dict[str, str]:
    return {"id": str(current_user.id), "role": current_user.role.value}


@router.get("/workspaces/{workspace_id}")
def workspace_member_only(
    member: Annotated[WorkspaceMember, Depends(get_workspace_member)],
) -> dict[str, str]:
    return {
        "workspace_id": str(member.workspace_id),
        "user_id": str(member.user_id),
        "role": member.role.value,
    }


@router.get("/workspaces/{workspace_id}/admin")
def workspace_admin(
    access: Annotated[
        WorkspaceAccess,
        Depends(require_workspace_role(WorkspaceRole.ADMIN)),
    ],
) -> dict[str, str | None]:
    return {
        "workspace_id": str(access.workspace.id),
        "role": access.member.role.value if access.member else None,
    }


@router.get("/workspaces/{workspace_id}/owner")
def workspace_owner(
    access: Annotated[
        WorkspaceAccess,
        Depends(require_workspace_role(WorkspaceRole.OWNER)),
    ],
) -> dict[str, str]:
    return {"workspace_id": str(access.workspace.id)}


@router.get("/workspaces/{workspace_id}/admin-with-super")
def workspace_admin_with_super(
    access: Annotated[
        WorkspaceAccess,
        Depends(require_workspace_role(WorkspaceRole.ADMIN, allow_super_admin=True)),
    ],
) -> dict[str, str | bool | None]:
    return {
        "workspace_id": str(access.workspace.id),
        "via_super_admin": access.via_super_admin,
        "role": access.member.role.value if access.member else None,
    }


@router.get("/workspaces/{workspace_id}/permissions/delete")
def workspace_delete_permission(
    access: Annotated[
        WorkspaceAccess,
        Depends(require_workspace_permission(WorkspacePermission.WORKSPACE_DELETE)),
    ],
) -> dict[str, str]:
    return {"workspace_id": str(access.workspace.id)}


@router.get("/workspaces/{workspace_id}/permissions/members")
def workspace_manage_members(
    access: Annotated[
        WorkspaceAccess,
        Depends(require_workspace_permission(WorkspacePermission.MEMBER_ADD)),
    ],
) -> dict[str, str]:
    return {"workspace_id": str(access.workspace.id)}


@router.get("/workspaces/{workspace_id}/probe/{other_id}")
def workspace_id_must_come_from_path(
    other_id: UUID,
    member: Annotated[WorkspaceMember, Depends(get_workspace_member)],
) -> dict[str, str]:
    return {
        "authorized_workspace_id": str(member.workspace_id),
        "other_id": str(other_id),
    }
