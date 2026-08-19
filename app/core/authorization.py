from dataclasses import dataclass

from app.db.models import DataSource, Organization, User, Workspace, WorkspaceMember
from app.enums import UserRole, WorkspacePermission, WorkspaceRole

USER_ROLE_RANK: dict[UserRole, int] = {
    UserRole.USER: 1,
    UserRole.ADMIN: 2,
    UserRole.SUPER_ADMIN: 3,
}

WORKSPACE_ROLE_RANK: dict[WorkspaceRole, int] = {
    WorkspaceRole.MEMBER: 1,
    WorkspaceRole.ADMIN: 2,
    WorkspaceRole.OWNER: 3,
}

_MEMBER_PERMISSIONS = frozenset(
    {
        WorkspacePermission.WORKSPACE_READ,
        WorkspacePermission.MEMBER_READ,
        WorkspacePermission.DATA_SOURCE_READ,
        WorkspacePermission.DATA_SOURCE_TEST,
    }
)
_ADMIN_PERMISSIONS = _MEMBER_PERMISSIONS | frozenset(
    {
        WorkspacePermission.WORKSPACE_UPDATE,
        WorkspacePermission.MEMBER_ADD,
        WorkspacePermission.MEMBER_UPDATE,
        WorkspacePermission.MEMBER_REMOVE,
        WorkspacePermission.DATA_SOURCE_CREATE,
        WorkspacePermission.DATA_SOURCE_UPDATE,
        WorkspacePermission.DATA_SOURCE_DELETE,
    }
)
_OWNER_PERMISSIONS = _ADMIN_PERMISSIONS | frozenset(
    {
        WorkspacePermission.WORKSPACE_DELETE,
    }
)

WORKSPACE_ROLE_PERMISSIONS: dict[WorkspaceRole, frozenset[WorkspacePermission]] = {
    WorkspaceRole.MEMBER: _MEMBER_PERMISSIONS,
    WorkspaceRole.ADMIN: _ADMIN_PERMISSIONS,
    WorkspaceRole.OWNER: _OWNER_PERMISSIONS,
}


@dataclass(frozen=True)
class WorkspaceAccess:
    user: User
    workspace: Workspace
    member: WorkspaceMember | None
    via_super_admin: bool = False


@dataclass(frozen=True)
class OrganizationAccess:
    user: User
    organization: Organization
    member: WorkspaceMember | None
    via_super_admin: bool = False


@dataclass(frozen=True)
class DataSourceAccess:
    user: User
    workspace: Workspace
    member: WorkspaceMember | None
    data_source: DataSource
    via_super_admin: bool = False


def user_has_role(user_role: UserRole, *required: UserRole) -> bool:
    if not required:
        raise ValueError("At least one role is required")
    return USER_ROLE_RANK[user_role] >= min(USER_ROLE_RANK[role] for role in required)


def workspace_has_role(member_role: WorkspaceRole, *required: WorkspaceRole) -> bool:
    if not required:
        raise ValueError("At least one role is required")
    return WORKSPACE_ROLE_RANK[member_role] >= min(
        WORKSPACE_ROLE_RANK[role] for role in required
    )


def is_super_admin(user_role: UserRole) -> bool:
    return user_role == UserRole.SUPER_ADMIN


def workspace_role_has_permission(
    member_role: WorkspaceRole,
    permission: WorkspacePermission,
) -> bool:
    return permission in WORKSPACE_ROLE_PERMISSIONS[member_role]


def effective_workspace_role(
    access: WorkspaceAccess | OrganizationAccess,
) -> WorkspaceRole | None:
    if access.via_super_admin:
        return WorkspaceRole.OWNER
    if access.member is None:
        return None
    return access.member.role


def can_assign_workspace_role(
    actor_role: WorkspaceRole, new_role: WorkspaceRole
) -> bool:
    if new_role == WorkspaceRole.OWNER:
        return False
    if new_role == WorkspaceRole.ADMIN:
        return actor_role == WorkspaceRole.OWNER
    return workspace_has_role(actor_role, WorkspaceRole.ADMIN)


def can_change_member_role(
    actor_role: WorkspaceRole,
    target_role: WorkspaceRole,
    new_role: WorkspaceRole,
    *,
    is_self: bool,
) -> bool:
    if is_self or target_role == WorkspaceRole.OWNER:
        return False
    if target_role == WorkspaceRole.ADMIN and actor_role != WorkspaceRole.OWNER:
        return False
    return can_assign_workspace_role(actor_role, new_role)


def can_remove_member(
    actor_role: WorkspaceRole,
    target_role: WorkspaceRole,
    *,
    is_self: bool,
) -> bool:
    if target_role == WorkspaceRole.OWNER:
        return False
    if is_self:
        return True
    if target_role == WorkspaceRole.ADMIN:
        return actor_role == WorkspaceRole.OWNER
    return workspace_has_role(actor_role, WorkspaceRole.ADMIN)
