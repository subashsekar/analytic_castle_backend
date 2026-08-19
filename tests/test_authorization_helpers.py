import pytest

from app.core.authorization import (
    USER_ROLE_RANK,
    WORKSPACE_ROLE_PERMISSIONS,
    WORKSPACE_ROLE_RANK,
    can_assign_workspace_role,
    can_change_member_role,
    can_remove_member,
    is_super_admin,
    user_has_role,
    workspace_has_role,
    workspace_role_has_permission,
)
from app.enums import UserRole, WorkspacePermission, WorkspaceRole


def test_global_role_hierarchy_order() -> None:
    assert USER_ROLE_RANK[UserRole.SUPER_ADMIN] > USER_ROLE_RANK[UserRole.ADMIN]
    assert USER_ROLE_RANK[UserRole.ADMIN] > USER_ROLE_RANK[UserRole.USER]


def test_workspace_role_hierarchy_order() -> None:
    assert (
        WORKSPACE_ROLE_RANK[WorkspaceRole.OWNER]
        > WORKSPACE_ROLE_RANK[WorkspaceRole.ADMIN]
    )
    assert (
        WORKSPACE_ROLE_RANK[WorkspaceRole.ADMIN]
        > WORKSPACE_ROLE_RANK[WorkspaceRole.MEMBER]
    )


def test_super_admin_satisfies_every_global_role() -> None:
    assert user_has_role(UserRole.SUPER_ADMIN, UserRole.USER) is True
    assert user_has_role(UserRole.SUPER_ADMIN, UserRole.ADMIN) is True
    assert user_has_role(UserRole.SUPER_ADMIN, UserRole.SUPER_ADMIN) is True


def test_admin_satisfies_admin_and_user_but_not_super_admin() -> None:
    assert user_has_role(UserRole.ADMIN, UserRole.USER) is True
    assert user_has_role(UserRole.ADMIN, UserRole.ADMIN) is True
    assert user_has_role(UserRole.ADMIN, UserRole.SUPER_ADMIN) is False


def test_user_satisfies_only_user_role() -> None:
    assert user_has_role(UserRole.USER, UserRole.USER) is True
    assert user_has_role(UserRole.USER, UserRole.ADMIN) is False
    assert user_has_role(UserRole.USER, UserRole.SUPER_ADMIN) is False


def test_user_has_role_uses_minimum_required_rank() -> None:
    assert user_has_role(UserRole.ADMIN, UserRole.ADMIN, UserRole.SUPER_ADMIN) is True
    assert user_has_role(UserRole.USER, UserRole.ADMIN, UserRole.SUPER_ADMIN) is False


def test_user_has_role_requires_at_least_one_role() -> None:
    with pytest.raises(ValueError, match="At least one role is required"):
        user_has_role(UserRole.USER)


def test_owner_satisfies_every_workspace_role() -> None:
    assert workspace_has_role(WorkspaceRole.OWNER, WorkspaceRole.MEMBER) is True
    assert workspace_has_role(WorkspaceRole.OWNER, WorkspaceRole.ADMIN) is True
    assert workspace_has_role(WorkspaceRole.OWNER, WorkspaceRole.OWNER) is True


def test_workspace_admin_cannot_perform_owner_only_operations() -> None:
    assert workspace_has_role(WorkspaceRole.ADMIN, WorkspaceRole.MEMBER) is True
    assert workspace_has_role(WorkspaceRole.ADMIN, WorkspaceRole.ADMIN) is True
    assert workspace_has_role(WorkspaceRole.ADMIN, WorkspaceRole.OWNER) is False


def test_member_is_restricted_to_member_access() -> None:
    assert workspace_has_role(WorkspaceRole.MEMBER, WorkspaceRole.MEMBER) is True
    assert workspace_has_role(WorkspaceRole.MEMBER, WorkspaceRole.ADMIN) is False
    assert workspace_has_role(WorkspaceRole.MEMBER, WorkspaceRole.OWNER) is False


def test_workspace_has_role_requires_at_least_one_role() -> None:
    with pytest.raises(ValueError, match="At least one role is required"):
        workspace_has_role(WorkspaceRole.MEMBER)


def test_is_super_admin() -> None:
    assert is_super_admin(UserRole.SUPER_ADMIN) is True
    assert is_super_admin(UserRole.ADMIN) is False
    assert is_super_admin(UserRole.USER) is False


def test_member_permissions_are_read_only() -> None:
    assert workspace_role_has_permission(
        WorkspaceRole.MEMBER, WorkspacePermission.WORKSPACE_READ
    )
    assert workspace_role_has_permission(
        WorkspaceRole.MEMBER, WorkspacePermission.MEMBER_READ
    )
    assert not workspace_role_has_permission(
        WorkspaceRole.MEMBER, WorkspacePermission.WORKSPACE_UPDATE
    )
    assert not workspace_role_has_permission(
        WorkspaceRole.MEMBER, WorkspacePermission.MEMBER_ADD
    )
    assert not workspace_role_has_permission(
        WorkspaceRole.MEMBER, WorkspacePermission.WORKSPACE_DELETE
    )
    assert workspace_role_has_permission(
        WorkspaceRole.MEMBER, WorkspacePermission.DATA_SOURCE_READ
    )
    assert workspace_role_has_permission(
        WorkspaceRole.MEMBER, WorkspacePermission.DATA_SOURCE_TEST
    )
    assert not workspace_role_has_permission(
        WorkspaceRole.MEMBER, WorkspacePermission.DATA_SOURCE_CREATE
    )
    assert not workspace_role_has_permission(
        WorkspaceRole.MEMBER, WorkspacePermission.DATA_SOURCE_UPDATE
    )
    assert not workspace_role_has_permission(
        WorkspaceRole.MEMBER, WorkspacePermission.DATA_SOURCE_DELETE
    )


def test_workspace_admin_cannot_delete_workspace() -> None:
    assert workspace_role_has_permission(
        WorkspaceRole.ADMIN, WorkspacePermission.WORKSPACE_UPDATE
    )
    assert workspace_role_has_permission(
        WorkspaceRole.ADMIN, WorkspacePermission.MEMBER_ADD
    )
    assert workspace_role_has_permission(
        WorkspaceRole.ADMIN, WorkspacePermission.MEMBER_REMOVE
    )
    assert not workspace_role_has_permission(
        WorkspaceRole.ADMIN, WorkspacePermission.WORKSPACE_DELETE
    )
    assert workspace_role_has_permission(
        WorkspaceRole.ADMIN, WorkspacePermission.DATA_SOURCE_CREATE
    )
    assert workspace_role_has_permission(
        WorkspaceRole.ADMIN, WorkspacePermission.DATA_SOURCE_DELETE
    )


def test_owner_has_all_workspace_permissions() -> None:
    for permission in WorkspacePermission:
        assert workspace_role_has_permission(WorkspaceRole.OWNER, permission)


def test_can_assign_workspace_role_blocks_owner_and_escalation() -> None:
    assert can_assign_workspace_role(WorkspaceRole.OWNER, WorkspaceRole.ADMIN) is True
    assert can_assign_workspace_role(WorkspaceRole.OWNER, WorkspaceRole.MEMBER) is True
    assert can_assign_workspace_role(WorkspaceRole.OWNER, WorkspaceRole.OWNER) is False
    assert can_assign_workspace_role(WorkspaceRole.ADMIN, WorkspaceRole.MEMBER) is True
    assert can_assign_workspace_role(WorkspaceRole.ADMIN, WorkspaceRole.ADMIN) is False
    assert (
        can_assign_workspace_role(WorkspaceRole.MEMBER, WorkspaceRole.MEMBER) is False
    )


def test_can_change_member_role_prevents_unsafe_updates() -> None:
    assert (
        can_change_member_role(
            WorkspaceRole.OWNER,
            WorkspaceRole.MEMBER,
            WorkspaceRole.ADMIN,
            is_self=False,
        )
        is True
    )
    assert (
        can_change_member_role(
            WorkspaceRole.OWNER,
            WorkspaceRole.OWNER,
            WorkspaceRole.ADMIN,
            is_self=False,
        )
        is False
    )
    assert (
        can_change_member_role(
            WorkspaceRole.ADMIN,
            WorkspaceRole.MEMBER,
            WorkspaceRole.ADMIN,
            is_self=False,
        )
        is False
    )
    assert (
        can_change_member_role(
            WorkspaceRole.OWNER,
            WorkspaceRole.MEMBER,
            WorkspaceRole.ADMIN,
            is_self=True,
        )
        is False
    )


def test_can_remove_member_prevents_owner_removal() -> None:
    assert can_remove_member(WorkspaceRole.OWNER, WorkspaceRole.ADMIN, is_self=False)
    assert can_remove_member(WorkspaceRole.ADMIN, WorkspaceRole.MEMBER, is_self=False)
    assert not can_remove_member(
        WorkspaceRole.ADMIN, WorkspaceRole.ADMIN, is_self=False
    )
    assert not can_remove_member(
        WorkspaceRole.OWNER, WorkspaceRole.OWNER, is_self=False
    )
    assert not can_remove_member(WorkspaceRole.OWNER, WorkspaceRole.OWNER, is_self=True)
    assert can_remove_member(WorkspaceRole.MEMBER, WorkspaceRole.MEMBER, is_self=True)


def test_permission_sets_do_not_overlap_unexpectedly() -> None:
    assert (
        WorkspacePermission.WORKSPACE_DELETE
        in WORKSPACE_ROLE_PERMISSIONS[WorkspaceRole.OWNER]
    )
    assert (
        WorkspacePermission.WORKSPACE_DELETE
        not in WORKSPACE_ROLE_PERMISSIONS[WorkspaceRole.ADMIN]
    )
    assert (
        WorkspacePermission.MEMBER_ADD
        not in WORKSPACE_ROLE_PERMISSIONS[WorkspaceRole.MEMBER]
    )
