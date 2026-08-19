from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, model_validator

from app.enums import WorkspaceRole


class WorkspaceMemberCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: UUID
    role: WorkspaceRole = WorkspaceRole.MEMBER


class WorkspaceMemberUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: WorkspaceRole


class WorkspaceMemberRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    workspace_id: UUID
    user_id: UUID
    first_name: str
    last_name: str
    email: str
    role: WorkspaceRole
    created_at: datetime

    @model_validator(mode="before")
    @classmethod
    def populate_user_fields(cls, data: Any) -> Any:
        user = getattr(data, "user", None)
        if user is None or isinstance(data, dict):
            return data
        return {
            "id": data.id,
            "workspace_id": data.workspace_id,
            "user_id": data.user_id,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "email": user.email,
            "role": data.role,
            "created_at": data.created_at,
        }
