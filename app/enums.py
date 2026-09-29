import enum


class UserRole(str, enum.Enum):
    SUPER_ADMIN = "SUPER_ADMIN"
    ADMIN = "ADMIN"
    USER = "USER"


class WorkspaceRole(str, enum.Enum):
    OWNER = "OWNER"
    ADMIN = "ADMIN"
    MEMBER = "MEMBER"


class WorkspacePermission(str, enum.Enum):
    WORKSPACE_READ = "workspace:read"
    WORKSPACE_UPDATE = "workspace:update"
    WORKSPACE_DELETE = "workspace:delete"
    MEMBER_READ = "member:read"
    MEMBER_ADD = "member:add"
    MEMBER_UPDATE = "member:update"
    MEMBER_REMOVE = "member:remove"
    DATA_SOURCE_READ = "data_source:read"
    DATA_SOURCE_QUERY = "data_source:query"
    DATA_SOURCE_CREATE = "data_source:create"
    DATA_SOURCE_UPDATE = "data_source:update"
    DATA_SOURCE_DELETE = "data_source:delete"
    DATA_SOURCE_TEST = "data_source:test"


class DataSourceType(str, enum.Enum):
    POSTGRESQL = "POSTGRESQL"
    MYSQL = "MYSQL"
    CSV = "CSV"
    EXCEL = "EXCEL"
    GOOGLE_SHEETS = "GOOGLE_SHEETS"
    REST_API = "REST_API"


class DataSourceStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    ERROR = "ERROR"


class DataSourceTableType(str, enum.Enum):
    TABLE = "TABLE"
    VIEW = "VIEW"


class DataSourceRelationshipType(str, enum.Enum):
    ONE_TO_ONE = "ONE_TO_ONE"
    ONE_TO_MANY = "ONE_TO_MANY"
    MANY_TO_ONE = "MANY_TO_ONE"
    MANY_TO_MANY = "MANY_TO_MANY"


class MetadataSyncStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class MetadataSearchType(str, enum.Enum):
    SCHEMA = "SCHEMA"
    TABLE = "TABLE"
    COLUMN = "COLUMN"


class AnalysisSessionStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class QueryHistoryStatus(str, enum.Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    CORRECTED = "CORRECTED"


class ColumnSensitivity(str, enum.Enum):
    PUBLIC = "PUBLIC"
    SENSITIVE = "SENSITIVE"
    PII = "PII"
    SECRET = "SECRET"


class AgentPhase(str, enum.Enum):
    INITIAL = "INITIAL"
    INTENT = "INTENT"
    PLANNING = "PLANNING"
    AWAITING_CLARIFICATION = "AWAITING_CLARIFICATION"
    EXECUTING = "EXECUTING"
    RESPONDING = "RESPONDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
