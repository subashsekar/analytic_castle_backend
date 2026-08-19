from __future__ import annotations

from collections.abc import Callable

from app.connectors.base import DataConnector
from app.connectors.exceptions import UnsupportedConnectorError
from app.enums import DataSourceType

ConnectorBuilder = Callable[[], DataConnector]

_BUILDERS: dict[DataSourceType, ConnectorBuilder] = {}


def recognized_types() -> frozenset[DataSourceType]:
    """Data source types the registry knows about.

    Recognition does not mean an implementation exists.
    """
    return frozenset(DataSourceType)


def implemented_types() -> frozenset[DataSourceType]:
    return frozenset(_BUILDERS)


def is_implemented(source_type: DataSourceType) -> bool:
    return source_type in _BUILDERS


def register_connector(
    source_type: DataSourceType,
    builder: ConnectorBuilder,
) -> None:
    _BUILDERS[source_type] = builder


def unregister_connector(source_type: DataSourceType) -> None:
    _BUILDERS.pop(source_type, None)


def create_connector(source_type: DataSourceType) -> DataConnector:
    """Return a connector for ``source_type``.

    Raises ``UnsupportedConnectorError`` when the type is recognized but no
    implementation has been registered. Does not fall back to another type.
    """
    builder = _BUILDERS.get(source_type)
    if builder is None:
        raise UnsupportedConnectorError(
            f"Connector type {source_type.value} is not implemented"
        )
    return builder()
