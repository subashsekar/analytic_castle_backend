from __future__ import annotations

from typing import Protocol
from uuid import UUID

from sqlalchemy.orm import Session

from app.ai.types import DEFAULT_AI_CAPABILITIES, AIContext, MetadataSnippet
from app.core.authorization import DataSourceAccess
from app.core.config import settings
from app.enums import WorkspaceRole
from app.services.metadata_search import MetadataSearchService
from app.services.metadata_search_exceptions import MetadataSearchError
from app.services.metadata_search_types import MetadataSearchResult


class MetadataContextProvider(Protocol):
    """Retrieve compact, relevant metadata for an authorized data source.

    Implementations must reuse persisted AnalyticCastle metadata. They must not
    copy the catalog into the AI module or query the customer database.
    """

    def get_relevant_metadata(
        self,
        *,
        data_source_id: UUID,
        workspace_id: UUID,
        query: str,
        limit: int | None = None,
    ) -> tuple[MetadataSnippet, ...]: ...


class MetadataSearchContextProvider:
    """Adapter over Phase 4 metadata search. Does not implement RAG."""

    def __init__(self, session: Session) -> None:
        self._search = MetadataSearchService(session)

    def get_relevant_metadata(
        self,
        *,
        data_source_id: UUID,
        workspace_id: UUID,
        query: str,
        limit: int | None = None,
    ) -> tuple[MetadataSnippet, ...]:
        from app.ai.metadata_resolver import expand_metadata_search_terms

        resolved_limit = (
            limit if limit is not None else settings.AI_METADATA_SEARCH_LIMIT
        )
        search_query = query.strip()[: settings.METADATA_SEARCH_MAX_QUERY_LENGTH]
        if not search_query:
            return ()
        terms = list(expand_metadata_search_terms(search_query))
        if search_query not in terms:
            terms.insert(0, search_query)
        snippets: list[MetadataSnippet] = []
        seen: set[tuple[str | None, str | None, str | None]] = set()
        try:
            for term in terms:
                if len(snippets) >= resolved_limit:
                    break
                page = self._search.search_metadata(
                    data_source_id,
                    term,
                    workspace_id=workspace_id,
                    limit=resolved_limit,
                )
                for item in page.results:
                    key = (item.schema_name, item.table_name, item.column_name)
                    if key in seen:
                        continue
                    seen.add(key)
                    snippets.append(_snippet_from_result(item))
                    if len(snippets) >= resolved_limit:
                        break
        except (MetadataSearchError, ValueError):
            return tuple(snippets)
        return tuple(snippets)


def build_ai_context(access: DataSourceAccess) -> AIContext:
    """Build analyst context from an already-authorized data source access.

    Authorization must already have confirmed workspace membership. This helper
    does not load metadata or contact an LLM.
    """
    if access.via_super_admin:
        role: str | None = WorkspaceRole.OWNER.value
    elif access.member is None:
        role = None
    else:
        role = access.member.role.value
    return AIContext(
        user_id=access.user.id,
        workspace_id=access.workspace.id,
        organization_id=access.workspace.organization_id,
        data_source_id=access.data_source.id,
        data_source_name=access.data_source.name,
        data_source_type=access.data_source.type.value,
        workspace_name=access.workspace.name,
        workspace_role=role,
        allowed_capabilities=DEFAULT_AI_CAPABILITIES,
    )


def _snippet_from_result(item: MetadataSearchResult) -> MetadataSnippet:
    return MetadataSnippet(
        schema_name=item.schema_name,
        table_name=item.table_name,
        column_name=item.column_name,
        matched_name=item.matched_name,
    )
