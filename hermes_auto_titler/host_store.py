"""The plugin's access to Hermes' session store (Hermes 0.23.3+).

Hermes 0.23.3 moved session storage behind ``persistence_assembly``; ``hermes_state.SessionDB`` is a deprecated stand-in without the title-provenance reads and writes the plugin needs. ``SessionStore`` gives the titler the session operations it uses over a storage-interface handle. Host modules are imported on use so the unit tests run without a Hermes checkout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from persistence_assembly import Persistence


def open_session_store() -> SessionStore:
    """The current profile's session store, released with ``close()``."""
    from persistence_assembly import open_persistence, profile_scope

    return SessionStore(open_persistence(profile_scope(), eager=True))


class SessionStore:
    """The titler's session operations over a ``persistence_assembly`` handle.

    It has no ``_execute_write``: the storage interface offers no compare-and-set that replaces one llm title with another, so the titler takes its fail-closed two-step write instead (user title, read back, then llm provenance).
    """

    def __init__(self, store: Persistence) -> None:
        self._store = store

    def close(self) -> None:
        self._store.close()

    def resolve_session_id(self, session_id_or_prefix: str) -> Optional[str]:
        return self._store.sessions.resolve_id(session_id_or_prefix)

    def get_session_title(self, session_id: str) -> Optional[str]:
        return self._store.sessions.get_title(session_id)

    def get_session_title_source(self, session_id: str) -> Optional[str]:
        source = self._store.sessions.get_title_source(session_id)
        return None if source is None else source.value

    def set_session_title(self, session_id: str, title: str) -> bool:
        return self._store.sessions.set_title(session_id, title)

    def set_session_title_source(self, session_id: str, source: str) -> bool:
        from persistence.conversation.sessions import TitleSource

        return self._store.sessions.set_title_source(session_id, TitleSource(source))

    def set_auto_title(self, session_id: str, title: str, source: str) -> bool:
        from persistence.conversation.sessions import TitleSource

        return self._store.sessions.set_auto_title(session_id, title, source=TitleSource(source))

    def get_messages_as_conversation(self, session_id: str, include_ancestors: bool = False, include_compacted: bool = False) -> List[Dict[str, Any]]:
        from persistence.conversation.messages import ConversationRequest, TranscriptView

        # DISPLAY is the old include_compacted read: live rows plus compaction-archived ones, never rewound rows.
        view = TranscriptView.DISPLAY if include_compacted else TranscriptView.ACTIVE
        request = ConversationRequest(view=view, include_lineage=include_ancestors)
        return [message.as_openai_message() for message in self._store.messages.read_conversation(session_id, request)]

    def list_sessions_rich(self, limit: int = 20, include_children: bool = False, min_message_count: int = 0, order_by_last_active: bool = False) -> List[Dict[str, Any]]:
        from persistence.conversation.sessions import SessionOrder, SessionQuery, SessionWindow

        query = SessionQuery(
            include_children=include_children,
            min_message_count=min_message_count,
            order=SessionOrder.LAST_ACTIVE if order_by_last_active else SessionOrder.STARTED_AT,
        )
        return [entry.as_list_row() for entry in self._store.sessions.list_sessions(query, SessionWindow(limit=limit))]

    def record_auxiliary_usage(
        self,
        session_id: str,
        task: str,
        *,
        model: Optional[str] = None,
        billing_provider: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        estimated_cost_usd: Optional[float] = None,
    ) -> None:
        from persistence.conversation.usage import TokenCounts

        tokens = TokenCounts(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        self._store.usage.record_auxiliary_usage(
            session_id, task, model=model, billing_provider=billing_provider, tokens=tokens, estimated_cost_usd=estimated_cost_usd
        )
