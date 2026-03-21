"""Session models and in-memory session tracking."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class Session(BaseModel):
    """Represents a single agent conversation session.

    Attributes:
        id: Unique session identifier.
        messages: Conversation history for the session.
        metadata: Arbitrary session-scoped metadata.
        created_at: Timestamp when the session was created.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    messages: list[dict] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SessionManager:
    """Manages active in-memory sessions."""

    def __init__(self) -> None:
        """Initialise an empty active-session store."""
        self._sessions: dict[str, Session] = {}

    def create(self) -> Session:
        """Create and track a new session."""
        session = Session()
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        """Return a tracked session by ID, if present."""
        return self._sessions.get(session_id)

    def close(self, session_id: str) -> None:
        """Remove a session from active tracking if present."""
        self._sessions.pop(session_id, None)

    def list_active(self) -> list[str]:
        """Return active session IDs in insertion order."""
        return list(self._sessions)
