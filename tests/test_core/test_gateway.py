# Copyright 2026 Zachary Brooks
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Gateway."""

from __future__ import annotations

import pytest

from safe_agent.core.gateway import Gateway
from safe_agent.core.session import Session


class MockEventLoop:
    """Test double for EventLoop."""

    def __init__(self) -> None:
        self.calls: list[tuple[Session, str]] = []
        self.responses: list[str] = []
        self.raise_error: Exception | None = None
        self.released_sessions: list[str] = []
        self._session_locks: set[str] = set()

    async def process_turn(self, session: Session, message: str) -> str:
        if self.raise_error:
            raise self.raise_error
        self.calls.append((session, message))
        self._session_locks.add(session.id)
        if not self.responses:
            return "default response"
        return self.responses.pop(0)

    def release_session(self, session_id: str) -> None:
        """Release per-session resources (called on eviction)."""
        self.released_sessions.append(session_id)
        self._session_locks.discard(session_id)


class MockSessionManager:
    """Test double for SessionManager."""

    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}
        self.create_calls = 0
        self._on_evict_callback = None

    def create(self) -> Session:
        session = Session(id=f"session-{self.create_calls}")
        self.sessions[session.id] = session
        self.create_calls += 1
        return session

    def get(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    def set_eviction_callback(self, callback) -> None:
        """Set eviction callback."""
        self._on_evict_callback = callback

    def trigger_eviction(self, session_id: str) -> None:
        """Simulate eviction for testing."""
        session = self.sessions.pop(session_id, None)
        if session and self._on_evict_callback:
            self._on_evict_callback(session)


@pytest.mark.asyncio
async def test_gateway_submit_new_session() -> None:
    """Test submit() with a new session ID (auto-creation)."""
    session_manager = MockSessionManager()
    event_loop = MockEventLoop()
    gateway = Gateway(session_manager, event_loop)

    response, session_id = await gateway.submit("test message")

    assert response == "default response"
    assert session_id is not None
    assert session_id in session_manager.sessions
    assert len(event_loop.calls) == 1
    assert event_loop.calls[0][0].id == session_id
    assert event_loop.calls[0][1] == "test message"


@pytest.mark.asyncio
async def test_gateway_submit_existing_session() -> None:
    """Test submit() with an existing session ID."""
    session_manager = MockSessionManager()
    event_loop = MockEventLoop()
    gateway = Gateway(session_manager, event_loop)

    # Create a session first
    session = session_manager.create()

    response, session_id = await gateway.submit("test message", session.id)

    assert response == "default response"
    assert session_id == session.id
    assert len(event_loop.calls) == 1
    assert event_loop.calls[0][0].id == session.id
    assert event_loop.calls[0][1] == "test message"


@pytest.mark.asyncio
async def test_gateway_submit_unknown_session_raises_key_error() -> None:
    """Test submit() with an invalid/unknown session ID raises KeyError."""
    session_manager = MockSessionManager()
    event_loop = MockEventLoop()
    gateway = Gateway(session_manager, event_loop)

    with pytest.raises(KeyError, match="nonexistent"):
        await gateway.submit("test message", "nonexistent")


@pytest.mark.asyncio
async def test_gateway_submit_empty_message() -> None:
    """Test submit() with an empty message."""
    session_manager = MockSessionManager()
    event_loop = MockEventLoop()
    gateway = Gateway(session_manager, event_loop)

    response, session_id = await gateway.submit("")

    assert response == "default response"
    assert session_id is not None
    assert len(event_loop.calls) == 1
    assert event_loop.calls[0][1] == ""


@pytest.mark.asyncio
async def test_gateway_submit_with_custom_response() -> None:
    """Test submit() with a custom response from event loop."""
    session_manager = MockSessionManager()
    event_loop = MockEventLoop()
    event_loop.responses = ["custom response"]
    gateway = Gateway(session_manager, event_loop)

    response, session_id = await gateway.submit("test message")

    assert response == "custom response"
    assert session_id is not None
    assert len(event_loop.calls) == 1
    assert event_loop.calls[0][1] == "test message"


@pytest.mark.asyncio
async def test_gateway_submit_multiple_sessions() -> None:
    """Test submit() with multiple sessions."""
    session_manager = MockSessionManager()
    event_loop = MockEventLoop()
    gateway = Gateway(session_manager, event_loop)

    # First session
    _, session_id1 = await gateway.submit("message 1")
    # Second session
    _, session_id2 = await gateway.submit("message 2")

    assert session_id1 != session_id2
    assert len(session_manager.sessions) == 2
    assert len(event_loop.calls) == 2
    assert event_loop.calls[0][0].id == session_id1
    assert event_loop.calls[0][1] == "message 1"
    assert event_loop.calls[1][0].id == session_id2
    assert event_loop.calls[1][1] == "message 2"


@pytest.mark.asyncio
async def test_gateway_submit_propagates_event_loop_error() -> None:
    """Test submit() propagates exceptions from event loop."""
    session_manager = MockSessionManager()
    event_loop = MockEventLoop()
    event_loop.raise_error = RuntimeError("LLM provider error")
    gateway = Gateway(session_manager, event_loop)

    with pytest.raises(RuntimeError, match="LLM provider error"):
        await gateway.submit("test message")


def test_gateway_wires_eviction_callback_on_init() -> None:
    """Gateway should wire eviction callback to release EventLoop locks."""
    session_manager = MockSessionManager()
    event_loop = MockEventLoop()
    Gateway(session_manager, event_loop)

    # Verify callback was set
    assert session_manager._on_evict_callback is not None

    # Simulate a session being created and locked
    session = session_manager.create()
    event_loop._session_locks.add(session.id)

    # Trigger eviction
    session_manager.trigger_eviction(session.id)

    # Verify release_session was called
    assert session.id in event_loop.released_sessions
    assert session.id not in event_loop._session_locks
