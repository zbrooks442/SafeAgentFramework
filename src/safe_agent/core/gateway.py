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

"""Gateway entrypoint for session-aware agent chat."""

from __future__ import annotations

from safe_agent.core.event_loop import EventLoop
from safe_agent.core.session import Session, SessionManager


class Gateway:
    """Routes incoming chat messages to the correct session event loop."""

    def __init__(
        self,
        session_manager: SessionManager,
        event_loop: EventLoop,
    ) -> None:
        """Initialise the gateway dependencies.

        Automatically wires the session eviction callback to release EventLoop
        session locks when sessions are evicted by TTL or LRU.
        """
        self._session_manager = session_manager
        self._event_loop = event_loop

        # Wire eviction callback to release EventLoop session locks
        # when sessions are evicted by TTL or LRU (fixes #59)
        session_manager.set_eviction_callback(
            lambda s: event_loop.release_session(s.id)
        )

    async def submit(
        self, message: str, session_id: str | None = None
    ) -> tuple[str, str]:
        """Submit a user message to a session and return the response and session ID.

        Args:
            message: The user message to process.
            session_id: Existing session identifier, or ``None`` to create a new one.

        Returns:
            A ``(response, session_id)`` tuple so callers can continue the conversation.

        Raises:
            KeyError: If *session_id* is provided but no tracked session exists.
        """
        session: Session
        if session_id is None:
            session = self._session_manager.create()
        else:
            existing = self._session_manager.get(session_id)
            if existing is None:
                raise KeyError(session_id)
            session = existing

        response = await self._event_loop.process_turn(session, message)
        return response, session.id
