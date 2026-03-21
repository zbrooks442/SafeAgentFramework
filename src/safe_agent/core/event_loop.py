"""Session-scoped event loop for LLM and tool orchestration."""

from __future__ import annotations

import asyncio
import json

from safe_agent.core.dispatcher import ToolDispatcher
from safe_agent.core.llm import LLMClient
from safe_agent.core.session import Session
from safe_agent.modules.registry import ModuleRegistry


class EventLoop:
    """Coordinates LLM responses and tool execution for a session.

    Callers should invoke :meth:`release_session` when a session is done so
    per-session locks do not accumulate indefinitely.
    """

    def __init__(
        self,
        dispatcher: ToolDispatcher,
        llm_client: LLMClient,
        registry: ModuleRegistry,
        max_turns: int = 10,
    ) -> None:
        """Initialise the event loop dependencies and turn limit."""
        self._dispatcher = dispatcher
        self._llm_client = llm_client
        self._registry = registry
        self._max_turns = max_turns
        self._session_locks: dict[str, asyncio.Lock] = {}

    def release_session(self, session_id: str) -> None:
        """Release any per-session resources once a session is finished."""
        self._session_locks.pop(session_id, None)

    async def process_turn(self, session: Session, user_message: str) -> str:
        """Process one user turn until the model returns final text.

        The loop serializes work per session, appends messages to the session
        transcript, dispatches any model-requested tools, and enforces a hard
        turn limit by making a final text-only model call when necessary.
        """
        lock = self._session_locks.setdefault(session.id, asyncio.Lock())

        async with lock:
            session.messages.append({"role": "user", "content": user_message})
            tool_definitions = [
                descriptor.model_dump()
                for descriptor in self._registry.get_all_tool_descriptors()
            ]

            turn_count = 0
            while True:
                response = await self._llm_client.chat(
                    session.messages,
                    tool_definitions,
                )

                if response.content is not None:
                    session.messages.append(
                        {"role": "assistant", "content": response.content}
                    )
                    return response.content

                if response.tool_calls:
                    session.messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                tool_call.model_dump()
                                for tool_call in response.tool_calls
                            ],
                        }
                    )
                    for tool_call in response.tool_calls:
                        try:
                            result = await self._dispatcher.dispatch(
                                tool_call.name,
                                tool_call.params,
                                session.id,
                            )
                            content = json.dumps(result.model_dump())
                        except Exception as exc:
                            content = json.dumps({"error": str(exc)})

                        session.messages.append(
                            {
                                "role": "tool",
                                "name": tool_call.name,
                                "content": content,
                            }
                        )

                turn_count += 1
                if turn_count >= self._max_turns:
                    final_response = await self._llm_client.chat(session.messages, [])
                    final_text = final_response.content or ""
                    session.messages.append(
                        {"role": "assistant", "content": final_text}
                    )
                    return final_text

                if not response.tool_calls:
                    session.messages.append({"role": "assistant", "content": ""})
                    return ""
