import asyncio
from typing import Any

import pytest

from safe_agent.core import EventLoop, LLMResponse, Session, ToolCall
from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)
from safe_agent.modules.registry import ModuleRegistry


class _FakeDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], str]] = []

    async def dispatch(
        self,
        tool_name: str,
        params: dict[str, Any],
        session_id: str,
    ) -> ToolResult[Any]:
        self.calls.append((tool_name, params, session_id))
        return ToolResult(success=True, data={"echo": params})


class _FakeLLM:
    def __init__(self, responses: list[LLMResponse], delay: float = 0.0) -> None:
        self._responses = responses
        self._delay = delay
        self.calls: list[tuple[list[dict], list[dict]]] = []
        self.active_calls = 0
        self.max_concurrent_calls = 0

    async def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.active_calls += 1
        self.max_concurrent_calls = max(self.max_concurrent_calls, self.active_calls)
        self.calls.append((list(messages), list(tools)))
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            return self._responses.pop(0)
        finally:
            self.active_calls -= 1


@pytest.fixture
def registry() -> ModuleRegistry:
    registry = ModuleRegistry()
    registry.register(_StaticToolModule())
    return registry


class _StaticToolModule(BaseModule):
    def describe(self) -> ModuleDescriptor:
        return ModuleDescriptor(
            namespace="demo",
            description="Demo tools",
            tools=[
                ToolDescriptor(
                    name="demo:echo",
                    description="Echo params",
                    parameters={"type": "object"},
                    action="demo:Echo",
                )
            ],
        )

    async def resolve_conditions(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        return {}

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult[Any]:
        return ToolResult(success=True, data=params)


def test_text_response_immediate(registry: ModuleRegistry) -> None:
    dispatcher = _FakeDispatcher()
    llm = _FakeLLM([LLMResponse(content="done")])
    event_loop = EventLoop(dispatcher, llm, registry)
    session = Session(id="session-1")

    result = asyncio.run(event_loop.process_turn(session, "hello"))

    assert result == "done"
    assert dispatcher.calls == []
    assert session.messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "done"},
    ]


def test_tool_call_then_text(registry: ModuleRegistry) -> None:
    dispatcher = _FakeDispatcher()
    llm = _FakeLLM(
        [
            LLMResponse(
                tool_calls=[ToolCall(name="demo:echo", params={"value": 1})]
            ),
            LLMResponse(content="tool complete"),
        ]
    )
    event_loop = EventLoop(dispatcher, llm, registry)
    session = Session(id="session-1")

    result = asyncio.run(event_loop.process_turn(session, "run tool"))

    assert result == "tool complete"
    assert dispatcher.calls == [("demo:echo", {"value": 1}, "session-1")]
    assert session.messages[0] == {"role": "user", "content": "run tool"}
    assert session.messages[1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"name": "demo:echo", "params": {"value": 1}}],
    }
    assert session.messages[2]["role"] == "tool"
    assert session.messages[2]["name"] == "demo:echo"
    assert session.messages[3] == {"role": "assistant", "content": "tool complete"}


def test_turn_limit_enforced(registry: ModuleRegistry) -> None:
    dispatcher = _FakeDispatcher()
    llm = _FakeLLM(
        [
            LLMResponse(tool_calls=[ToolCall(name="demo:echo", params={"step": 1})]),
            LLMResponse(tool_calls=[ToolCall(name="demo:echo", params={"step": 2})]),
            LLMResponse(content="forced final"),
        ]
    )
    event_loop = EventLoop(dispatcher, llm, registry, max_turns=2)
    session = Session(id="session-1")

    result = asyncio.run(event_loop.process_turn(session, "loop"))

    assert result == "forced final"
    assert [call[1] for call in dispatcher.calls] == [{"step": 1}, {"step": 2}]
    assert llm.calls[-1][1] == []
    assert session.messages[-1] == {"role": "assistant", "content": "forced final"}


@pytest.mark.asyncio
async def test_session_lock_serializes_concurrent_calls(
    registry: ModuleRegistry,
) -> None:
    dispatcher = _FakeDispatcher()
    llm = _FakeLLM(
        [
            LLMResponse(content="first"),
            LLMResponse(content="second"),
        ],
        delay=0.05,
    )
    event_loop = EventLoop(dispatcher, llm, registry)
    session = Session(id="session-1")

    results = await asyncio.gather(
        event_loop.process_turn(session, "one"),
        event_loop.process_turn(session, "two"),
    )

    assert results == ["first", "second"]
    assert llm.max_concurrent_calls == 1
    assert session.messages == [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "second"},
    ]
