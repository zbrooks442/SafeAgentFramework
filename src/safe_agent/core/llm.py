"""LLM protocol types for SafeAgent core."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel


class ToolCall(BaseModel):
    """Represents a model-requested tool invocation."""

    name: str
    params: dict[str, Any]


class LLMResponse(BaseModel):
    """Represents a single LLM turn response."""

    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class LLMClient(Protocol):
    """Protocol for chat-capable LLM clients used by the event loop."""

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> LLMResponse:
        """Return the next model response for the provided conversation."""
