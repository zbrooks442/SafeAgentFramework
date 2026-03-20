"""Tests for safe_agent.modules.registry — ModuleRegistry."""

from typing import Any

import pytest

from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)
from safe_agent.modules.registry import ModuleRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_module(
    namespace: str,
    tool_names: list[str],
) -> BaseModule:
    """Create a minimal concrete BaseModule for testing.

    Args:
        namespace: The namespace to assign to the module.
        tool_names: List of tool names the module should expose.

    Returns:
        A concrete ``BaseModule`` instance.
    """

    class _Module(BaseModule):
        """Dynamically created test module."""

        def describe(self) -> ModuleDescriptor:
            """Return a descriptor with the configured namespace and tools."""
            return ModuleDescriptor(
                namespace=namespace,
                description=f"Module {namespace}.",
                tools=[
                    ToolDescriptor(
                        name=t,
                        description=f"Tool {t}.",
                        action=t,
                    )
                    for t in tool_names
                ],
            )

        async def resolve_conditions(
            self,
            tool_name: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            """Return empty conditions."""
            return {}

        async def execute(
            self,
            tool_name: str,
            params: dict[str, Any],
        ) -> ToolResult:
            """Return a default success result."""
            return ToolResult(success=True)

    return _Module()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModuleRegistry:
    """Tests for ModuleRegistry registration and lookup."""

    def test_register_and_get_tool(self) -> None:
        """A registered tool should be retrievable by name."""
        registry = ModuleRegistry()
        module = _make_module("alpha", ["alpha:Read", "alpha:Write"])
        registry.register(module)

        result = registry.get_tool("alpha:Read")
        assert result is not None
        found_module, descriptor = result
        assert found_module is module
        assert descriptor.name == "alpha:Read"

    def test_get_tool_returns_none_for_unknown(self) -> None:
        """get_tool() should return None for an unregistered tool name."""
        registry = ModuleRegistry()
        assert registry.get_tool("nonexistent:tool") is None

    def test_get_all_tool_descriptors_empty(self) -> None:
        """get_all_tool_descriptors() should return [] when no modules registered."""
        registry = ModuleRegistry()
        assert registry.get_all_tool_descriptors() == []

    def test_get_all_tool_descriptors(self) -> None:
        """get_all_tool_descriptors() should return all registered tools."""
        registry = ModuleRegistry()
        registry.register(_make_module("a", ["a:T1", "a:T2"]))
        registry.register(_make_module("b", ["b:T3"]))

        descriptors = registry.get_all_tool_descriptors()
        names = {d.name for d in descriptors}
        assert names == {"a:T1", "a:T2", "b:T3"}

    def test_namespace_collision_raises(self) -> None:
        """Registering two different modules with the same namespace raises ValueError.

        Two distinct module instances that share a namespace should collide.
        """
        registry = ModuleRegistry()
        registry.register(_make_module("clash", ["clash:A"]))
        second = _make_module("clash", ["clash:B"])
        with pytest.raises(ValueError, match="Namespace collision"):
            registry.register(second)

    def test_idempotent_registration(self) -> None:
        """Registering the same module instance twice should not raise."""
        registry = ModuleRegistry()
        module = _make_module("idem", ["idem:X"])
        registry.register(module)
        registry.register(module)  # should be a no-op
        descriptors = registry.get_all_tool_descriptors()
        assert len(descriptors) == 1

    def test_multiple_tools_from_one_module(self) -> None:
        """All tools of a module should be accessible after registration."""
        registry = ModuleRegistry()
        tools = ["ns:T1", "ns:T2", "ns:T3"]
        registry.register(_make_module("ns", tools))
        for t in tools:
            assert registry.get_tool(t) is not None

    def test_get_tool_returns_correct_descriptor(self) -> None:
        """get_tool() descriptor should match the registered ToolDescriptor."""
        registry = ModuleRegistry()
        registry.register(_make_module("z", ["z:Foo"]))
        result = registry.get_tool("z:Foo")
        assert result is not None
        _, descriptor = result
        assert descriptor.action == "z:Foo"
        assert descriptor.description == "Tool z:Foo."
