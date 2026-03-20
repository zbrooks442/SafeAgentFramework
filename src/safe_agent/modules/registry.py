"""Module registry for discovering and dispatching SafeAgent modules."""

from importlib.metadata import entry_points

from safe_agent.modules.base import BaseModule, ToolDescriptor


class ModuleRegistry:
    """Registry for SafeAgent modules supporting discovery and dispatch.

    Modules can be registered manually via ``register()`` or discovered
    automatically from installed packages via ``discover()``.  Namespace
    collisions between modules raise a ``ValueError`` at registration time.

    Example:
        >>> registry = ModuleRegistry()
        >>> registry.register(my_module)
        >>> module, descriptor = registry.get_tool("my_namespace:my_tool")
    """

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self._tool_map: dict[str, tuple[BaseModule, ToolDescriptor]] = {}
        self._namespace_map: dict[str, BaseModule] = {}

    def register(self, module: BaseModule) -> None:
        """Register a module instance into the registry.

        Args:
            module: A concrete ``BaseModule`` instance to register.

        Raises:
            ValueError: If the module's namespace is already registered by a
                different module instance.
        """
        descriptor = module.describe()
        namespace = descriptor.namespace

        if namespace in self._namespace_map:
            existing = self._namespace_map[namespace]
            if existing is not module:
                raise ValueError(
                    f"Namespace collision: '{namespace}' is already registered "
                    f"by {type(existing).__name__}."
                )
            # Same module registered twice — idempotent, just return.
            return

        self._namespace_map[namespace] = module
        for tool in descriptor.tools:
            self._tool_map[tool.name] = (module, tool)

    def discover(self) -> None:
        """Discover and register modules from installed package entry points.

        Scans the ``safe_agent.modules`` entry-point group, instantiates each
        advertised class, and calls ``register()`` on it.  Namespace collisions
        are propagated as ``ValueError``.
        """
        eps = entry_points(group="safe_agent.modules")
        for ep in eps:
            module_class = ep.load()
            instance: BaseModule = module_class()
            self.register(instance)

    def get_tool(self, tool_name: str) -> tuple[BaseModule, ToolDescriptor] | None:
        """Look up a tool by its fully-qualified name.

        Args:
            tool_name: The tool name as registered (e.g. ``"fs:ReadFile"``).

        Returns:
            A ``(BaseModule, ToolDescriptor)`` tuple if found, else ``None``.
        """
        return self._tool_map.get(tool_name)

    def get_all_tool_descriptors(self) -> list[ToolDescriptor]:
        """Return descriptors for every registered tool.

        Returns:
            A list of ``ToolDescriptor`` objects across all registered modules.
        """
        return [td for _, td in self._tool_map.values()]
