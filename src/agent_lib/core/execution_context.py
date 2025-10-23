"""
ExecutionContext - Dependency injection container for agent execution.

Provides type-safe service registration and retrieval with support for:
- Singleton services (shared across all contexts)
- Factory services (new instance per request)
- Transient services (new instance per context)
- Child contexts (isolated scope with parent fallback)
"""

from typing import Any, Callable, Dict, Optional, TypeVar, Union
from loguru import logger

T = TypeVar("T")


class ExecutionContext:
    """Dependency injection container for managing shared state and services.

    The ExecutionContext provides a flexible service registry that supports:
    - Singleton pattern for shared services
    - Factory pattern for transient instances
    - Child contexts for isolated execution (e.g., parallel agents)
    - Parent lookup fallback for child contexts

    Example:
        ```python
        # Create context and register services
        context = ExecutionContext()
        context.register_singleton("db", database_client)
        context.register_factory("logger", lambda: create_logger())

        # Retrieve services
        db = context.get("db")
        logger = context.get("logger")

        # Create child context for parallel execution
        child = context.create_child()
        child.register_singleton("trace_id", "child-123")

        # Child inherits parent services but has isolated state
        assert child.get("db") is context.get("db")  # Same instance
        assert child.get("trace_id") == "child-123"  # Child-specific
        ```
    """

    def __init__(self, parent: Optional["ExecutionContext"] = None):
        """Initialize execution context.

        Args:
            parent: Optional parent context for service lookup fallback
        """
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._parent = parent

        logger.debug(
            f"ExecutionContext created (parent={'yes' if parent else 'no'})"
        )

    def register_singleton(self, name: str, instance: Any) -> None:
        """Register a singleton service instance.

        Singleton services are shared across all contexts and their children.
        Use this for database connections, configuration, etc.

        Args:
            name: Service name for retrieval
            instance: Service instance to register

        Example:
            ```python
            context.register_singleton("db", database_client)
            context.register_singleton("config", app_config)
            ```
        """
        self._singletons[name] = instance
        logger.debug(f"Registered singleton service: {name}")

    def register_factory(
        self, name: str, factory: Callable[[], Any]
    ) -> None:
        """Register a service factory for creating transient instances.

        Factory functions are called each time the service is requested,
        creating a new instance. Use this for loggers, temporary resources, etc.

        Args:
            name: Service name for retrieval
            factory: Callable that creates and returns a service instance

        Example:
            ```python
            context.register_factory("logger", lambda: create_logger())
            context.register_factory("session", lambda: create_session())
            ```
        """
        self._factories[name] = factory
        logger.debug(f"Registered factory service: {name}")

    def register(self, name: str, value: Any) -> None:
        """Register a simple value or service instance.

        This is a convenience method for registering context-specific values
        like job_id, user_id, etc. Values are not shared with child contexts
        unless explicitly registered in the child.

        Args:
            name: Service name for retrieval
            value: Value or service instance to register

        Example:
            ```python
            context.register("job_id", "job-123")
            context.register("user_id", "user-456")
            context.register("debug_mode", True)
            ```
        """
        self._services[name] = value
        logger.debug(f"Registered service: {name}")

    def get(self, name: str, default: Any = None) -> Any:
        """Retrieve a service by name.

        Lookup order:
        1. Check local services
        2. Check local singletons
        3. Check local factories (create new instance)
        4. If parent exists, check parent recursively
        5. Return default if not found

        Args:
            name: Service name to retrieve
            default: Default value if service not found

        Returns:
            Service instance or default value

        Example:
            ```python
            db = context.get("db")
            user_id = context.get("user_id", "unknown")
            ```
        """
        # Check local services first
        if name in self._services:
            return self._services[name]

        # Check singletons
        if name in self._singletons:
            return self._singletons[name]

        # Check factories (create new instance)
        if name in self._factories:
            instance = self._factories[name]()
            logger.debug(f"Created instance from factory: {name}")
            return instance

        # Fallback to parent if exists
        if self._parent is not None:
            return self._parent.get(name, default)

        # Return default if not found
        return default

    def has(self, name: str) -> bool:
        """Check if a service is registered.

        Checks local context and parent contexts recursively.

        Args:
            name: Service name to check

        Returns:
            True if service exists, False otherwise

        Example:
            ```python
            if context.has("db"):
                db = context.get("db")
            ```
        """
        if name in self._services or name in self._singletons or name in self._factories:
            return True

        if self._parent is not None:
            return self._parent.has(name)

        return False

    def create_child(self, **overrides: Any) -> "ExecutionContext":
        """Create a child context with isolated state.

        Child contexts:
        - Inherit all parent services via fallback lookup
        - Can override parent services with local values
        - Useful for parallel execution where each child needs isolated state

        Args:
            **overrides: Key-value pairs to register in child context

        Returns:
            New child context with this context as parent

        Example:
            ```python
            # Parent context
            context = ExecutionContext()
            context.register("job_id", "parent-job")
            context.register_singleton("db", database)

            # Child context for parallel execution
            child1 = context.create_child(trace_id="child-1")
            child2 = context.create_child(trace_id="child-2")

            # Children inherit parent services
            assert child1.get("db") is context.get("db")

            # Children have isolated state
            assert child1.get("trace_id") == "child-1"
            assert child2.get("trace_id") == "child-2"
            ```
        """
        child = ExecutionContext(parent=self)

        # Register overrides
        for key, value in overrides.items():
            child.register(key, value)

        logger.debug(
            f"Created child context with {len(overrides)} overrides"
        )

        return child

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging and debugging.

        Returns dictionary of all services, singletons, and factories
        in the current context (not including parent).

        Returns:
            Dictionary with service names and values

        Example:
            ```python
            context_data = context.to_dict()
            logger.info(f"Context state: {context_data}")
            ```
        """
        return {
            "services": {k: repr(v) for k, v in self._services.items()},
            "singletons": {k: repr(v) for k, v in self._singletons.items()},
            "factories": {k: repr(v) for k, v in self._factories.items()},
            "has_parent": self._parent is not None,
        }

    def __repr__(self) -> str:
        """String representation of context."""
        service_count = (
            len(self._services)
            + len(self._singletons)
            + len(self._factories)
        )
        parent_info = " (with parent)" if self._parent else ""
        return f"ExecutionContext({service_count} services{parent_info})"
