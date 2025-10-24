"""Unit tests for ExecutionContext."""

import pytest
from agent_lib.core import ExecutionContext


class TestExecutionContextBasics:
    """Test basic ExecutionContext functionality."""

    def test_create_context(self):
        """Test creating an execution context."""
        context = ExecutionContext()
        assert context is not None
        assert repr(context) == "ExecutionContext(0 services)"

    def test_register_and_get_service(self):
        """Test registering and retrieving a simple service."""
        context = ExecutionContext()
        context.register("test_key", "test_value")

        assert context.has("test_key")
        assert context.get("test_key") == "test_value"

    def test_get_nonexistent_service(self):
        """Test getting a service that doesn't exist."""
        context = ExecutionContext()

        assert not context.has("nonexistent")
        assert context.get("nonexistent") is None
        assert context.get("nonexistent", "default") == "default"

    def test_register_multiple_services(self):
        """Test registering multiple services."""
        context = ExecutionContext()
        context.register("service1", "value1")
        context.register("service2", "value2")
        context.register("service3", "value3")

        assert context.get("service1") == "value1"
        assert context.get("service2") == "value2"
        assert context.get("service3") == "value3"


class TestExecutionContextSingletons:
    """Test singleton service functionality."""

    def test_register_singleton(self):
        """Test registering a singleton service."""
        context = ExecutionContext()

        class TestService:
            pass

        service = TestService()
        context.register_singleton("test_service", service)

        # Same instance should be returned
        assert context.get("test_service") is service
        assert context.get("test_service") is service

    def test_singleton_shared_with_children(self):
        """Test that singletons are shared with child contexts."""
        context = ExecutionContext()

        shared_service = {"shared": True}
        context.register_singleton("shared", shared_service)

        child = context.create_child()

        # Child should get same instance
        assert child.get("shared") is shared_service


class TestExecutionContextFactories:
    """Test factory service functionality."""

    def test_register_factory(self):
        """Test registering a factory service."""
        context = ExecutionContext()

        call_count = []

        def factory():
            call_count.append(1)
            return {"instance": len(call_count)}

        context.register_factory("test_factory", factory)

        # Each get() should create new instance
        instance1 = context.get("test_factory")
        instance2 = context.get("test_factory")

        assert instance1 != instance2
        assert instance1["instance"] == 1
        assert instance2["instance"] == 2
        assert len(call_count) == 2

    def test_factory_inherited_by_children(self):
        """Test that factories are inherited by child contexts."""
        context = ExecutionContext()

        counter = []

        def factory():
            counter.append(1)
            return {"count": len(counter)}

        context.register_factory("counter", factory)

        child = context.create_child()

        # Child should use parent's factory
        result = child.get("counter")
        assert result["count"] == 1


class TestExecutionContextChildContexts:
    """Test child context functionality."""

    def test_create_child_context(self):
        """Test creating a child context."""
        parent = ExecutionContext()
        parent.register("parent_value", "from_parent")

        child = parent.create_child()

        # Child should inherit parent value
        assert child.get("parent_value") == "from_parent"

    def test_child_context_overrides(self):
        """Test that child contexts can override parent values."""
        parent = ExecutionContext()
        parent.register("key", "parent_value")

        child = parent.create_child(key="child_value")

        # Child should have its own value
        assert child.get("key") == "child_value"
        # Parent should still have original value
        assert parent.get("key") == "parent_value"

    def test_child_context_isolation(self):
        """Test that child contexts are isolated from each other."""
        parent = ExecutionContext()

        child1 = parent.create_child(id="child1")
        child2 = parent.create_child(id="child2")

        assert child1.get("id") == "child1"
        assert child2.get("id") == "child2"
        assert parent.get("id") is None

    def test_nested_child_contexts(self):
        """Test creating nested child contexts."""
        root = ExecutionContext()
        root.register("level", "root")

        child = root.create_child()
        grandchild = child.create_child()

        # All should inherit root value
        assert root.get("level") == "root"
        assert child.get("level") == "root"
        assert grandchild.get("level") == "root"

    def test_child_has_parent_services(self):
        """Test that child.has() checks parent contexts."""
        parent = ExecutionContext()
        parent.register("parent_service", "value")

        child = parent.create_child()

        assert child.has("parent_service")


class TestExecutionContextUtility:
    """Test utility methods."""

    def test_to_dict(self):
        """Test converting context to dictionary."""
        context = ExecutionContext()
        context.register("service1", "value1")
        context.register_singleton("singleton1", "singleton_value")
        context.register_factory("factory1", lambda: "factory_value")

        data = context.to_dict()

        assert "services" in data
        assert "singletons" in data
        assert "factories" in data
        assert "has_parent" in data
        assert data["has_parent"] is False

    def test_to_dict_with_parent(self):
        """Test to_dict with parent context."""
        parent = ExecutionContext()
        child = parent.create_child()

        data = child.to_dict()
        assert data["has_parent"] is True

    def test_repr(self):
        """Test string representation."""
        context = ExecutionContext()
        assert "ExecutionContext" in repr(context)
        assert "0 services" in repr(context)

        context.register("test", "value")
        assert "1 services" in repr(context)


class TestExecutionContextComplexScenarios:
    """Test complex usage scenarios."""

    def test_mixed_service_types(self):
        """Test context with mixed service types."""
        context = ExecutionContext()

        # Simple value
        context.register("config", {"debug": True})

        # Singleton
        class Database:
            def __init__(self):
                self.connected = True

        db = Database()
        context.register_singleton("db", db)

        # Factory
        context.register_factory("logger", lambda: {"level": "info"})

        # All should work together
        assert context.get("config")["debug"] is True
        assert context.get("db").connected is True
        assert context.get("logger")["level"] == "info"

    def test_parallel_execution_pattern(self):
        """Test pattern for parallel agent execution."""
        # Parent context with shared resources
        parent = ExecutionContext()
        parent.register_singleton("database", {"type": "postgres"})
        parent.register("job_id", "parent-job-123")

        # Create child contexts for parallel tasks
        task1 = parent.create_child(task_id="task-1", trace_id="trace-1")
        task2 = parent.create_child(task_id="task-2", trace_id="trace-2")

        # Children share database
        assert task1.get("database") is parent.get("database")
        assert task2.get("database") is parent.get("database")

        # Children have isolated task data
        assert task1.get("task_id") == "task-1"
        assert task2.get("task_id") == "task-2"
        assert task1.get("trace_id") == "trace-1"
        assert task2.get("trace_id") == "trace-2"

        # Children inherit job_id
        assert task1.get("job_id") == "parent-job-123"
        assert task2.get("job_id") == "parent-job-123"

    def test_service_override_precedence(self):
        """Test service lookup precedence."""
        context = ExecutionContext()

        # Register in different ways
        context.register("key", "simple")
        context.register_singleton("key_singleton", "singleton")
        context.register_factory("key_factory", lambda: "factory")

        # Local services should be found first
        assert context.get("key") == "simple"
        assert context.get("key_singleton") == "singleton"

        # Factory should create new instance each time
        val1 = context.get("key_factory")
        val2 = context.get("key_factory")
        assert val1 == "factory"
        assert val2 == "factory"
