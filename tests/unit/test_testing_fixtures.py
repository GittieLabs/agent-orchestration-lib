"""Tests for test fixtures."""

import pytest
from agent_lib.testing import (
    create_test_context,
    create_test_emitter,
    create_test_flow,
    MockAgent,
)


class TestCreateTestContext:
    """Test create_test_context fixture."""

    def test_creates_context(self):
        """Test that fixture creates an ExecutionContext."""
        context = create_test_context()
        assert context is not None

    def test_context_with_initial_data(self):
        """Test creating context with initial data."""
        context = create_test_context({"key1": "value1", "key2": 42})

        assert context.get("key1") == "value1"
        assert context.get("key2") == 42

    def test_context_with_empty_data(self):
        """Test creating context with no initial data."""
        context = create_test_context()
        # Should not raise, context just empty
        assert not context.has("any_key")

    def test_context_with_custom_emitter(self):
        """Test creating context with custom emitter."""
        emitter = create_test_emitter()
        context = create_test_context(emitter=emitter)

        assert context._emitter is emitter

    def test_context_operations(self):
        """Test that context supports normal operations."""
        context = create_test_context()

        context.set("test", "value")
        assert context.get("test") == "value"
        assert context.has("test")


class TestCreateTestEmitter:
    """Test create_test_emitter fixture."""

    def test_creates_emitter(self):
        """Test that fixture creates an EventEmitter."""
        emitter = create_test_emitter()
        assert emitter is not None

    @pytest.mark.asyncio
    async def test_emitter_captures_events(self):
        """Test that emitter captures events when enabled."""
        emitter = create_test_emitter(capture_events=True)

        await emitter.emit("test_event", {"data": "value"})

        captured = emitter.get_captured_events()
        assert len(captured) == 1
        assert captured[0]["type"] == "test_event"
        assert captured[0]["data"] == {"data": "value"}

    @pytest.mark.asyncio
    async def test_emitter_captures_multiple_events(self):
        """Test capturing multiple events."""
        emitter = create_test_emitter(capture_events=True)

        await emitter.emit("event1", {"num": 1})
        await emitter.emit("event2", {"num": 2})
        await emitter.emit("event1", {"num": 3})

        captured = emitter.get_captured_events()
        assert len(captured) == 3
        assert captured[0]["type"] == "event1"
        assert captured[1]["type"] == "event2"
        assert captured[2]["type"] == "event1"

    @pytest.mark.asyncio
    async def test_emitter_clear_captured_events(self):
        """Test clearing captured events."""
        emitter = create_test_emitter(capture_events=True)

        await emitter.emit("test", {})
        assert len(emitter.get_captured_events()) == 1

        emitter.clear_captured_events()
        assert len(emitter.get_captured_events()) == 0

    @pytest.mark.asyncio
    async def test_emitter_without_capture(self):
        """Test that emitter without capture doesn't have capture methods."""
        emitter = create_test_emitter(capture_events=False)

        # Should work normally
        await emitter.emit("test", {})

        # Should not have capture methods
        assert not hasattr(emitter, 'get_captured_events')

    @pytest.mark.asyncio
    async def test_emitter_still_emits_events(self):
        """Test that capturing doesn't break normal event emission."""
        emitter = create_test_emitter(capture_events=True)

        events_received = []
        emitter.on("test", lambda e: events_received.append(e))

        await emitter.emit("test", {"data": "value"})

        # Both captured and received via subscription
        assert len(emitter.get_captured_events()) == 1
        assert len(events_received) == 1


class TestCreateTestFlow:
    """Test create_test_flow fixture."""

    def test_creates_flow(self):
        """Test that fixture creates a Flow."""
        flow = create_test_flow()
        assert flow is not None
        assert flow.name == "test_flow"

    def test_flow_with_custom_name(self):
        """Test creating flow with custom name."""
        flow = create_test_flow(name="my_flow")
        assert flow.name == "my_flow"

    def test_flow_with_agents(self):
        """Test creating flow with agents."""
        agent1 = MockAgent("agent1", return_value=1)
        agent2 = MockAgent("agent2", return_value=2)

        flow = create_test_flow(agents=[agent1, agent2])

        # Check agents were added
        assert len(flow._agents) == 2
        assert agent1 in flow._agents.values()
        assert agent2 in flow._agents.values()

    def test_flow_with_empty_agents(self):
        """Test creating flow with no agents."""
        flow = create_test_flow(agents=[])
        assert len(flow._agents) == 0

    def test_flow_with_custom_emitter(self):
        """Test creating flow with custom emitter."""
        emitter = create_test_emitter()
        flow = create_test_flow(emitter=emitter)

        assert flow._emitter is emitter

    @pytest.mark.asyncio
    async def test_flow_executes_agents(self):
        """Test that created flow can execute agents."""
        agent1 = MockAgent("agent1", return_value={"step": 1})
        agent2 = MockAgent("agent2", return_value={"step": 2})

        flow = create_test_flow(agents=[agent1, agent2])

        result = await flow.execute_sequential({"input": "data"})

        assert agent1.call_count == 1
        assert agent2.call_count == 1
        assert result == {"step": 2}

    def test_flow_operations(self):
        """Test that flow supports normal operations."""
        flow = create_test_flow()

        # Should be able to add agents
        agent = MockAgent("test", return_value=42)
        flow.add_agent(agent)

        assert "test" in flow._agents
