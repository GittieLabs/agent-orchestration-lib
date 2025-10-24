"""Tests for assertion utilities."""

import pytest
from agent_lib.testing import (
    MockAgent,
    AgentSpy,
    create_test_context,
    create_test_emitter,
    assert_agent_called,
    assert_event_emitted,
    assert_context_has,
)
from agent_lib.testing.assertions import AssertionError as TestAssertionError


class TestAssertAgentCalled:
    """Test assert_agent_called function."""

    @pytest.mark.asyncio
    async def test_assert_called_once(self):
        """Test asserting agent called exactly once."""
        mock = MockAgent("test", return_value=42)
        context = create_test_context()

        await mock.execute({}, context)

        # Should not raise
        assert_agent_called(mock, times=1)

    @pytest.mark.asyncio
    async def test_assert_called_fails_wrong_count(self):
        """Test that assertion fails with wrong call count."""
        mock = MockAgent("test", return_value=42)
        context = create_test_context()

        await mock.execute({}, context)
        await mock.execute({}, context)

        with pytest.raises(TestAssertionError, match="2 time"):
            assert_agent_called(mock, times=1)

    @pytest.mark.asyncio
    async def test_assert_called_at_least(self):
        """Test asserting agent called at least N times."""
        mock = MockAgent("test", return_value=42)
        context = create_test_context()

        await mock.execute({}, context)
        await mock.execute({}, context)

        assert_agent_called(mock, at_least=1)
        assert_agent_called(mock, at_least=2)

        with pytest.raises(TestAssertionError):
            assert_agent_called(mock, at_least=3)

    @pytest.mark.asyncio
    async def test_assert_called_at_most(self):
        """Test asserting agent called at most N times."""
        mock = MockAgent("test", return_value=42)
        context = create_test_context()

        await mock.execute({}, context)

        assert_agent_called(mock, at_most=1)
        assert_agent_called(mock, at_most=2)

        with pytest.raises(TestAssertionError):
            assert_agent_called(mock, at_most=0)

    @pytest.mark.asyncio
    async def test_assert_called_no_params_checks_at_least_once(self):
        """Test that no parameters checks for at least one call."""
        mock = MockAgent("test", return_value=42)
        context = create_test_context()

        # Should fail when not called
        with pytest.raises(TestAssertionError, match="never called"):
            assert_agent_called(mock)

        # Should pass when called
        await mock.execute({}, context)
        assert_agent_called(mock)

    @pytest.mark.asyncio
    async def test_assert_works_with_spy(self):
        """Test that assertion works with AgentSpy."""
        mock = MockAgent("real", return_value=42)
        spy = AgentSpy(mock)
        context = create_test_context()

        await spy.execute({}, context)

        assert_agent_called(spy, times=1)

    def test_assert_raises_on_wrong_type(self):
        """Test that assertion raises on wrong agent type."""
        with pytest.raises(ValueError, match="MockAgent or AgentSpy"):
            assert_agent_called("not_an_agent", times=1)


class TestAssertEventEmitted:
    """Test assert_event_emitted function."""

    @pytest.mark.asyncio
    async def test_assert_event_emitted_once(self):
        """Test asserting event emitted exactly once."""
        emitter = create_test_emitter(capture_events=True)

        await emitter.emit("test_event", {"data": "value"})

        assert_event_emitted(emitter, "test_event", times=1)

    @pytest.mark.asyncio
    async def test_assert_event_emitted_fails_wrong_count(self):
        """Test that assertion fails with wrong event count."""
        emitter = create_test_emitter(capture_events=True)

        await emitter.emit("test_event", {})
        await emitter.emit("test_event", {})

        with pytest.raises(TestAssertionError, match="2 time"):
            assert_event_emitted(emitter, "test_event", times=1)

    @pytest.mark.asyncio
    async def test_assert_event_emitted_fails_not_found(self):
        """Test that assertion fails when event not found."""
        emitter = create_test_emitter(capture_events=True)

        await emitter.emit("other_event", {})

        with pytest.raises(TestAssertionError, match="not found"):
            assert_event_emitted(emitter, "test_event")

    @pytest.mark.asyncio
    async def test_assert_event_with_data(self):
        """Test asserting event with specific data."""
        emitter = create_test_emitter(capture_events=True)

        await emitter.emit("test", {"key": "value", "num": 42})

        assert_event_emitted(emitter, "test", with_data={"key": "value"})
        assert_event_emitted(emitter, "test", with_data={"num": 42})

    @pytest.mark.asyncio
    async def test_assert_event_fails_wrong_data(self):
        """Test that assertion fails with wrong data."""
        emitter = create_test_emitter(capture_events=True)

        await emitter.emit("test", {"key": "value"})

        with pytest.raises(TestAssertionError):
            assert_event_emitted(emitter, "test", with_data={"key": "other"})

    @pytest.mark.asyncio
    async def test_assert_event_no_count_checks_at_least_once(self):
        """Test that no count parameter checks for at least one emission."""
        emitter = create_test_emitter(capture_events=True)

        with pytest.raises(TestAssertionError):
            assert_event_emitted(emitter, "test")

        await emitter.emit("test", {})
        assert_event_emitted(emitter, "test")

    def test_assert_raises_without_capture(self):
        """Test that assertion raises if capture not enabled."""
        emitter = create_test_emitter(capture_events=False)

        with pytest.raises(AttributeError, match="event capture"):
            assert_event_emitted(emitter, "test")

    @pytest.mark.asyncio
    async def test_assert_event_nested_data(self):
        """Test asserting event with nested data."""
        emitter = create_test_emitter(capture_events=True)

        await emitter.emit("test", {
            "user": {"id": 123, "name": "Alice"},
            "status": "active"
        })

        assert_event_emitted(emitter, "test", with_data={
            "user": {"id": 123}
        })

class TestAssertContextHas:
    """Test assert_context_has function."""

    def test_assert_context_has_key(self):
        """Test asserting context has a key."""
        context = create_test_context({"key": "value"})

        assert_context_has(context, "key")

    def test_assert_context_has_key_fails(self):
        """Test that assertion fails when key not found."""
        context = create_test_context()

        with pytest.raises(TestAssertionError, match="not found"):
            assert_context_has(context, "missing_key")

    def test_assert_context_has_value(self):
        """Test asserting context has specific value."""
        context = create_test_context({"key": "value"})

        assert_context_has(context, "key", "value")

    def test_assert_context_has_value_fails(self):
        """Test that assertion fails with wrong value."""
        context = create_test_context({"key": "value"})

        with pytest.raises(TestAssertionError, match="value"):
            assert_context_has(context, "key", "other")

    def test_assert_context_has_complex_value(self):
        """Test asserting context with complex values."""
        context = create_test_context({
            "user": {"id": 123, "name": "Alice"}
        })

        assert_context_has(context, "user", {"id": 123, "name": "Alice"})

    def test_assert_context_has_none_value(self):
        """Test asserting context has None value."""
        context = create_test_context({"key": None})

        assert_context_has(context, "key")
        assert_context_has(context, "key", None)

    def test_assert_context_has_numeric_value(self):
        """Test asserting context with numeric values."""
        context = create_test_context({"count": 42, "price": 19.99})

        assert_context_has(context, "count", 42)
        assert_context_has(context, "price", 19.99)
