"""Tests for mock agents and spies."""

import pytest
from agent_lib.testing import MockAgent, AgentSpy, create_test_context


class TestMockAgent:
    """Test MockAgent functionality."""

    @pytest.mark.asyncio
    async def test_mock_returns_configured_value(self):
        """Test that mock returns the configured value."""
        mock = MockAgent(name="test", return_value={"result": "success"})
        context = create_test_context()

        result = await mock.execute({"input": "data"}, context)

        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_mock_raises_side_effect(self):
        """Test that mock raises configured exception."""
        mock = MockAgent(name="test", side_effect=ValueError("test error"))
        context = create_test_context()

        with pytest.raises(ValueError, match="test error"):
            await mock.execute({"input": "data"}, context)

    @pytest.mark.asyncio
    async def test_mock_tracks_calls(self):
        """Test that mock tracks number of calls."""
        mock = MockAgent(name="test", return_value=42)
        context = create_test_context()

        assert mock.call_count == 0

        await mock.execute({"input": "data"}, context)
        assert mock.call_count == 1

        await mock.execute({"input": "more"}, context)
        assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_mock_stores_call_history(self):
        """Test that mock stores call history."""
        mock = MockAgent(name="test", return_value=42)
        context = create_test_context()

        await mock.execute({"first": 1}, context)
        await mock.execute({"second": 2}, context)

        calls = mock.calls
        assert len(calls) == 2
        assert calls[0][0] == {"first": 1}
        assert calls[1][0] == {"second": 2}

    @pytest.mark.asyncio
    async def test_mock_last_call(self):
        """Test accessing last call."""
        mock = MockAgent(name="test", return_value=42)
        context = create_test_context()

        assert mock.last_call is None

        await mock.execute({"first": 1}, context)
        await mock.execute({"second": 2}, context)

        last_input, last_context = mock.last_call
        assert last_input == {"second": 2}
        assert last_context is context

    @pytest.mark.asyncio
    async def test_mock_reset(self):
        """Test resetting mock tracking."""
        mock = MockAgent(name="test", return_value=42)
        context = create_test_context()

        await mock.execute({"input": "data"}, context)
        assert mock.call_count == 1

        mock.reset()
        assert mock.call_count == 0
        assert len(mock.calls) == 0

    @pytest.mark.asyncio
    async def test_mock_reconfigure(self):
        """Test reconfiguring mock behavior."""
        mock = MockAgent(name="test", return_value=1)
        context = create_test_context()

        result = await mock.execute({}, context)
        assert result == 1

        mock.configure(return_value=2)
        result = await mock.execute({}, context)
        assert result == 2

        mock.configure(side_effect=ValueError("error"))
        with pytest.raises(ValueError):
            await mock.execute({}, context)

    @pytest.mark.asyncio
    async def test_mock_with_delay(self):
        """Test mock with simulated delay."""
        import time

        mock = MockAgent(name="test", return_value=42, delay=0.1)
        context = create_test_context()

        start = time.time()
        await mock.execute({}, context)
        elapsed = time.time() - start

        assert elapsed >= 0.1

    @pytest.mark.asyncio
    async def test_mock_returns_none_by_default(self):
        """Test that mock returns None if no return value configured."""
        mock = MockAgent(name="test")
        context = create_test_context()

        result = await mock.execute({}, context)
        assert result is None


class TestAgentSpy:
    """Test AgentSpy functionality."""

    @pytest.mark.asyncio
    async def test_spy_wraps_agent(self):
        """Test that spy wraps and calls real agent."""
        mock = MockAgent(name="real", return_value={"result": 42})
        spy = AgentSpy(mock)
        context = create_test_context()

        result = await spy.execute({"input": "data"}, context)

        assert result == {"result": 42}
        assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_spy_tracks_calls(self):
        """Test that spy tracks calls."""
        mock = MockAgent(name="real", return_value=42)
        spy = AgentSpy(mock)
        context = create_test_context()

        assert spy.call_count == 0

        await spy.execute({"first": 1}, context)
        assert spy.call_count == 1

        await spy.execute({"second": 2}, context)
        assert spy.call_count == 2

    @pytest.mark.asyncio
    async def test_spy_stores_call_history(self):
        """Test that spy stores full call history."""
        mock = MockAgent(name="real", return_value=42)
        spy = AgentSpy(mock)
        context = create_test_context()

        await spy.execute({"first": 1}, context)
        await spy.execute({"second": 2}, context)

        calls = spy.calls
        assert len(calls) == 2
        assert calls[0][0] == {"first": 1}
        assert calls[0][2] == 42  # result
        assert calls[1][0] == {"second": 2}
        assert calls[1][2] == 42  # result

    @pytest.mark.asyncio
    async def test_spy_tracks_errors(self):
        """Test that spy tracks errors."""
        mock = MockAgent(name="real", side_effect=ValueError("test"))
        spy = AgentSpy(mock)
        context = create_test_context()

        with pytest.raises(ValueError):
            await spy.execute({"input": "data"}, context)

        assert spy.call_count == 1
        assert len(spy.errors) == 1
        assert len(spy.calls) == 0  # No successful calls

        error_input, error_context, error = spy.errors[0]
        assert error_input == {"input": "data"}
        assert isinstance(error, ValueError)

    @pytest.mark.asyncio
    async def test_spy_last_call(self):
        """Test accessing last successful call."""
        mock = MockAgent(name="real", return_value=42)
        spy = AgentSpy(mock)
        context = create_test_context()

        assert spy.last_call is None

        await spy.execute({"first": 1}, context)
        await spy.execute({"second": 2}, context)

        last_input, last_context, last_result = spy.last_call
        assert last_input == {"second": 2}
        assert last_result == 42

    @pytest.mark.asyncio
    async def test_spy_last_input_output(self):
        """Test convenience properties for last input/output."""
        mock = MockAgent(name="real", return_value=42)
        spy = AgentSpy(mock)
        context = create_test_context()

        assert spy.last_input is None
        assert spy.last_output is None

        await spy.execute({"input": "data"}, context)

        assert spy.last_input == {"input": "data"}
        assert spy.last_output == 42

    @pytest.mark.asyncio
    async def test_spy_wrapped_agent(self):
        """Test accessing wrapped agent."""
        mock = MockAgent(name="real", return_value=42)
        spy = AgentSpy(mock)

        assert spy.wrapped_agent is mock

    @pytest.mark.asyncio
    async def test_spy_reset(self):
        """Test resetting spy tracking."""
        mock = MockAgent(name="real", return_value=42)
        spy = AgentSpy(mock)
        context = create_test_context()

        await spy.execute({"input": "data"}, context)
        assert spy.call_count == 1

        spy.reset()
        assert spy.call_count == 0
        assert len(spy.calls) == 0
        assert len(spy.errors) == 0

    @pytest.mark.asyncio
    async def test_spy_name_includes_wrapped_agent(self):
        """Test that spy name includes wrapped agent name."""
        mock = MockAgent(name="my_agent", return_value=42)
        spy = AgentSpy(mock)

        assert "my_agent" in spy.name
        assert "spy" in spy.name.lower()
