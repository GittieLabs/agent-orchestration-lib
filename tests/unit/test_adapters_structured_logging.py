"""Tests for StructuredLogAdapter."""

import pytest
from typing import List, Dict, Any
from agent_lib.events.adapters import StructuredLogAdapter
from agent_lib.core.event_emitter import EventEmitter


class MockLogger:
    """Mock logger for testing."""

    def __init__(self):
        self.logs: List[Dict[str, Any]] = []

    def info(self, message: str, **kwargs):
        self.logs.append({"level": "info", "message": message, "data": kwargs})

    def debug(self, message: str, **kwargs):
        self.logs.append({"level": "debug", "message": message, "data": kwargs})

    def warning(self, message: str, **kwargs):
        self.logs.append({"level": "warning", "message": message, "data": kwargs})

    def error(self, message: str, **kwargs):
        self.logs.append({"level": "error", "message": message, "data": kwargs})

    def critical(self, message: str, **kwargs):
        self.logs.append({"level": "critical", "message": message, "data": kwargs})

    def clear(self):
        self.logs.clear()


@pytest.fixture
def mock_logger():
    """Create mock logger."""
    return MockLogger()


@pytest.fixture
def emitter():
    """Create event emitter."""
    return EventEmitter()


@pytest.fixture
def adapter(mock_logger):
    """Create adapter with mock logger."""
    return StructuredLogAdapter(log_instance=mock_logger)


class TestStructuredLogAdapterInit:
    """Test adapter initialization."""

    def test_default_init(self):
        """Test initialization with defaults."""
        adapter = StructuredLogAdapter()
        assert adapter._logger is not None
        assert adapter._include_events is None
        assert adapter._exclude_events == set()
        assert adapter._format_fn is None
        assert adapter._log_level == "INFO"

    def test_init_with_custom_logger(self, mock_logger):
        """Test initialization with custom logger."""
        adapter = StructuredLogAdapter(log_instance=mock_logger)
        assert adapter._logger is mock_logger

    def test_init_with_include_events(self):
        """Test initialization with include filter."""
        adapter = StructuredLogAdapter(include_events=["agent:start", "agent:complete"])
        assert adapter._include_events == {"agent:start", "agent:complete"}

    def test_init_with_exclude_events(self):
        """Test initialization with exclude filter."""
        adapter = StructuredLogAdapter(exclude_events=["agent:progress"])
        assert adapter._exclude_events == {"agent:progress"}

    def test_init_with_custom_format_fn(self):
        """Test initialization with custom format function."""
        def custom_format(event_type, data):
            return {"custom": True}

        adapter = StructuredLogAdapter(format_fn=custom_format)
        assert adapter._format_fn is custom_format

    def test_init_with_custom_log_level(self):
        """Test initialization with custom log level."""
        adapter = StructuredLogAdapter(log_level="DEBUG")
        assert adapter._log_level == "DEBUG"


class TestAttachDetach:
    """Test attaching and detaching from emitter."""

    def test_attach_to_emitter(self, adapter, emitter):
        """Test attaching to emitter."""
        adapter.attach_to_emitter(emitter)
        assert adapter._emitter is emitter

    def test_detach_from_emitter(self, adapter, emitter):
        """Test detaching from emitter."""
        adapter.attach_to_emitter(emitter)
        adapter.detach_from_emitter()
        assert adapter._emitter is None


class TestEventHandling:
    """Test event handling."""

    @pytest.mark.asyncio
    async def test_handle_agent_start(self, adapter, emitter, mock_logger):
        """Test handling agent:start event."""
        adapter.attach_to_emitter(emitter)

        await emitter.emit("agent:start", {
            "agent_name": "test_agent",
            "input_data": {"key": "value"}
        })

        assert len(mock_logger.logs) == 1
        log = mock_logger.logs[0]
        assert log["level"] == "info"
        assert "Agent Event: agent:start" in log["message"]
        assert log["data"]["event_type"] == "agent:start"
        assert log["data"]["agent"] == "test_agent"
        assert "input_preview" in log["data"]

    @pytest.mark.asyncio
    async def test_handle_agent_progress(self, adapter, emitter, mock_logger):
        """Test handling agent:progress event."""
        adapter.attach_to_emitter(emitter)

        await emitter.emit("agent:progress", {
            "agent_name": "test_agent",
            "progress": 0.5,
            "message": "Processing..."
        })

        assert len(mock_logger.logs) == 1
        log = mock_logger.logs[0]
        assert log["data"]["event_type"] == "agent:progress"
        assert log["data"]["progress"] == 0.5
        assert log["data"]["message"] == "Processing..."

    @pytest.mark.asyncio
    async def test_handle_agent_error(self, adapter, emitter, mock_logger):
        """Test handling agent:error event."""
        adapter.attach_to_emitter(emitter)

        await emitter.emit("agent:error", {
            "agent_name": "test_agent",
            "error": "Something went wrong",
            "error_type": "ValueError"
        })

        assert len(mock_logger.logs) == 1
        log = mock_logger.logs[0]
        assert log["data"]["event_type"] == "agent:error"
        assert log["data"]["error"] == "Something went wrong"
        assert log["data"]["error_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_handle_agent_complete(self, adapter, emitter, mock_logger):
        """Test handling agent:complete event."""
        adapter.attach_to_emitter(emitter)

        await emitter.emit("agent:complete", {
            "agent_name": "test_agent",
            "duration_seconds": 1.5,
            "success": True
        })

        assert len(mock_logger.logs) == 1
        log = mock_logger.logs[0]
        assert log["data"]["event_type"] == "agent:complete"
        assert log["data"]["duration_s"] == 1.5
        assert log["data"]["success"] is True


class TestEventFiltering:
    """Test event filtering."""

    @pytest.mark.asyncio
    async def test_include_events_filter(self, emitter, mock_logger):
        """Test include events filter."""
        adapter = StructuredLogAdapter(
            log_instance=mock_logger,
            include_events=["agent:start", "agent:complete"]
        )
        adapter.attach_to_emitter(emitter)

        await emitter.emit("agent:start", {"agent_name": "test"})
        await emitter.emit("agent:progress", {"agent_name": "test"})
        await emitter.emit("agent:complete", {"agent_name": "test"})

        # Only start and complete should be logged
        assert len(mock_logger.logs) == 2
        assert mock_logger.logs[0]["data"]["event_type"] == "agent:start"
        assert mock_logger.logs[1]["data"]["event_type"] == "agent:complete"

    @pytest.mark.asyncio
    async def test_exclude_events_filter(self, emitter, mock_logger):
        """Test exclude events filter."""
        adapter = StructuredLogAdapter(
            log_instance=mock_logger,
            exclude_events=["agent:progress"]
        )
        adapter.attach_to_emitter(emitter)

        await emitter.emit("agent:start", {"agent_name": "test"})
        await emitter.emit("agent:progress", {"agent_name": "test"})
        await emitter.emit("agent:complete", {"agent_name": "test"})

        # Progress should be excluded
        assert len(mock_logger.logs) == 2
        assert mock_logger.logs[0]["data"]["event_type"] == "agent:start"
        assert mock_logger.logs[1]["data"]["event_type"] == "agent:complete"


class TestCustomFormatting:
    """Test custom formatting."""

    @pytest.mark.asyncio
    async def test_custom_format_function(self, emitter, mock_logger):
        """Test custom format function."""
        def custom_format(event_type: str, data: dict) -> dict:
            return {
                "type": event_type,
                "name": data.get("agent_name"),
                "custom": True
            }

        adapter = StructuredLogAdapter(
            log_instance=mock_logger,
            format_fn=custom_format
        )
        adapter.attach_to_emitter(emitter)

        await emitter.emit("agent:start", {"agent_name": "test"})

        assert len(mock_logger.logs) == 1
        log = mock_logger.logs[0]
        assert log["data"]["type"] == "agent:start"
        assert log["data"]["name"] == "test"
        assert log["data"]["custom"] is True


class TestLogLevels:
    """Test log levels."""

    @pytest.mark.asyncio
    async def test_debug_log_level(self, emitter, mock_logger):
        """Test DEBUG log level."""
        adapter = StructuredLogAdapter(
            log_instance=mock_logger,
            log_level="DEBUG"
        )
        adapter.attach_to_emitter(emitter)

        await emitter.emit("agent:start", {"agent_name": "test"})

        assert mock_logger.logs[0]["level"] == "debug"

    @pytest.mark.asyncio
    async def test_error_log_level(self, emitter, mock_logger):
        """Test ERROR log level."""
        adapter = StructuredLogAdapter(
            log_instance=mock_logger,
            log_level="ERROR"
        )
        adapter.attach_to_emitter(emitter)

        await emitter.emit("agent:start", {"agent_name": "test"})

        assert mock_logger.logs[0]["level"] == "error"


class TestConfigurationMethods:
    """Test configuration methods."""

    def test_set_log_level(self, adapter):
        """Test setting log level."""
        adapter.set_log_level("DEBUG")
        assert adapter._log_level == "DEBUG"

        adapter.set_log_level("error")
        assert adapter._log_level == "ERROR"

    def test_set_format_function(self, adapter):
        """Test setting format function."""
        def new_format(event_type, data):
            return {"new": True}

        adapter.set_format_function(new_format)
        assert adapter._format_fn is new_format


class TestDefaultFormatting:
    """Test default formatting logic."""

    def test_format_with_execution_id(self, adapter):
        """Test formatting with execution_id."""
        result = adapter._default_format("agent:start", {
            "agent_name": "test",
            "execution_id": "exec-123",
            "timestamp": "2025-01-23T10:00:00Z"
        })

        assert result["agent"] == "test"
        assert result["execution_id"] == "exec-123"
        assert result["timestamp"] == "2025-01-23T10:00:00Z"

    def test_format_start_with_input_data(self, adapter):
        """Test formatting start event with input data."""
        long_input = "x" * 200
        result = adapter._default_format("agent:start", {
            "agent_name": "test",
            "input_data": long_input
        })

        # Should truncate to 100 chars
        assert len(result["input_preview"]) == 100

    def test_format_progress(self, adapter):
        """Test formatting progress event."""
        result = adapter._default_format("agent:progress", {
            "agent_name": "test",
            "progress": 0.75,
            "message": "Almost done"
        })

        assert result["progress"] == 0.75
        assert result["message"] == "Almost done"

    def test_format_error(self, adapter):
        """Test formatting error event."""
        result = adapter._default_format("agent:error", {
            "agent_name": "test",
            "error": ValueError("test error"),
            "error_type": "ValueError"
        })

        assert "ValueError" in result["error"]
        assert result["error_type"] == "ValueError"

    def test_format_complete(self, adapter):
        """Test formatting complete event."""
        result = adapter._default_format("agent:complete", {
            "agent_name": "test",
            "duration_seconds": 2.5,
            "success": False
        })

        assert result["duration_s"] == 2.5
        assert result["success"] is False


class TestIntegration:
    """Integration tests."""

    @pytest.mark.asyncio
    async def test_full_agent_lifecycle(self, emitter, mock_logger):
        """Test logging full agent lifecycle."""
        adapter = StructuredLogAdapter(log_instance=mock_logger)
        adapter.attach_to_emitter(emitter)

        # Simulate agent lifecycle
        await emitter.emit("agent:start", {
            "agent_name": "test_agent",
            "input_data": {"task": "process"}
        })

        await emitter.emit("agent:progress", {
            "agent_name": "test_agent",
            "progress": 0.5,
            "message": "Halfway done"
        })

        await emitter.emit("agent:complete", {
            "agent_name": "test_agent",
            "duration_seconds": 1.0,
            "success": True
        })

        # Should have 3 log entries
        assert len(mock_logger.logs) == 3
        assert mock_logger.logs[0]["data"]["event_type"] == "agent:start"
        assert mock_logger.logs[1]["data"]["event_type"] == "agent:progress"
        assert mock_logger.logs[2]["data"]["event_type"] == "agent:complete"
