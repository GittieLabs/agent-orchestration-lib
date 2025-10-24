"""Unit tests for Event models."""

import pytest
from datetime import datetime, timezone
from agent_lib.events import (
    Event,
    StartEvent,
    ProgressEvent,
    ErrorEvent,
    CompletionEvent,
    RetryEvent,
    MetricEvent
)


class TestEventBase:
    """Test base Event model."""

    def test_create_event(self):
        """Test creating a basic event."""
        event = Event(type="test", source="test_agent")

        assert event.type == "test"
        assert event.source == "test_agent"
        assert event.event_id is not None
        assert event.timestamp is not None
        assert isinstance(event.data, dict)

    def test_event_auto_fields(self):
        """Test that event_id and timestamp are auto-populated."""
        event1 = Event(type="test", source="agent1")
        event2 = Event(type="test", source="agent2")

        # Each event should have unique ID
        assert event1.event_id != event2.event_id

        # Timestamp should be recent
        now = datetime.now(timezone.utc)
        assert event1.timestamp <= now

    def test_event_with_data(self):
        """Test event with custom data."""
        event = Event(
            type="test",
            source="agent",
            data={"key": "value", "count": 42}
        )

        assert event.data["key"] == "value"
        assert event.data["count"] == 42


class TestStartEvent:
    """Test StartEvent model."""

    def test_create_start_event(self):
        """Test creating a start event."""
        event = StartEvent(source="test_agent")

        assert event.type == "start"
        assert event.source == "test_agent"

    def test_start_event_with_input_summary(self):
        """Test start event with input summary."""
        event = StartEvent(
            source="test_agent",
            input_summary="Processing 10 items"
        )

        assert event.input_summary == "Processing 10 items"


class TestProgressEvent:
    """Test ProgressEvent model."""

    def test_create_progress_event(self):
        """Test creating a progress event."""
        event = ProgressEvent(
            source="test_agent",
            stage="processing",
            progress=0.5,
            message="Halfway done"
        )

        assert event.type == "progress"
        assert event.stage == "processing"
        assert event.progress == 0.5
        assert event.message == "Halfway done"

    def test_progress_event_validation(self):
        """Test progress event validation."""
        # Valid progress values
        event = ProgressEvent(
            source="agent",
            stage="test",
            progress=0.0,
            message="Start"
        )
        assert event.progress == 0.0

        event = ProgressEvent(
            source="agent",
            stage="test",
            progress=1.0,
            message="Complete"
        )
        assert event.progress == 1.0

        # Invalid progress should raise
        with pytest.raises(Exception):  # Pydantic ValidationError
            ProgressEvent(
                source="agent",
                stage="test",
                progress=1.5,  # > 1.0
                message="Invalid"
            )

        with pytest.raises(Exception):  # Pydantic ValidationError
            ProgressEvent(
                source="agent",
                stage="test",
                progress=-0.1,  # < 0.0
                message="Invalid"
            )

    def test_progress_event_with_details(self):
        """Test progress event with details."""
        event = ProgressEvent(
            source="agent",
            stage="processing",
            progress=0.75,
            message="Almost done",
            details={"items_processed": 75, "items_total": 100}
        )

        assert event.details["items_processed"] == 75
        assert event.details["items_total"] == 100


class TestErrorEvent:
    """Test ErrorEvent model."""

    def test_create_error_event(self):
        """Test creating an error event."""
        event = ErrorEvent(
            source="test_agent",
            error_type="ValueError",
            error_message="Invalid input"
        )

        assert event.type == "error"
        assert event.error_type == "ValueError"
        assert event.error_message == "Invalid input"
        assert event.recoverable is False  # Default

    def test_error_event_with_stack_trace(self):
        """Test error event with stack trace."""
        event = ErrorEvent(
            source="agent",
            error_type="RuntimeError",
            error_message="Something went wrong",
            stack_trace="Traceback...\nLine 1\nLine 2",
            recoverable=True
        )

        assert event.stack_trace is not None
        assert "Traceback" in event.stack_trace
        assert event.recoverable is True

    def test_error_event_with_attempt(self):
        """Test error event with retry attempt number."""
        event = ErrorEvent(
            source="agent",
            error_type="TimeoutError",
            error_message="Request timed out",
            recoverable=True,
            attempt=2
        )

        assert event.attempt == 2


class TestCompletionEvent:
    """Test CompletionEvent model."""

    def test_create_completion_event(self):
        """Test creating a completion event."""
        event = CompletionEvent(
            source="test_agent",
            duration_seconds=2.5
        )

        assert event.type == "completion"
        assert event.duration_seconds == 2.5
        assert event.success is True  # Default

    def test_completion_event_with_summary(self):
        """Test completion event with output summary."""
        event = CompletionEvent(
            source="agent",
            duration_seconds=1.5,
            output_summary="Processed 100 items successfully"
        )

        assert event.output_summary == "Processed 100 items successfully"

    def test_completion_event_with_metrics(self):
        """Test completion event with metrics."""
        event = CompletionEvent(
            source="agent",
            duration_seconds=3.7,
            metrics={
                "tokens_used": 1500,
                "api_calls": 3,
                "items_processed": 50
            }
        )

        assert event.metrics["tokens_used"] == 1500
        assert event.metrics["api_calls"] == 3


class TestRetryEvent:
    """Test RetryEvent model."""

    def test_create_retry_event(self):
        """Test creating a retry event."""
        event = RetryEvent(
            source="test_agent",
            attempt=2,
            max_attempts=3,
            reason="Rate limit exceeded",
            wait_seconds=4.0
        )

        assert event.type == "retry"
        assert event.attempt == 2
        assert event.max_attempts == 3
        assert event.reason == "Rate limit exceeded"
        assert event.wait_seconds == 4.0

    def test_retry_event_with_strategy(self):
        """Test retry event with strategy name."""
        event = RetryEvent(
            source="agent",
            attempt=1,
            max_attempts=5,
            reason="Connection failed",
            wait_seconds=2.0,
            strategy="exponential_backoff"
        )

        assert event.strategy == "exponential_backoff"


class TestMetricEvent:
    """Test MetricEvent model."""

    def test_create_metric_event(self):
        """Test creating a metric event."""
        event = MetricEvent(
            source="test_agent",
            metric_name="tokens_used",
            metric_value=1500.0
        )

        assert event.type == "metric"
        assert event.metric_name == "tokens_used"
        assert event.metric_value == 1500.0

    def test_metric_event_with_unit(self):
        """Test metric event with unit."""
        event = MetricEvent(
            source="agent",
            metric_name="duration",
            metric_value=2.5,
            unit="seconds"
        )

        assert event.unit == "seconds"

    def test_metric_event_with_tags(self):
        """Test metric event with tags."""
        event = MetricEvent(
            source="agent",
            metric_name="api_calls",
            metric_value=5.0,
            tags={
                "endpoint": "completions",
                "model": "gpt-4",
                "user_id": "123"
            }
        )

        assert event.tags["endpoint"] == "completions"
        assert event.tags["model"] == "gpt-4"
        assert event.tags["user_id"] == "123"


class TestEventSerialization:
    """Test event serialization."""

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        event = ProgressEvent(
            source="agent",
            stage="processing",
            progress=0.5,
            message="Halfway"
        )

        data = event.model_dump()

        assert data["type"] == "progress"
        assert data["source"] == "agent"
        assert data["stage"] == "processing"
        assert data["progress"] == 0.5
        assert "event_id" in data
        assert "timestamp" in data

    def test_event_to_json(self):
        """Test converting event to JSON."""
        event = CompletionEvent(
            source="agent",
            duration_seconds=1.5
        )

        json_str = event.model_dump_json()

        assert '"type":"completion"' in json_str
        assert '"source":"agent"' in json_str
        assert "event_id" in json_str
        assert "timestamp" in json_str


class TestEventUsagePatterns:
    """Test common event usage patterns."""

    def test_agent_lifecycle_events(self):
        """Test typical agent lifecycle event sequence."""
        # Start
        start = StartEvent(
            source="parsing_agent",
            input_summary="Resume for John Doe"
        )
        assert start.type == "start"

        # Progress updates
        progress1 = ProgressEvent(
            source="parsing_agent",
            stage="initialization",
            progress=0.0,
            message="Initializing..."
        )

        progress2 = ProgressEvent(
            source="parsing_agent",
            stage="processing",
            progress=0.5,
            message="Processing..."
        )

        # Completion
        completion = CompletionEvent(
            source="parsing_agent",
            duration_seconds=2.5,
            output_summary="Extracted 15 work experiences"
        )

        assert completion.success is True

    def test_error_and_retry_pattern(self):
        """Test error and retry event pattern."""
        # Error occurs
        error = ErrorEvent(
            source="agent",
            error_type="RateLimitError",
            error_message="API rate limit exceeded",
            recoverable=True,
            attempt=1
        )

        # Retry attempted
        retry = RetryEvent(
            source="agent",
            attempt=2,
            max_attempts=3,
            reason="RateLimitError on attempt 1",
            wait_seconds=2.0,
            strategy="exponential_backoff"
        )

        assert error.recoverable is True
        assert retry.attempt == 2

    def test_metric_tracking_pattern(self):
        """Test metric tracking event pattern."""
        metrics = [
            MetricEvent(
                source="agent",
                metric_name="tokens_input",
                metric_value=500.0,
                unit="tokens",
                tags={"model": "gpt-4"}
            ),
            MetricEvent(
                source="agent",
                metric_name="tokens_output",
                metric_value=1000.0,
                unit="tokens",
                tags={"model": "gpt-4"}
            ),
            MetricEvent(
                source="agent",
                metric_name="duration",
                metric_value=2.5,
                unit="seconds"
            )
        ]

        assert len(metrics) == 3
        assert all(m.type == "metric" for m in metrics)
