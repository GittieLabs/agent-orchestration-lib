"""Unit tests for EventEmitter."""

import pytest
from agent_lib.core import EventEmitter
from agent_lib.events import Event, ProgressEvent, ErrorEvent, CompletionEvent


class TestEventEmitterBasics:
    """Test basic EventEmitter functionality."""

    def test_create_emitter(self):
        """Test creating an event emitter."""
        emitter = EventEmitter()
        assert emitter is not None
        assert emitter.get_subscriber_count() == 0

    def test_repr(self):
        """Test string representation."""
        emitter = EventEmitter()
        assert "EventEmitter" in repr(emitter)
        assert "0 event types" in repr(emitter)
        assert "0 subscribers" in repr(emitter)


class TestEventEmitterSubscription:
    """Test subscription functionality."""

    @pytest.mark.asyncio
    async def test_subscribe_to_event(self):
        """Test subscribing to an event type."""
        emitter = EventEmitter()
        events_received = []

        async def handler(event: Event):
            events_received.append(event)

        emitter.subscribe("test", handler)

        assert emitter.get_subscriber_count("test") == 1
        assert emitter.get_subscriber_count() == 1

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self):
        """Test multiple subscribers to same event type."""
        emitter = EventEmitter()
        calls1 = []
        calls2 = []

        async def handler1(event: Event):
            calls1.append(event)

        async def handler2(event: Event):
            calls2.append(event)

        emitter.subscribe("test", handler1)
        emitter.subscribe("test", handler2)

        assert emitter.get_subscriber_count("test") == 2

        # Emit event
        event = Event(type="test", source="test_source")
        await emitter.emit(event)

        # Both handlers should be called
        assert len(calls1) == 1
        assert len(calls2) == 1

    @pytest.mark.asyncio
    async def test_subscribe_to_different_events(self):
        """Test subscribing to different event types."""
        emitter = EventEmitter()

        async def progress_handler(event: Event):
            pass

        async def error_handler(event: Event):
            pass

        emitter.subscribe("progress", progress_handler)
        emitter.subscribe("error", error_handler)

        assert emitter.get_subscriber_count("progress") == 1
        assert emitter.get_subscriber_count("error") == 1
        assert emitter.get_subscriber_count() == 2


class TestEventEmitterUnsubscribe:
    """Test unsubscribe functionality."""

    @pytest.mark.asyncio
    async def test_unsubscribe_specific_handler(self):
        """Test unsubscribing a specific handler."""
        emitter = EventEmitter()

        async def handler1(event: Event):
            pass

        async def handler2(event: Event):
            pass

        emitter.subscribe("test", handler1)
        emitter.subscribe("test", handler2)
        assert emitter.get_subscriber_count("test") == 2

        emitter.unsubscribe("test", handler1)
        assert emitter.get_subscriber_count("test") == 1

    @pytest.mark.asyncio
    async def test_unsubscribe_all_handlers(self):
        """Test unsubscribing all handlers for an event type."""
        emitter = EventEmitter()

        async def handler1(event: Event):
            pass

        async def handler2(event: Event):
            pass

        emitter.subscribe("test", handler1)
        emitter.subscribe("test", handler2)
        assert emitter.get_subscriber_count("test") == 2

        emitter.unsubscribe("test")  # Remove all
        assert emitter.get_subscriber_count("test") == 0

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent(self):
        """Test unsubscribing from nonexistent event type."""
        emitter = EventEmitter()

        # Should not raise error
        emitter.unsubscribe("nonexistent")


class TestEventEmitterEmit:
    """Test event emission."""

    @pytest.mark.asyncio
    async def test_emit_event(self):
        """Test emitting an event."""
        emitter = EventEmitter()
        received_events = []

        async def handler(event: Event):
            received_events.append(event)

        emitter.subscribe("test", handler)

        event = Event(type="test", source="test_agent")
        await emitter.emit(event)

        assert len(received_events) == 1
        assert received_events[0] is event

    @pytest.mark.asyncio
    async def test_emit_to_no_subscribers(self):
        """Test emitting when there are no subscribers."""
        emitter = EventEmitter()

        event = Event(type="test", source="test_agent")
        # Should not raise error
        await emitter.emit(event)

    @pytest.mark.asyncio
    async def test_emit_progress_event(self):
        """Test emitting a ProgressEvent."""
        emitter = EventEmitter()
        received_events = []

        async def handler(event: ProgressEvent):
            received_events.append(event)

        emitter.subscribe("progress", handler)

        event = ProgressEvent(
            source="test_agent",
            stage="processing",
            progress=0.5,
            message="Halfway done"
        )
        await emitter.emit(event)

        assert len(received_events) == 1
        assert received_events[0].progress == 0.5
        assert received_events[0].message == "Halfway done"

    @pytest.mark.asyncio
    async def test_emit_error_event(self):
        """Test emitting an ErrorEvent."""
        emitter = EventEmitter()
        received_events = []

        async def handler(event: ErrorEvent):
            received_events.append(event)

        emitter.subscribe("error", handler)

        event = ErrorEvent(
            source="test_agent",
            error_type="ValueError",
            error_message="Test error",
            recoverable=True
        )
        await emitter.emit(event)

        assert len(received_events) == 1
        assert received_events[0].error_type == "ValueError"
        assert received_events[0].recoverable is True

    @pytest.mark.asyncio
    async def test_handler_exception_isolation(self):
        """Test that one handler's exception doesn't stop other handlers."""
        emitter = EventEmitter()
        handler1_called = []
        handler2_called = []

        async def failing_handler(event: Event):
            handler1_called.append(True)
            raise ValueError("Handler failed")

        async def working_handler(event: Event):
            handler2_called.append(True)

        emitter.subscribe("test", failing_handler)
        emitter.subscribe("test", working_handler)

        event = Event(type="test", source="test_agent")
        await emitter.emit(event)

        # Both handlers should have been called
        assert len(handler1_called) == 1
        assert len(handler2_called) == 1


class TestEventEmitterUtility:
    """Test utility methods."""

    @pytest.mark.asyncio
    async def test_get_subscriber_count_by_type(self):
        """Test getting subscriber count for specific event type."""
        emitter = EventEmitter()

        async def handler(event: Event):
            pass

        emitter.subscribe("progress", handler)
        emitter.subscribe("progress", handler)
        emitter.subscribe("error", handler)

        assert emitter.get_subscriber_count("progress") == 2
        assert emitter.get_subscriber_count("error") == 1
        assert emitter.get_subscriber_count("nonexistent") == 0

    @pytest.mark.asyncio
    async def test_get_total_subscriber_count(self):
        """Test getting total subscriber count."""
        emitter = EventEmitter()

        async def handler(event: Event):
            pass

        emitter.subscribe("progress", handler)
        emitter.subscribe("progress", handler)
        emitter.subscribe("error", handler)

        assert emitter.get_subscriber_count() == 3

    @pytest.mark.asyncio
    async def test_clear_all_subscribers(self):
        """Test clearing all subscribers."""
        emitter = EventEmitter()

        async def handler(event: Event):
            pass

        emitter.subscribe("progress", handler)
        emitter.subscribe("error", handler)
        assert emitter.get_subscriber_count() == 2

        emitter.clear()
        assert emitter.get_subscriber_count() == 0


class TestEventEmitterComplexScenarios:
    """Test complex usage scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_event_types(self):
        """Test emitting multiple event types."""
        emitter = EventEmitter()
        progress_events = []
        error_events = []
        completion_events = []

        async def progress_handler(event: ProgressEvent):
            progress_events.append(event)

        async def error_handler(event: ErrorEvent):
            error_events.append(event)

        async def completion_handler(event: CompletionEvent):
            completion_events.append(event)

        emitter.subscribe("progress", progress_handler)
        emitter.subscribe("error", error_handler)
        emitter.subscribe("completion", completion_handler)

        # Emit different event types
        await emitter.emit(ProgressEvent(
            source="agent",
            stage="start",
            progress=0.0,
            message="Starting"
        ))

        await emitter.emit(ErrorEvent(
            source="agent",
            error_type="TestError",
            error_message="Test"
        ))

        await emitter.emit(CompletionEvent(
            source="agent",
            duration_seconds=1.5
        ))

        # Each handler should only receive its event type
        assert len(progress_events) == 1
        assert len(error_events) == 1
        assert len(completion_events) == 1

    @pytest.mark.asyncio
    async def test_agent_progress_tracking_pattern(self):
        """Test pattern for tracking agent progress."""
        emitter = EventEmitter()
        progress_history = []

        async def track_progress(event: ProgressEvent):
            progress_history.append({
                "stage": event.stage,
                "progress": event.progress,
                "message": event.message
            })

        emitter.subscribe("progress", track_progress)

        # Simulate agent progress
        stages = [
            ("init", 0.0, "Initializing"),
            ("processing", 0.25, "Processing input"),
            ("analyzing", 0.5, "Analyzing data"),
            ("generating", 0.75, "Generating output"),
            ("complete", 1.0, "Complete")
        ]

        for stage, progress, message in stages:
            await emitter.emit(ProgressEvent(
                source="agent",
                stage=stage,
                progress=progress,
                message=message
            ))

        assert len(progress_history) == 5
        assert progress_history[0]["progress"] == 0.0
        assert progress_history[-1]["progress"] == 1.0

    @pytest.mark.asyncio
    async def test_event_data_passing(self):
        """Test passing custom data in events."""
        emitter = EventEmitter()
        received_data = []

        async def handler(event: Event):
            received_data.append(event.data)

        emitter.subscribe("custom", handler)

        event = Event(
            type="custom",
            source="agent",
            data={
                "key1": "value1",
                "key2": 123,
                "nested": {"data": True}
            }
        )

        await emitter.emit(event)

        assert len(received_data) == 1
        assert received_data[0]["key1"] == "value1"
        assert received_data[0]["key2"] == 123
        assert received_data[0]["nested"]["data"] is True
