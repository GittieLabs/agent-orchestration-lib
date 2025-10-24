"""Structured logging adapter for agent events."""

from typing import Optional, List, Callable, Any, Dict
from loguru import logger

from ..event_models import (
    StartEvent,
    ProgressEvent,
    ErrorEvent,
    CompletionEvent,
    Event,
)
from ...core.event_emitter import EventEmitter


class StructuredLogAdapter:
    """Adapter to send agent events to structured logging.

    This adapter subscribes to an EventEmitter and forwards events to a
    structured logger (like loguru or structlog), with optional filtering
    and formatting.

    Example:
        ```python
        from agent_lib.core import EventEmitter
        from agent_lib.events.adapters import StructuredLogAdapter
        from agent_lib.events.event_models import StartEvent

        emitter = EventEmitter()

        # Create adapter with default logger
        adapter = StructuredLogAdapter()
        adapter.attach_to_emitter(emitter)

        # Events will now be logged
        await emitter.emit(StartEvent(
            source="test_agent",
            stage="init"
        ))
        # Logs: [INFO] Agent Event: start - source=test_agent stage=init

        # With custom formatting
        def custom_format(event: Event) -> dict:
            return {
                "event": event.type,
                "timestamp": str(event.timestamp),
                "source": event.source
            }

        adapter = StructuredLogAdapter(format_fn=custom_format)
        ```
    """

    def __init__(
        self,
        log_instance: Any = None,
        include_events: Optional[List[str]] = None,
        exclude_events: Optional[List[str]] = None,
        format_fn: Optional[Callable[[Event], dict]] = None,
        log_level: str = "INFO",
    ):
        """Initialize structured logging adapter.

        Args:
            log_instance: Logger instance (default: loguru logger)
            include_events: Only log these event types (default: all)
            exclude_events: Never log these event types (default: none)
            format_fn: Custom function to format event data for logging
            log_level: Log level to use (default: "INFO")
        """
        self._logger = log_instance or logger
        self._include_events = set(include_events) if include_events else None
        self._exclude_events = set(exclude_events) if exclude_events else set()
        self._format_fn = format_fn
        self._log_level = log_level.upper()
        self._emitter: Optional[EventEmitter] = None

    def attach_to_emitter(self, emitter: EventEmitter) -> None:
        """Subscribe to emitter and forward events to logger.

        Args:
            emitter: EventEmitter to subscribe to
        """
        self._emitter = emitter

        # Subscribe to all event types
        event_types = [
            "start",
            "progress",
            "error",
            "complete",
        ]

        for event_type in event_types:
            emitter.subscribe(event_type, self._create_handler())

    def _create_handler(self):
        """Create async event handler."""
        async def handler(event: Event):
            await self._handle_event(event)
        return handler

    def detach_from_emitter(self) -> None:
        """Unsubscribe from emitter."""
        if self._emitter:
            self._emitter = None

    async def _handle_event(self, event: Event) -> None:
        """Handle an event by logging it.

        Args:
            event: Event instance
        """
        # Check filters
        if self._include_events and event.type not in self._include_events:
            return

        if event.type in self._exclude_events:
            return

        # Format data
        if self._format_fn:
            log_data = self._format_fn(event)
        else:
            log_data = self._default_format(event)

        # Log at appropriate level
        log_method = getattr(self._logger, self._log_level.lower(), self._logger.info)

        # Log with structured data
        log_method(
            f"Agent Event: {event.type}",
            **log_data
        )

    def _default_format(self, event: Event) -> dict:
        """Default formatting for event data.

        Args:
            event: Event instance

        Returns:
            Formatted dictionary for structured logging
        """
        formatted = {
            "event_type": event.type,
            "source": event.source,
            "stage": event.stage,
            "timestamp": str(event.timestamp),
        }

        # Event-specific fields
        if isinstance(event, StartEvent):
            if hasattr(event, "input_preview"):
                formatted["input_preview"] = str(event.input_preview)[:100]

        elif isinstance(event, ProgressEvent):
            formatted["progress"] = event.progress
            if event.message:
                formatted["message"] = event.message

        elif isinstance(event, ErrorEvent):
            formatted["error"] = event.error_message
            if event.error_type:
                formatted["error_type"] = event.error_type
            if event.recoverable is not None:
                formatted["recoverable"] = event.recoverable

        elif isinstance(event, CompletionEvent):
            if event.duration_seconds is not None:
                formatted["duration_s"] = event.duration_seconds
            formatted["success"] = event.success

        return formatted

    def set_log_level(self, level: str) -> None:
        """Change the log level.

        Args:
            level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self._log_level = level.upper()

    def set_format_function(self, format_fn: Callable[[Event], dict]) -> None:
        """Set a custom formatting function.

        Args:
            format_fn: Function that takes an Event and returns
                      a dictionary of fields to log
        """
        self._format_fn = format_fn
