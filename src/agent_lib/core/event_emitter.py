"""
EventEmitter - Pub/sub event system for agent notifications.

Provides event-driven architecture with support for:
- Multiple subscribers per event type
- Async event handlers
- Event filtering by type
- Type-safe event handling
"""

from typing import Awaitable, Callable, Dict, List, Optional
from loguru import logger

from ..events.event_models import AnyEvent, Event


# Type alias for event handlers
EventHandler = Callable[[Event], Awaitable[None]]


class EventEmitter:
    """Event-driven notification system with pub/sub pattern.

    Allows agents to emit events and subscribers to listen for specific event types.
    Supports multiple handlers per event type and async event processing.

    Example:
        ```python
        # Create emitter
        emitter = EventEmitter()

        # Subscribe to events
        async def log_progress(event: ProgressEvent):
            print(f"Progress: {event.progress * 100}%")

        emitter.subscribe("progress", log_progress)

        # Emit events
        await emitter.emit(ProgressEvent(
            source="agent",
            stage="processing",
            progress=0.5,
            message="Halfway done"
        ))
        ```
    """

    def __init__(self):
        """Initialize event emitter with empty subscriber registry."""
        self._subscribers: Dict[str, List[EventHandler]] = {}
        logger.debug("EventEmitter initialized")

    def subscribe(
        self,
        event_type: str,
        handler: EventHandler
    ) -> None:
        """Subscribe a handler to a specific event type.

        Args:
            event_type: Type of event to listen for (e.g., "progress", "error")
            handler: Async function to call when event is emitted

        Example:
            ```python
            async def handle_error(event: ErrorEvent):
                logger.error(f"Error in {event.source}: {event.error_message}")

            emitter.subscribe("error", handle_error)
            ```
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        self._subscribers[event_type].append(handler)
        logger.debug(
            f"Subscribed handler to '{event_type}' "
            f"({len(self._subscribers[event_type])} total)"
        )

    def unsubscribe(
        self,
        event_type: str,
        handler: Optional[EventHandler] = None
    ) -> None:
        """Unsubscribe handler(s) from an event type.

        Args:
            event_type: Event type to unsubscribe from
            handler: Specific handler to remove, or None to remove all

        Example:
            ```python
            # Remove specific handler
            emitter.unsubscribe("progress", my_handler)

            # Remove all handlers for event type
            emitter.unsubscribe("progress")
            ```
        """
        if event_type not in self._subscribers:
            return

        if handler is None:
            # Remove all handlers for this event type
            count = len(self._subscribers[event_type])
            del self._subscribers[event_type]
            logger.debug(f"Unsubscribed all {count} handlers from '{event_type}'")
        else:
            # Remove specific handler
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
                logger.debug(f"Unsubscribed handler from '{event_type}'")

    async def emit(self, event: AnyEvent) -> None:
        """Emit an event to all subscribed handlers.

        Calls all handlers subscribed to this event type asynchronously.
        If a handler raises an exception, it is logged but doesn't stop
        other handlers from executing.

        Args:
            event: Event instance to emit

        Example:
            ```python
            await emitter.emit(ProgressEvent(
                source="parsing_agent",
                stage="extraction",
                progress=0.75,
                message="Extracting data..."
            ))
            ```
        """
        event_type = event.type
        handlers = self._subscribers.get(event_type, [])

        if not handlers:
            logger.debug(
                f"No subscribers for event type '{event_type}' from {event.source}"
            )
            return

        logger.debug(
            f"Emitting '{event_type}' event from {event.source} "
            f"to {len(handlers)} handlers"
        )

        # Call all handlers asynchronously
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(
                    f"Error in event handler for '{event_type}': {e}",
                    exc_info=True
                )

    def get_subscriber_count(self, event_type: Optional[str] = None) -> int:
        """Get number of subscribers for an event type or total.

        Args:
            event_type: Specific event type, or None for total count

        Returns:
            Number of subscribers

        Example:
            ```python
            # Get subscribers for specific event
            progress_count = emitter.get_subscriber_count("progress")

            # Get total subscribers
            total_count = emitter.get_subscriber_count()
            ```
        """
        if event_type is None:
            return sum(len(handlers) for handlers in self._subscribers.values())

        return len(self._subscribers.get(event_type, []))

    def clear(self) -> None:
        """Remove all event subscribers.

        Useful for cleanup or testing.

        Example:
            ```python
            emitter.clear()
            ```
        """
        count = self.get_subscriber_count()
        self._subscribers.clear()
        logger.debug(f"Cleared all {count} event subscribers")

    def __repr__(self) -> str:
        """String representation of emitter."""
        event_types = len(self._subscribers)
        total_subscribers = self.get_subscriber_count()
        return f"EventEmitter({event_types} event types, {total_subscribers} subscribers)"
