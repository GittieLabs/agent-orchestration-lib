"""Webhook adapter for agent events."""

from typing import Optional, List, Dict, Any, Callable
import asyncio
import json
from datetime import datetime

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from ...core.event_emitter import EventEmitter


class WebhookAdapter:
    """Send agent events to webhooks.

    This adapter forwards agent events to HTTP webhooks, allowing external
    systems to be notified of agent execution progress.

    Example:
        ```python
        from agent_lib.core import EventEmitter
        from agent_lib.events.adapters import WebhookAdapter

        emitter = EventEmitter()

        # Create webhook adapter
        adapter = WebhookAdapter(
            url="https://example.com/webhooks/agents",
            headers={"Authorization": "Bearer token123"},
            event_types=["agent:complete", "agent:error"]
        )
        adapter.attach_to_emitter(emitter)

        # Events will now be posted to the webhook
        await emitter.emit("agent:complete", {
            "agent_name": "test",
            "success": True
        })
        # POST https://example.com/webhooks/agents
        # {"event_type": "agent:complete", "data": {"agent_name": "test", "success": true}}
        ```
    """

    def __init__(
        self,
        url: str,
        event_types: Optional[List[str]] = None,
        headers: Optional[Dict[str, str]] = None,
        transform_fn: Optional[Callable[[str, dict], dict]] = None,
        timeout: float = 10.0,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize webhook adapter.

        Args:
            url: Webhook URL to POST events to
            event_types: Only send these event types (default: all)
            headers: HTTP headers to include in requests
            transform_fn: Custom function to transform event data before sending
            timeout: Request timeout in seconds (default: 10.0)
            retry_count: Number of retries on failure (default: 3)
            retry_delay: Delay between retries in seconds (default: 1.0)
        """
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for WebhookAdapter. "
                "Install with: pip install httpx"
            )

        self._url = url
        self._event_types = set(event_types) if event_types else None
        self._headers = headers or {}
        self._transform_fn = transform_fn
        self._timeout = timeout
        self._retry_count = retry_count
        self._retry_delay = retry_delay
        self._emitter: Optional[EventEmitter] = None

        # Set default headers
        if "Content-Type" not in self._headers:
            self._headers["Content-Type"] = "application/json"

        # Create HTTP client
        self._client = httpx.AsyncClient(timeout=timeout)

    def attach_to_emitter(self, emitter: EventEmitter) -> None:
        """Subscribe to emitter and forward events to webhook.

        Args:
            emitter: EventEmitter to subscribe to
        """
        self._emitter = emitter

        # Subscribe to all event types
        event_types = [
            "agent:start",
            "agent:progress",
            "agent:error",
            "agent:complete",
        ]

        for event_type in event_types:
            emitter.on(event_type, lambda e, et=event_type: asyncio.create_task(
                self._handle_event(et, e)
            ))

    def detach_from_emitter(self) -> None:
        """Unsubscribe from emitter."""
        if self._emitter:
            self._emitter = None

    async def close(self) -> None:
        """Close the HTTP client.

        Should be called when done using the adapter.
        """
        await self._client.aclose()

    async def _handle_event(self, event_type: str, event_data: dict) -> None:
        """Handle an event by posting to webhook.

        Args:
            event_type: Type of event
            event_data: Event data dictionary
        """
        # Check if we should send this event type
        if self._event_types and event_type not in self._event_types:
            return

        # Transform data if custom function provided
        if self._transform_fn:
            payload = self._transform_fn(event_type, event_data)
        else:
            payload = self._default_transform(event_type, event_data)

        # Send to webhook with retries
        await self._send_with_retry(payload)

    def _default_transform(self, event_type: str, event_data: dict) -> dict:
        """Default transformation of event data to webhook payload.

        Args:
            event_type: Type of event
            event_data: Event data dictionary

        Returns:
            Payload dictionary to send to webhook
        """
        return {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data": event_data
        }

    async def _send_with_retry(self, payload: dict) -> None:
        """Send payload to webhook with retry logic.

        Args:
            payload: Data to send
        """
        last_exception = None

        for attempt in range(self._retry_count):
            try:
                response = await self._client.post(
                    self._url,
                    json=payload,
                    headers=self._headers
                )

                # Check if successful
                response.raise_for_status()

                # Success - return
                return

            except (httpx.HTTPError, httpx.TimeoutException) as e:
                last_exception = e

                # If this isn't the last attempt, wait before retrying
                if attempt < self._retry_count - 1:
                    await asyncio.sleep(self._retry_delay * (attempt + 1))

        # All retries failed - log error but don't raise
        # (We don't want webhook failures to break agent execution)
        if last_exception:
            import logging
            logging.error(
                f"Failed to send webhook after {self._retry_count} attempts: {last_exception}"
            )

    def set_url(self, url: str) -> None:
        """Update webhook URL.

        Args:
            url: New webhook URL
        """
        self._url = url

    def set_headers(self, headers: Dict[str, str]) -> None:
        """Update HTTP headers.

        Args:
            headers: Dictionary of headers to set/update
        """
        self._headers.update(headers)

    def set_event_types(self, event_types: List[str]) -> None:
        """Update which event types to forward.

        Args:
            event_types: List of event types to forward
        """
        self._event_types = set(event_types)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - closes HTTP client."""
        await self.close()
