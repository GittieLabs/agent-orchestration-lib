"""Metrics adapter for agent events."""

from typing import Optional, Dict, Any, Callable
from datetime import datetime
import time

from ..event_models import (
    StartEvent,
    ProgressEvent,
    ErrorEvent,
    CompletionEvent,
)
from ...core.event_emitter import EventEmitter


class MetricsAdapter:
    """Adapter to send metrics to monitoring systems.

    This adapter collects metrics from agent events and sends them to
    monitoring systems like Prometheus, DataDog, CloudWatch, etc.

    Example:
        ```python
        from agent_lib.core import EventEmitter
        from agent_lib.events.adapters import MetricsAdapter

        # Create a simple metrics collector
        class SimpleMetricsCollector:
            def __init__(self):
                self.metrics = []

            def increment(self, name, value=1, tags=None):
                self.metrics.append(("counter", name, value, tags))

            def gauge(self, name, value, tags=None):
                self.metrics.append(("gauge", name, value, tags))

            def histogram(self, name, value, tags=None):
                self.metrics.append(("histogram", name, value, tags))

        collector = SimpleMetricsCollector()
        emitter = EventEmitter()

        adapter = MetricsAdapter(collector, prefix="myapp")
        adapter.attach_to_emitter(emitter)

        # Metrics will now be recorded
        await emitter.emit("agent:start", {"agent_name": "test"})
        # Records: myapp.agent.started (counter)

        await emitter.emit("agent:complete", {
            "agent_name": "test",
            "duration_seconds": 1.5,
            "success": True
        })
        # Records: myapp.agent.completed (counter)
        #          myapp.agent.duration (histogram, 1.5s)
        ```
    """

    def __init__(
        self,
        metrics_client: Any,
        prefix: str = "agent_lib",
        tags: Optional[Dict[str, str]] = None,
        transform_fn: Optional[Callable[[str, dict], Dict[str, Any]]] = None,
    ):
        """Initialize metrics adapter.

        Args:
            metrics_client: Metrics client (Prometheus, DataDog, etc.)
                           Must have methods: increment(), gauge(), histogram()
            prefix: Prefix for all metric names (default: "agent_lib")
            tags: Global tags to add to all metrics (default: None)
            transform_fn: Custom function to transform event data into metrics
        """
        self._client = metrics_client
        self._prefix = prefix
        self._global_tags = tags or {}
        self._transform_fn = transform_fn
        self._emitter: Optional[EventEmitter] = None

        # Track agent execution times for duration metrics
        self._start_times: Dict[str, float] = {}

    def attach_to_emitter(self, emitter: EventEmitter) -> None:
        """Subscribe to emitter and record metrics.

        Args:
            emitter: EventEmitter to subscribe to
        """
        self._emitter = emitter

        # Subscribe to all event types
        emitter.on("agent:start", self._handle_start)
        emitter.on("agent:progress", self._handle_progress)
        emitter.on("agent:error", self._handle_error)
        emitter.on("agent:complete", self._handle_complete)

    def detach_from_emitter(self) -> None:
        """Unsubscribe from emitter."""
        if self._emitter:
            self._emitter = None

    def _handle_start(self, event_data: dict) -> None:
        """Handle agent start event.

        Args:
            event_data: Event data dictionary
        """
        if self._transform_fn:
            metrics = self._transform_fn("agent:start", event_data)
            self._record_metrics(metrics)
        else:
            # Default metrics
            tags = self._build_tags(event_data)

            # Increment started counter
            self._increment("agent.started", tags=tags)

            # Track start time for duration calculation
            agent_name = event_data.get("agent_name")
            if agent_name:
                self._start_times[agent_name] = time.time()

    def _handle_progress(self, event_data: dict) -> None:
        """Handle agent progress event.

        Args:
            event_data: Event data dictionary
        """
        if self._transform_fn:
            metrics = self._transform_fn("agent:progress", event_data)
            self._record_metrics(metrics)
        else:
            # Default metrics
            tags = self._build_tags(event_data)

            # Record progress as gauge
            progress = event_data.get("progress")
            if progress is not None:
                self._gauge("agent.progress", progress, tags=tags)

    def _handle_error(self, event_data: dict) -> None:
        """Handle agent error event.

        Args:
            event_data: Event data dictionary
        """
        if self._transform_fn:
            metrics = self._transform_fn("agent:error", event_data)
            self._record_metrics(metrics)
        else:
            # Default metrics
            tags = self._build_tags(event_data)

            # Add error type tag
            error_type = event_data.get("error_type", "unknown")
            tags["error_type"] = error_type

            # Increment error counter
            self._increment("agent.errors", tags=tags)

    def _handle_complete(self, event_data: dict) -> None:
        """Handle agent completion event.

        Args:
            event_data: Event data dictionary
        """
        if self._transform_fn:
            metrics = self._transform_fn("agent:complete", event_data)
            self._record_metrics(metrics)
        else:
            # Default metrics
            tags = self._build_tags(event_data)

            # Add success/failure tag
            success = event_data.get("success", True)
            tags["status"] = "success" if success else "failure"

            # Increment completed counter
            self._increment("agent.completed", tags=tags)

            # Record duration
            agent_name = event_data.get("agent_name")
            duration = event_data.get("duration_seconds")

            if duration is not None:
                self._histogram("agent.duration", duration, tags=tags)
            elif agent_name and agent_name in self._start_times:
                # Calculate duration from tracked start time
                duration = time.time() - self._start_times[agent_name]
                self._histogram("agent.duration", duration, tags=tags)
                del self._start_times[agent_name]

    def _build_tags(self, event_data: dict) -> Dict[str, str]:
        """Build tags dictionary from event data.

        Args:
            event_data: Event data dictionary

        Returns:
            Dictionary of tags
        """
        tags = self._global_tags.copy()

        # Add agent name tag
        if "agent_name" in event_data:
            tags["agent"] = event_data["agent_name"]

        return tags

    def _record_metrics(self, metrics: Dict[str, Any]) -> None:
        """Record metrics from custom transform function.

        Args:
            metrics: Dictionary of metrics to record
                    Format: {"metric_name": (type, value, tags)}
        """
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, tuple) and len(metric_data) >= 2:
                metric_type, value = metric_data[:2]
                tags = metric_data[2] if len(metric_data) > 2 else {}

                if metric_type == "counter":
                    self._increment(metric_name, value, tags)
                elif metric_type == "gauge":
                    self._gauge(metric_name, value, tags)
                elif metric_type == "histogram":
                    self._histogram(metric_name, value, tags)

    def _increment(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric.

        Args:
            name: Metric name
            value: Value to increment by (default: 1)
            tags: Metric tags
        """
        full_name = f"{self._prefix}.{name}"
        if hasattr(self._client, 'increment'):
            self._client.increment(full_name, value=value, tags=tags)
        elif hasattr(self._client, 'inc'):  # Prometheus style
            self._client.inc(full_name, value, tags)

    def _gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric.

        Args:
            name: Metric name
            value: Gauge value
            tags: Metric tags
        """
        full_name = f"{self._prefix}.{name}"
        if hasattr(self._client, 'gauge'):
            self._client.gauge(full_name, value, tags=tags)
        elif hasattr(self._client, 'set'):  # Prometheus style
            self._client.set(full_name, value, tags)

    def _histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram/timing metric.

        Args:
            name: Metric name
            value: Value to record
            tags: Metric tags
        """
        full_name = f"{self._prefix}.{name}"
        if hasattr(self._client, 'histogram'):
            self._client.histogram(full_name, value, tags=tags)
        elif hasattr(self._client, 'observe'):  # Prometheus style
            self._client.observe(full_name, value, tags)
        elif hasattr(self._client, 'timing'):  # StatsD style
            self._client.timing(full_name, value * 1000, tags=tags)  # Convert to ms

    def set_global_tags(self, tags: Dict[str, str]) -> None:
        """Update global tags added to all metrics.

        Args:
            tags: Dictionary of tag key-value pairs
        """
        self._global_tags.update(tags)

    def clear_tracked_executions(self) -> None:
        """Clear tracked execution start times.

        Useful for cleanup or testing.
        """
        self._start_times.clear()
