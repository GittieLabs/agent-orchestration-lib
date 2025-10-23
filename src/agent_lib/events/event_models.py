"""
Event models for agent notifications and observability.

Provides structured event types for tracking agent execution:
- Event: Base event with common fields
- ProgressEvent: Track agent progress (0-100%)
- ErrorEvent: Track errors and exceptions
- CompletionEvent: Track successful completions
- StartEvent: Track execution start
"""

from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Event(BaseModel):
    """Base event model for all agent notifications.

    All events include:
    - Unique event_id for tracking
    - Timestamp (auto-populated)
    - Event type for filtering
    - Source agent name
    - Optional data payload

    Example:
        ```python
        event = Event(
            type="custom",
            source="my_agent",
            data={"key": "value"}
        )
        ```
    """

    event_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique event identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event creation timestamp (UTC)"
    )
    type: str = Field(
        description="Event type for filtering and routing"
    )
    source: str = Field(
        description="Name of the agent that emitted this event"
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional event payload data"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StartEvent(Event):
    """Event emitted when agent execution starts.

    Example:
        ```python
        event = StartEvent(
            source="parsing_agent",
            input_summary="Resume for John Doe"
        )
        ```
    """

    type: Literal["start"] = "start"
    input_summary: Optional[str] = Field(
        default=None,
        description="Brief description of input data"
    )


class ProgressEvent(Event):
    """Event emitted to track agent progress.

    Progress is represented as a float between 0.0 and 1.0 (or 0-100 if using percentage).

    Example:
        ```python
        event = ProgressEvent(
            source="parsing_agent",
            stage="analyzing_structure",
            progress=0.25,  # 25%
            message="Analyzing PDF structure..."
        )
        ```
    """

    type: Literal["progress"] = "progress"
    stage: str = Field(
        description="Current processing stage"
    )
    progress: float = Field(
        ge=0.0,
        le=1.0,
        description="Progress value between 0.0 and 1.0"
    )
    message: str = Field(
        description="Human-readable progress message"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional detailed progress information"
    )


class ErrorEvent(Event):
    """Event emitted when an error occurs during execution.

    Example:
        ```python
        try:
            result = await agent.execute(input_data)
        except Exception as e:
            event = ErrorEvent(
                source="parsing_agent",
                error_type="ValidationError",
                error_message=str(e),
                recoverable=True
            )
            await events.emit(event)
        ```
    """

    type: Literal["error"] = "error"
    error_type: str = Field(
        description="Type/class of the error"
    )
    error_message: str = Field(
        description="Error message"
    )
    stack_trace: Optional[str] = Field(
        default=None,
        description="Optional stack trace for debugging"
    )
    recoverable: bool = Field(
        default=False,
        description="Whether the error is recoverable (will retry)"
    )
    attempt: Optional[int] = Field(
        default=None,
        description="Attempt number if retrying"
    )


class CompletionEvent(Event):
    """Event emitted when agent execution completes successfully.

    Example:
        ```python
        event = CompletionEvent(
            source="parsing_agent",
            duration_seconds=2.5,
            output_summary="Extracted 15 work experiences",
            metrics={"tokens_used": 1500}
        )
        ```
    """

    type: Literal["completion"] = "completion"
    duration_seconds: float = Field(
        description="Total execution duration in seconds"
    )
    output_summary: Optional[str] = Field(
        default=None,
        description="Brief description of output/result"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional execution metrics (tokens, API calls, etc.)"
    )
    success: bool = Field(
        default=True,
        description="Whether execution was successful"
    )


class RetryEvent(Event):
    """Event emitted when a retry attempt is made.

    Example:
        ```python
        event = RetryEvent(
            source="parsing_agent",
            attempt=2,
            max_attempts=3,
            reason="Rate limit exceeded",
            wait_seconds=4.0
        )
        ```
    """

    type: Literal["retry"] = "retry"
    attempt: int = Field(
        ge=1,
        description="Current attempt number (1-indexed)"
    )
    max_attempts: int = Field(
        ge=1,
        description="Maximum number of attempts"
    )
    reason: str = Field(
        description="Reason for retry"
    )
    wait_seconds: float = Field(
        ge=0.0,
        description="Wait time before next attempt"
    )
    strategy: Optional[str] = Field(
        default=None,
        description="Retry strategy being used"
    )


class MetricEvent(Event):
    """Event emitted for tracking metrics and observability.

    Example:
        ```python
        event = MetricEvent(
            source="parsing_agent",
            metric_name="tokens_used",
            metric_value=1500,
            unit="tokens",
            tags={"model": "gpt-4", "user_id": "123"}
        )
        ```
    """

    type: Literal["metric"] = "metric"
    metric_name: str = Field(
        description="Name of the metric"
    )
    metric_value: float = Field(
        description="Numeric value of the metric"
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unit of measurement"
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Tags for metric aggregation and filtering"
    )


# Type alias for all event types
AnyEvent = Event | StartEvent | ProgressEvent | ErrorEvent | CompletionEvent | RetryEvent | MetricEvent
