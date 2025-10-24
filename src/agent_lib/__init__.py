"""
Agent Orchestration Library

A framework for building composable, observable, and maintainable agent workflows.

Core Components:
- ExecutionContext: Dependency injection container
- EventEmitter: Event-driven notification system
- AgentBlock: Base class for agent implementations
- RetryStrategy: Configurable retry logic with LLM fallbacks (coming soon)
- Flow: Multi-agent workflow orchestration (coming soon)

See documentation at: https://agent-orchestration-lib.readthedocs.io
"""

__version__ = "0.2.0"

# Core components
from .core import (
    ExecutionContext,
    EventEmitter,
    AgentBlock,
    Flow,
    ConditionalStep,
    FlowAdapter,
)

# Event models
from .events import (
    Event,
    StartEvent,
    ProgressEvent,
    ErrorEvent,
    CompletionEvent,
    RetryEvent,
    MetricEvent,
)

# Retry strategies
from .retry import (
    RetryStrategy,
    NoRetry,
    ExponentialBackoffRetry,
    FixedDelayRetry,
    LinearBackoffRetry,
    LLMFallbackRetry,
    retry_on_exception_type,
    retry_on_error_message,
)

__all__ = [
    "__version__",
    # Core components
    "ExecutionContext",
    "EventEmitter",
    "AgentBlock",
    "Flow",
    "ConditionalStep",
    "FlowAdapter",
    # Event models
    "Event",
    "StartEvent",
    "ProgressEvent",
    "ErrorEvent",
    "CompletionEvent",
    "RetryEvent",
    "MetricEvent",
    # Retry strategies
    "RetryStrategy",
    "NoRetry",
    "ExponentialBackoffRetry",
    "FixedDelayRetry",
    "LinearBackoffRetry",
    "LLMFallbackRetry",
    "retry_on_exception_type",
    "retry_on_error_message",
]
