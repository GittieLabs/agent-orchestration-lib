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

__version__ = "0.1.0"

# Core components
from .core import ExecutionContext, EventEmitter, AgentBlock

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

# Components to be implemented
# from .core import Flow
# from .retry import RetryStrategy, ExponentialBackoffRetry, LLMFallbackRetry

__all__ = [
    "__version__",
    # Core components
    "ExecutionContext",
    "EventEmitter",
    "AgentBlock",
    # Event models
    "Event",
    "StartEvent",
    "ProgressEvent",
    "ErrorEvent",
    "CompletionEvent",
    "RetryEvent",
    "MetricEvent",
    # To be implemented
    # "Flow",
    # "RetryStrategy",
    # "ExponentialBackoffRetry",
    # "LLMFallbackRetry",
]
