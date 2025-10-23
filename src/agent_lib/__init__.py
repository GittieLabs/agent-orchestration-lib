"""
Agent Orchestration Library

A framework for building composable, observable, and maintainable agent workflows.

Core Components:
- ExecutionContext: Dependency injection container
- EventEmitter: Event-driven notification system
- AgentBlock: Base class for agent implementations
- RetryStrategy: Configurable retry logic with LLM fallbacks
- Flow: Multi-agent workflow orchestration

See documentation at: https://agent-orchestration-lib.readthedocs.io
"""

__version__ = "0.1.0"

# Core components will be imported here once implemented
# from .core import ExecutionContext, EventEmitter, AgentBlock, Flow
# from .retry import RetryStrategy, ExponentialBackoffRetry, LLMFallbackRetry
# from .events import Event, ProgressEvent, ErrorEvent, CompletionEvent

__all__ = [
    "__version__",
    # Core components (to be added)
    # "ExecutionContext",
    # "EventEmitter",
    # "AgentBlock",
    # "Flow",
    # Retry strategies (to be added)
    # "RetryStrategy",
    # "ExponentialBackoffRetry",
    # "LLMFallbackRetry",
    # Events (to be added)
    # "Event",
    # "ProgressEvent",
    # "ErrorEvent",
    # "CompletionEvent",
]
