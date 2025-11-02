"""
Agent Orchestration Library

A framework for building composable, observable, and maintainable agent workflows.

Core Components:
- ExecutionContext: Dependency injection container
- EventEmitter: Event-driven notification system
- AgentBlock: Base class for agent implementations
- RetryStrategy: Configurable retry logic with LLM fallbacks
- Flow: Multi-agent workflow orchestration

LLM Integrations (optional):
- OpenAI: GPT-4 and GPT-3.5-turbo support
- Anthropic: Claude 3 models (Opus, Sonnet, Haiku)
- Google Gemini: Gemini Pro and Gemini 1.5 models

Install integrations:
    pip install agent-orchestration-lib[openai]
    pip install agent-orchestration-lib[anthropic]
    pip install agent-orchestration-lib[gemini]
    pip install agent-orchestration-lib[all-llm]

See documentation at: https://agent-orchestration-lib.readthedocs.io
"""

__version__ = "0.3.3"

# Core components
from .core import (
    ExecutionContext,
    EventEmitter,
    AgentBlock,
    Flow,
    ConditionalStep,
    SwitchStep,
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
    "SwitchStep",
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
    # LLM integrations (available via submodules)
    # Import as: from agent_lib.integrations.openai import OpenAIAgent
    # Import as: from agent_lib.integrations.anthropic import AnthropicAgent
    # Import as: from agent_lib.integrations.gemini import GeminiAgent
]
