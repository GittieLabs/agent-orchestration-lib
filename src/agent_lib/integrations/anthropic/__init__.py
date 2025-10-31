"""Anthropic (Claude) integration for agent orchestration.

This module provides integration with Anthropic's Claude models, including
Claude 3 Opus, Sonnet, and Haiku. It includes token counting, cost estimation,
and seamless integration with the agent orchestration framework.

Examples:
    Basic usage:
    >>> from agent_lib import ExecutionContext, EventEmitter
    >>> from agent_lib.integrations.anthropic import (
    ...     AnthropicAgent,
    ...     AnthropicPrompt,
    ...     AnthropicMessage,
    ...     create_simple_prompt
    ... )
    >>>
    >>> context = ExecutionContext()
    >>> emitter = EventEmitter()
    >>>
    >>> agent = AnthropicAgent(
    ...     name="claude",
    ...     api_key="sk-ant-...",
    ...     context=context,
    ...     emitter=emitter
    ... )
    >>>
    >>> prompt = create_simple_prompt(
    ...     "What is Python?",
    ...     model="claude-3-sonnet-20240229",
    ...     max_tokens=1024
    ... )
    >>> result = await agent.execute(prompt)
    >>> print(result.content)
    >>> print(f"Cost: ${result.cost_usd:.6f}")

    With system message:
    >>> from agent_lib.integrations.anthropic import create_system_prompt
    >>>
    >>> prompt = create_system_prompt(
    ...     system="You are a helpful coding assistant",
    ...     user="How do I reverse a list in Python?",
    ...     max_tokens=2048
    ... )
    >>> result = await agent.execute(prompt)

    In a workflow:
    >>> from agent_lib import Flow
    >>> flow = Flow(context=context, emitter=emitter)
    >>> result = await flow.execute_sequential([agent], prompt)
"""

from .agent import AnthropicAgent, create_simple_prompt, create_system_prompt
from .models import AnthropicMessage, AnthropicPrompt, AnthropicResponse
from .utils import (
    calculate_cost,
    count_tokens,
    estimate_tokens,
    get_model_context_window,
    validate_model,
)

__all__ = [
    # Agent
    "AnthropicAgent",
    # Models
    "AnthropicPrompt",
    "AnthropicResponse",
    "AnthropicMessage",
    # Helper functions
    "create_simple_prompt",
    "create_system_prompt",
    # Utilities
    "calculate_cost",
    "count_tokens",
    "estimate_tokens",
    "validate_model",
    "get_model_context_window",
]
