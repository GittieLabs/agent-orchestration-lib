"""OpenAI integration for agent orchestration.

This module provides integration with OpenAI's GPT models, including
GPT-4 and GPT-3.5-turbo. It includes token counting, cost estimation,
and seamless integration with the agent orchestration framework.

Examples:
    Basic usage:
    >>> from agent_lib import ExecutionContext, EventEmitter
    >>> from agent_lib.integrations.openai import (
    ...     OpenAIAgent,
    ...     OpenAIPrompt,
    ...     OpenAIMessage,
    ...     create_simple_prompt
    ... )
    >>>
    >>> context = ExecutionContext()
    >>> emitter = EventEmitter()
    >>>
    >>> agent = OpenAIAgent(
    ...     name="gpt4",
    ...     api_key="sk-...",
    ...     context=context,
    ...     emitter=emitter
    ... )
    >>>
    >>> prompt = create_simple_prompt("What is Python?", model="gpt-4")
    >>> result = await agent.execute(prompt)
    >>> print(result.content)
    >>> print(f"Cost: ${result.cost_usd:.4f}")

    With system message:
    >>> from agent_lib.integrations.openai import create_system_prompt
    >>>
    >>> prompt = create_system_prompt(
    ...     system="You are a helpful coding assistant",
    ...     user="How do I reverse a list in Python?"
    ... )
    >>> result = await agent.execute(prompt)

    In a workflow:
    >>> from agent_lib import Flow
    >>> flow = Flow(context=context, emitter=emitter)
    >>> result = await flow.execute_sequential([agent], prompt)
"""

from .agent import OpenAIAgent, create_simple_prompt, create_system_prompt
from .models import OpenAIMessage, OpenAIPrompt, OpenAIResponse
from .utils import calculate_cost, count_message_tokens, count_tokens, validate_model

__all__ = [
    # Agent
    "OpenAIAgent",
    # Models
    "OpenAIPrompt",
    "OpenAIResponse",
    "OpenAIMessage",
    # Helper functions
    "create_simple_prompt",
    "create_system_prompt",
    # Utilities
    "calculate_cost",
    "count_tokens",
    "count_message_tokens",
    "validate_model",
]
