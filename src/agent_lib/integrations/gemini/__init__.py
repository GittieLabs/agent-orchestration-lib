"""Google Gemini integration for agent orchestration.

This module provides integration with Google's Gemini models, including
Gemini Pro, Gemini Pro Vision, and Gemini 1.5 models. It includes token
estimation, cost calculation, and seamless integration with the agent
orchestration framework.

Examples:
    Basic usage:
    >>> import os
    >>> from agent_lib import ExecutionContext, EventEmitter
    >>> from agent_lib.integrations.gemini import (
    ...     GeminiAgent,
    ...     GeminiPrompt,
    ...     create_simple_prompt
    ... )
    >>>
    >>> context = ExecutionContext()
    >>> emitter = EventEmitter()
    >>>
    >>> agent = GeminiAgent(
    ...     name="gemini",
    ...     api_key=os.getenv("GOOGLE_API_KEY"),
    ...     context=context,
    ...     emitter=emitter
    ... )
    >>>
    >>> prompt = create_simple_prompt("What is Python?", model="gemini-pro")
    >>> result = await agent.execute(prompt)
    >>> print(result.content)
    >>> print(f"Cost: ${result.cost_usd:.6f}")

    With custom parameters:
    >>> prompt = GeminiPrompt(
    ...     prompt="Explain quantum computing",
    ...     model="gemini-1.5-flash",
    ...     temperature=0.7,
    ...     max_output_tokens=2048
    ... )
    >>> result = await agent.execute(prompt)

    In a workflow:
    >>> from agent_lib import Flow
    >>> flow = Flow(context=context, emitter=emitter)
    >>> result = await flow.execute_sequential([agent], prompt)
"""

from .agent import GeminiAgent, create_simple_prompt
from .models import GeminiPrompt, GeminiResponse, GeminiSafetySettings
from .utils import (
    calculate_cost,
    count_tokens,
    create_generation_config,
    estimate_tokens,
    get_model_context_window,
    validate_model,
)

__all__ = [
    # Agent
    "GeminiAgent",
    # Models
    "GeminiPrompt",
    "GeminiResponse",
    "GeminiSafetySettings",
    # Helper functions
    "create_simple_prompt",
    # Utilities
    "calculate_cost",
    "count_tokens",
    "estimate_tokens",
    "validate_model",
    "get_model_context_window",
    "create_generation_config",
]
