"""Integrations with LLM providers and frameworks.

This module provides integrations with various LLM providers and frameworks,
making it easy to build AI agent workflows with popular models.

Available Integrations:
    - OpenAI (GPT-4, GPT-3.5-turbo)
    - Anthropic (Claude) - Coming soon
    - Google Gemini - Coming soon

Examples:
    Using OpenAI:
    >>> from agent_lib.integrations.openai import OpenAIAgent, create_simple_prompt
    >>> agent = OpenAIAgent(name="gpt4", api_key="sk-...", context=context, emitter=emitter)
    >>> prompt = create_simple_prompt("What is Python?")
    >>> result = await agent.execute(prompt)
"""

# Import all integration modules
# Note: Imports are optional - they will only work if the corresponding
# packages are installed (e.g., pip install agent-orchestration-lib[openai])

__all__ = []

# Try to import OpenAI integration
try:
    from . import openai

    __all__.append("openai")
except ImportError:
    pass

# Try to import Anthropic integration (when implemented)
try:
    from . import anthropic

    __all__.append("anthropic")
except ImportError:
    pass

# Try to import Gemini integration (when implemented)
try:
    from . import gemini

    __all__.append("gemini")
except ImportError:
    pass
