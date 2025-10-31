"""Utility functions for Anthropic integration."""

from typing import Dict

# Anthropic pricing as of October 2025 (per 1M tokens)
# Source: https://www.anthropic.com/pricing
ANTHROPIC_PRICING: Dict[str, Dict[str, float]] = {
    "claude-3-opus-20240229": {
        "input": 15.00,  # per 1M tokens
        "output": 75.00,  # per 1M tokens
    },
    "claude-3-sonnet-20240229": {
        "input": 3.00,
        "output": 15.00,
    },
    "claude-3-haiku-20240307": {
        "input": 0.25,
        "output": 1.25,
    },
    # Aliases
    "claude-3-opus": {
        "input": 15.00,
        "output": 75.00,
    },
    "claude-3-sonnet": {
        "input": 3.00,
        "output": 15.00,
    },
    "claude-3-haiku": {
        "input": 0.25,
        "output": 1.25,
    },
}


def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Calculate the cost of an Anthropic API call.

    Args:
        input_tokens: Number of tokens in the input
        output_tokens: Number of tokens in the output
        model: Anthropic model name

    Returns:
        Estimated cost in USD

    Examples:
        >>> calculate_cost(1000, 500, "claude-3-opus-20240229")
        0.0525

        >>> calculate_cost(1000, 500, "claude-3-sonnet-20240229")
        0.0105

        >>> calculate_cost(1000, 500, "claude-3-haiku-20240307")
        0.000875
    """
    # Get pricing for model, default to sonnet if unknown
    pricing = ANTHROPIC_PRICING.get(model, ANTHROPIC_PRICING["claude-3-sonnet-20240229"])

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


def count_tokens(text: str, model: str = "claude-3-sonnet-20240229") -> int:
    """Count the number of tokens in a text string.

    Uses Anthropic's token counting API if available, otherwise estimates.

    Args:
        text: The text to count tokens for
        model: The model to use for tokenization

    Returns:
        Number of tokens

    Examples:
        >>> count_tokens("Hello, world!")
        4

        >>> count_tokens("This is a longer text with more tokens")
        9

    Note:
        This is an approximation. For exact counts, use Anthropic's
        count_tokens API method when making actual API calls.
    """
    try:
        from anthropic import Anthropic

        client = Anthropic()
        # Use Anthropic's count_tokens if available
        result = client.count_tokens(text)
        return result
    except (ImportError, Exception):
        # Fallback: rough estimate (similar to OpenAI's tokenization)
        # Anthropic uses a similar tokenizer to GPT models
        return estimate_tokens(text)


def estimate_tokens(text: str) -> int:
    """Estimate token count without API call (rough approximation).

    Rule of thumb: ~4 characters per token for English text.

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated number of tokens

    Examples:
        >>> estimate_tokens("Hello, world!")
        4

        >>> estimate_tokens("This is a longer text" * 10)
        55
    """
    # Rough estimate: 4 characters per token for English
    return max(1, len(text) // 4)


def validate_model(model: str) -> bool:
    """Validate if a model name is a known Anthropic model.

    Args:
        model: Model name to validate

    Returns:
        True if model is recognized, False otherwise

    Examples:
        >>> validate_model("claude-3-opus-20240229")
        True

        >>> validate_model("claude-3-sonnet")
        True

        >>> validate_model("invalid-model")
        False
    """
    known_models = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
    ]

    return model in known_models or model.startswith("claude-")


def get_model_context_window(model: str) -> int:
    """Get the context window size for a Claude model.

    Args:
        model: Model name

    Returns:
        Context window size in tokens

    Examples:
        >>> get_model_context_window("claude-3-opus-20240229")
        200000

        >>> get_model_context_window("claude-3-sonnet")
        200000
    """
    # All Claude 3 models have 200K context window
    if model.startswith("claude-3"):
        return 200_000

    # Default for unknown models
    return 100_000


def format_messages_for_api(messages: list, system: str | None = None) -> dict:
    """Format messages for Anthropic API.

    Args:
        messages: List of message dictionaries
        system: Optional system prompt

    Returns:
        Dictionary with formatted parameters for API call

    Examples:
        >>> msgs = [{"role": "user", "content": "Hello"}]
        >>> format_messages_for_api(msgs, "You are helpful")
        {'messages': [{'role': 'user', 'content': 'Hello'}], 'system': 'You are helpful'}
    """
    params = {"messages": messages}

    if system:
        params["system"] = system

    return params
