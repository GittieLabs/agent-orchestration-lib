"""Utility functions for OpenAI integration."""

from typing import Dict, List

# OpenAI pricing as of October 2025 (per 1K tokens)
# Source: https://openai.com/pricing
OPENAI_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4": {
        "input": 0.03,
        "output": 0.06,
    },
    "gpt-4-turbo": {
        "input": 0.01,
        "output": 0.03,
    },
    "gpt-4-turbo-preview": {
        "input": 0.01,
        "output": 0.03,
    },
    "gpt-4-1106-preview": {
        "input": 0.01,
        "output": 0.03,
    },
    "gpt-3.5-turbo": {
        "input": 0.0005,
        "output": 0.0015,
    },
    "gpt-3.5-turbo-1106": {
        "input": 0.001,
        "output": 0.002,
    },
    "gpt-3.5-turbo-16k": {
        "input": 0.003,
        "output": 0.004,
    },
}


def calculate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Calculate the cost of an OpenAI API call.

    Args:
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        model: OpenAI model name

    Returns:
        Estimated cost in USD

    Examples:
        >>> calculate_cost(100, 50, "gpt-4")
        0.006

        >>> calculate_cost(1000, 500, "gpt-3.5-turbo")
        0.0013
    """
    # Get pricing for model, default to gpt-4 if unknown
    pricing = OPENAI_PRICING.get(model, OPENAI_PRICING["gpt-4"])

    input_cost = (prompt_tokens / 1000) * pricing["input"]
    output_cost = (completion_tokens / 1000) * pricing["output"]

    return input_cost + output_cost


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count the number of tokens in a text string.

    Uses tiktoken for accurate token counting.

    Args:
        text: The text to count tokens for
        model: The model to use for tokenization

    Returns:
        Number of tokens

    Raises:
        ImportError: If tiktoken is not installed

    Examples:
        >>> count_tokens("Hello, world!")
        4

        >>> count_tokens("This is a longer text with more tokens", "gpt-3.5-turbo")
        9
    """
    try:
        import tiktoken
    except ImportError:
        raise ImportError(
            "tiktoken is required for token counting. "
            "Install with: pip install agent-orchestration-lib[openai]"
        )

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Default to cl100k_base encoding (used by gpt-4 and gpt-3.5-turbo)
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def count_message_tokens(messages: List[Dict[str, str]], model: str = "gpt-4") -> int:
    """Count tokens in a list of messages.

    This accounts for the special tokens used in the chat format.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: The model to use for tokenization

    Returns:
        Number of tokens including formatting tokens

    Examples:
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful"},
        ...     {"role": "user", "content": "Hello!"}
        ... ]
        >>> count_message_tokens(messages)
        17
    """
    try:
        import tiktoken
    except ImportError:
        raise ImportError(
            "tiktoken is required for token counting. "
            "Install with: pip install agent-orchestration-lib[openai]"
        )

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    # Token counting logic based on OpenAI's cookbook
    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    tokens_per_message = 3  # Every message follows <|start|>{role/name}\n{content}<|end|>\n
    tokens_per_name = 1  # If there's a name field

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name

    num_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>

    return num_tokens


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate token count without tiktoken (rough approximation).

    This is a fallback when tiktoken is not available. It's less accurate
    but doesn't require additional dependencies.

    Rule of thumb: ~4 characters per token for English text.

    Args:
        text: The text to estimate tokens for
        model: The model name (not used, for API compatibility)

    Returns:
        Estimated number of tokens

    Examples:
        >>> estimate_tokens("Hello, world!")
        4

        >>> estimate_tokens("This is a longer text" * 10)
        55
    """
    # Rough estimate: 4 characters per token for English
    return len(text) // 4


def validate_model(model: str) -> bool:
    """Validate if a model name is a known OpenAI model.

    Args:
        model: Model name to validate

    Returns:
        True if model is recognized, False otherwise

    Examples:
        >>> validate_model("gpt-4")
        True

        >>> validate_model("invalid-model")
        False
    """
    known_models = [
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4-1106-preview",
        "gpt-4-0613",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-0613",
    ]

    return model in known_models or model.startswith(("gpt-4", "gpt-3.5"))
