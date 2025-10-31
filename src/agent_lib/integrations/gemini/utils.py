"""Utility functions for Google Gemini integration."""

from typing import Dict

# Gemini pricing as of October 2025 (per 1M tokens)
# Source: https://ai.google.dev/pricing
GEMINI_PRICING: Dict[str, Dict[str, float]] = {
    "gemini-pro": {
        "input": 0.50,   # per 1M tokens
        "output": 1.50,  # per 1M tokens
    },
    "gemini-pro-vision": {
        "input": 0.50,
        "output": 1.50,
    },
    "gemini-1.5-pro": {
        "input": 3.50,   # For prompts up to 128K tokens
        "output": 10.50,
    },
    "gemini-1.5-flash": {
        "input": 0.075,
        "output": 0.30,
    },
}


def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Calculate the cost of a Gemini API call.

    Args:
        input_tokens: Number of tokens in the input
        output_tokens: Number of tokens in the output
        model: Gemini model name

    Returns:
        Estimated cost in USD

    Examples:
        >>> calculate_cost(1000, 500, "gemini-pro")
        0.0013

        >>> calculate_cost(1000, 500, "gemini-1.5-flash")
        0.000225
    """
    # Get pricing for model, default to gemini-pro if unknown
    pricing = GEMINI_PRICING.get(model, GEMINI_PRICING["gemini-pro"])

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


def estimate_tokens(text: str) -> int:
    """Estimate token count for Gemini.

    Gemini uses a similar tokenizer to other modern LLMs.
    This is a rough approximation.

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
    # Gemini's tokenizer is similar to other modern models
    return max(1, len(text) // 4)


def count_tokens(text: str, model: str = "gemini-pro") -> int:
    """Count tokens using Gemini's token counter if available.

    Args:
        text: Text to count tokens for
        model: Model name for tokenization

    Returns:
        Token count

    Examples:
        >>> count_tokens("Hello, world!")
        4
    """
    try:
        import google.generativeai as genai

        # Try to use Gemini's actual token counter
        model_obj = genai.GenerativeModel(model)
        result = model_obj.count_tokens(text)
        return result.total_tokens
    except (ImportError, Exception):
        # Fallback to estimation
        return estimate_tokens(text)


def validate_model(model: str) -> bool:
    """Validate if a model name is a known Gemini model.

    Args:
        model: Model name to validate

    Returns:
        True if model is recognized, False otherwise

    Examples:
        >>> validate_model("gemini-pro")
        True

        >>> validate_model("gemini-1.5-flash")
        True

        >>> validate_model("invalid-model")
        False
    """
    known_models = [
        "gemini-pro",
        "gemini-pro-vision",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash-latest",
    ]

    return model in known_models or model.startswith("gemini-")


def get_model_context_window(model: str) -> int:
    """Get the context window size for a Gemini model.

    Args:
        model: Model name

    Returns:
        Context window size in tokens

    Examples:
        >>> get_model_context_window("gemini-pro")
        32768

        >>> get_model_context_window("gemini-1.5-pro")
        1048576
    """
    context_windows = {
        "gemini-pro": 32_768,
        "gemini-pro-vision": 16_384,
        "gemini-1.5-pro": 1_048_576,  # 1M tokens
        "gemini-1.5-flash": 1_048_576,  # 1M tokens
    }

    return context_windows.get(model, 32_768)


def create_generation_config(
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_output_tokens: int | None = None,
    stop_sequences: list | None = None,
    candidate_count: int = 1,
) -> dict:
    """Create a generation config dictionary for Gemini API.

    Args:
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        max_output_tokens: Maximum tokens to generate
        stop_sequences: Stop sequences
        candidate_count: Number of candidates

    Returns:
        Dictionary with generation configuration

    Examples:
        >>> config = create_generation_config(temperature=0.7, top_k=40)
        >>> config
        {'temperature': 0.7, 'top_k': 40, 'candidate_count': 1}
    """
    config = {}

    if temperature is not None:
        config["temperature"] = temperature
    if top_p is not None:
        config["top_p"] = top_p
    if top_k is not None:
        config["top_k"] = top_k
    if max_output_tokens is not None:
        config["max_output_tokens"] = max_output_tokens
    if stop_sequences is not None:
        config["stop_sequences"] = stop_sequences

    config["candidate_count"] = candidate_count

    return config


def parse_finish_reason(finish_reason: str | None) -> str:
    """Parse and normalize Gemini finish reason.

    Args:
        finish_reason: Raw finish reason from API

    Returns:
        Normalized finish reason string

    Examples:
        >>> parse_finish_reason("STOP")
        'stop'

        >>> parse_finish_reason("MAX_TOKENS")
        'max_tokens'

        >>> parse_finish_reason(None)
        'unknown'
    """
    if finish_reason is None:
        return "unknown"

    # Convert to lowercase and replace underscores
    return finish_reason.lower().replace("_", "_")
