"""Pydantic models for Anthropic integration."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ..common.tool_models import ToolCall, ToolDefinition


class AnthropicMessage(BaseModel):
    """A single message in the conversation."""

    role: Literal["user", "assistant"] = Field(description="The role of the message sender")
    content: str = Field(description="The content of the message")


class AnthropicPrompt(BaseModel):
    """Input model for Anthropic agent.

    Attributes:
        messages: List of conversation messages (alternating user/assistant)
        system: Optional system prompt to set context
        model: The Anthropic model to use (e.g., 'claude-3-opus-20240229')
        temperature: Sampling temperature between 0 and 1. Higher values make output
                    more random, lower values make it more deterministic.
        max_tokens: Maximum number of tokens to generate (required by Anthropic API)
        top_p: Nucleus sampling parameter. Alternative to temperature.
        top_k: Only sample from top K options for each token
        stop_sequences: Custom sequences that will stop generation

    Examples:
        Simple prompt:
        >>> prompt = AnthropicPrompt(
        ...     messages=[AnthropicMessage(role="user", content="Hello!")],
        ...     model="claude-3-sonnet-20240229",
        ...     max_tokens=1024
        ... )

        With system prompt:
        >>> prompt = AnthropicPrompt(
        ...     system="You are a helpful coding assistant",
        ...     messages=[
        ...         AnthropicMessage(role="user", content="How do I reverse a list in Python?")
        ...     ],
        ...     model="claude-3-opus-20240229",
        ...     max_tokens=2048,
        ...     temperature=0.7
        ... )
    """

    messages: List[AnthropicMessage] = Field(description="Conversation messages")
    system: Optional[str] = Field(default=None, description="System prompt")
    model: str = Field(default="claude-3-sonnet-20240229", description="Anthropic model name")
    temperature: float = Field(default=1.0, ge=0.0, le=1.0, description="Sampling temperature")
    max_tokens: int = Field(ge=1, le=4096, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling")
    top_k: Optional[int] = Field(default=None, ge=0, description="Top-k sampling")
    stop_sequences: Optional[List[str]] = Field(
        default=None, max_length=4, description="Stop sequences"
    )
    tools: Optional[List[ToolDefinition]] = Field(
        default=None, description="Tools available for the model to call"
    )
    tool_choice: Optional[Dict[str, Any]] = Field(
        default=None, description="Control tool calling behavior"
    )


class AnthropicResponse(BaseModel):
    """Output model from Anthropic agent.

    Attributes:
        content: The generated text content
        model: The model that generated the response
        stop_reason: Reason the generation stopped ('end_turn', 'max_tokens', 'stop_sequence')
        input_tokens: Number of tokens in the input
        output_tokens: Number of tokens in the output
        total_tokens: Total tokens used (input + output)
        cost_usd: Estimated cost in USD for this request

    Examples:
        >>> response = AnthropicResponse(
        ...     content="Hello! How can I help you?",
        ...     model="claude-3-sonnet-20240229",
        ...     stop_reason="end_turn",
        ...     input_tokens=10,
        ...     output_tokens=8,
        ...     total_tokens=18,
        ...     cost_usd=0.000054
        ... )
    """

    content: Optional[str] = Field(default=None, description="Generated text content")
    model: str = Field(description="Model used for generation")
    stop_reason: str = Field(description="Reason generation stopped")
    input_tokens: int = Field(description="Tokens in input")
    output_tokens: int = Field(description="Tokens in output")
    total_tokens: int = Field(description="Total tokens used")
    cost_usd: float = Field(description="Estimated cost in USD")
    tool_calls: Optional[List[ToolCall]] = Field(
        default=None, description="Tool calls made by the model"
    )
