"""Pydantic models for OpenAI integration."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class OpenAIMessage(BaseModel):
    """A single message in the conversation."""

    role: Literal["system", "user", "assistant"] = Field(
        description="The role of the message sender"
    )
    content: str = Field(description="The content of the message")


class OpenAIPrompt(BaseModel):
    """Input model for OpenAI agent.

    Attributes:
        messages: List of conversation messages. If a single string is provided,
                 it will be converted to a user message.
        model: The OpenAI model to use (e.g., 'gpt-4', 'gpt-3.5-turbo')
        temperature: Sampling temperature between 0 and 2. Higher values make output
                    more random, lower values make it more deterministic.
        max_tokens: Maximum number of tokens to generate
        top_p: Nucleus sampling parameter. Alternative to temperature.
        frequency_penalty: Penalty for token frequency (-2.0 to 2.0)
        presence_penalty: Penalty for token presence (-2.0 to 2.0)
        stop: Up to 4 sequences where the API will stop generating
        response_format: Format for the response (e.g., {"type": "json_object"})

    Examples:
        Simple prompt:
        >>> prompt = OpenAIPrompt(
        ...     messages=[OpenAIMessage(role="user", content="Hello!")],
        ...     model="gpt-4"
        ... )

        With system message:
        >>> prompt = OpenAIPrompt(
        ...     messages=[
        ...         OpenAIMessage(role="system", content="You are a helpful assistant"),
        ...         OpenAIMessage(role="user", content="What is Python?")
        ...     ],
        ...     model="gpt-3.5-turbo",
        ...     temperature=0.7
        ... )
    """

    messages: List[OpenAIMessage] = Field(description="Conversation messages")
    model: str = Field(default="gpt-4", description="OpenAI model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(
        default=None, ge=1, description="Maximum tokens to generate"
    )
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling")
    frequency_penalty: float = Field(
        default=0.0, ge=-2.0, le=2.0, description="Frequency penalty"
    )
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    stop: Optional[List[str]] = Field(default=None, max_length=4, description="Stop sequences")
    response_format: Optional[Dict[str, str]] = Field(
        default=None, description="Response format specification"
    )


class OpenAIResponse(BaseModel):
    """Output model from OpenAI agent.

    Attributes:
        content: The generated text content
        model: The model that generated the response
        finish_reason: Reason the generation stopped ('stop', 'length', 'content_filter', etc.)
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total tokens used (prompt + completion)
        cost_usd: Estimated cost in USD for this request

    Examples:
        >>> response = OpenAIResponse(
        ...     content="Hello! How can I help you?",
        ...     model="gpt-4",
        ...     finish_reason="stop",
        ...     prompt_tokens=10,
        ...     completion_tokens=8,
        ...     total_tokens=18,
        ...     cost_usd=0.00054
        ... )
    """

    content: str = Field(description="Generated text content")
    model: str = Field(description="Model used for generation")
    finish_reason: str = Field(description="Reason generation stopped")
    prompt_tokens: int = Field(description="Tokens in prompt")
    completion_tokens: int = Field(description="Tokens in completion")
    total_tokens: int = Field(description="Total tokens used")
    cost_usd: float = Field(description="Estimated cost in USD")
