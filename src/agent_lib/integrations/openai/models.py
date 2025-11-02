"""Pydantic models for OpenAI integration."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ..common.tool_models import ToolCall, ToolDefinition


class OpenAIMessage(BaseModel):
    """A single message in the conversation."""

    role: Literal["system", "user", "assistant", "tool"] = Field(
        description="The role of the message sender"
    )
    content: Optional[str] = Field(default=None, description="The content of the message")
    tool_calls: Optional[List[ToolCall]] = Field(
        default=None, description="Tool calls made by assistant"
    )
    tool_call_id: Optional[str] = Field(
        default=None, description="ID of the tool call this message responds to (for role=tool)"
    )
    name: Optional[str] = Field(
        default=None, description="Name of the tool for tool messages"
    )


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
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    stop: Optional[List[str]] = Field(default=None, max_length=4, description="Stop sequences")
    response_format: Optional[Dict[str, str]] = Field(
        default=None, description="Response format specification"
    )
    tools: Optional[List[ToolDefinition]] = Field(
        default=None, description="Tools available for the model to call"
    )
    tool_choice: Optional[str] = Field(
        default=None, description="Control tool calling behavior (auto, none, or specific tool)"
    )


class OpenAIResponse(BaseModel):
    """Output model from OpenAI agent.

    Attributes:
        content: The generated text content (None if only tool calls)
        model: The model that generated the response
        finish_reason: Reason the generation stopped ('stop', 'length', 'tool_calls', etc.)
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total tokens used (prompt + completion)
        cost_usd: Estimated cost in USD for this request
        tool_calls: List of tool calls made by the model (None if no tools called)

    Examples:
        Text response:
        >>> response = OpenAIResponse(
        ...     content="Hello! How can I help you?",
        ...     model="gpt-4",
        ...     finish_reason="stop",
        ...     prompt_tokens=10,
        ...     completion_tokens=8,
        ...     total_tokens=18,
        ...     cost_usd=0.00054,
        ...     tool_calls=None
        ... )

        Tool call response:
        >>> response = OpenAIResponse(
        ...     content=None,
        ...     model="gpt-4",
        ...     finish_reason="tool_calls",
        ...     prompt_tokens=15,
        ...     completion_tokens=25,
        ...     total_tokens=40,
        ...     cost_usd=0.00120,
        ...     tool_calls=[ToolCall(id="call_abc", name="get_weather", arguments={"city": "NYC"})]
        ... )
    """

    content: Optional[str] = Field(default=None, description="Generated text content")
    model: str = Field(description="Model used for generation")
    finish_reason: str = Field(description="Reason generation stopped")
    prompt_tokens: int = Field(description="Tokens in prompt")
    completion_tokens: int = Field(description="Tokens in completion")
    total_tokens: int = Field(description="Total tokens used")
    cost_usd: float = Field(description="Estimated cost in USD")
    tool_calls: Optional[List[ToolCall]] = Field(
        default=None, description="Tool calls made by the model"
    )
