"""Pydantic models for Google Gemini integration."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class GeminiPrompt(BaseModel):
    """Input model for Gemini agent.

    Attributes:
        prompt: The text prompt to send to Gemini
        model: The Gemini model to use (e.g., 'gemini-pro', 'gemini-pro-vision')
        temperature: Sampling temperature between 0 and 1. Higher values make output
                    more random, lower values make it more deterministic.
        max_output_tokens: Maximum number of tokens to generate
        top_p: Nucleus sampling parameter. Alternative to temperature.
        top_k: Only sample from top K options for each token
        stop_sequences: Custom sequences that will stop generation
        candidate_count: Number of candidate responses to generate

    Examples:
        Simple prompt:
        >>> prompt = GeminiPrompt(
        ...     prompt="What is Python?",
        ...     model="gemini-pro"
        ... )

        With custom parameters:
        >>> prompt = GeminiPrompt(
        ...     prompt="Explain quantum computing",
        ...     model="gemini-pro",
        ...     temperature=0.7,
        ...     max_output_tokens=2048,
        ...     top_k=40
        ... )
    """

    prompt: str = Field(description="Text prompt for Gemini")
    model: str = Field(default="gemini-pro", description="Gemini model name")
    temperature: Optional[float] = Field(
        default=0.9, ge=0.0, le=1.0, description="Sampling temperature"
    )
    max_output_tokens: Optional[int] = Field(
        default=None, ge=1, description="Maximum tokens to generate"
    )
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling")
    top_k: Optional[int] = Field(default=None, ge=1, description="Top-k sampling")
    stop_sequences: Optional[List[str]] = Field(default=None, description="Stop sequences")
    candidate_count: int = Field(default=1, ge=1, le=8, description="Number of candidates")


class GeminiSafetySettings(BaseModel):
    """Safety settings for Gemini content filtering.

    Attributes:
        category: Safety category name
        threshold: Block threshold level
    """

    category: str = Field(description="Safety category")
    threshold: Literal[
        "BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"
    ] = Field(description="Block threshold")


class GeminiResponse(BaseModel):
    """Output model from Gemini agent.

    Attributes:
        content: The generated text content
        model: The model that generated the response
        finish_reason: Reason the generation stopped
        prompt_tokens: Number of tokens in the prompt (estimated)
        completion_tokens: Number of tokens in the completion (estimated)
        total_tokens: Total tokens used (prompt + completion)
        cost_usd: Estimated cost in USD for this request
        safety_ratings: Optional safety ratings from the model

    Examples:
        >>> response = GeminiResponse(
        ...     content="Python is a high-level programming language...",
        ...     model="gemini-pro",
        ...     finish_reason="STOP",
        ...     prompt_tokens=5,
        ...     completion_tokens=50,
        ...     total_tokens=55,
        ...     cost_usd=0.0000138
        ... )
    """

    content: str = Field(description="Generated text content")
    model: str = Field(description="Model used for generation")
    finish_reason: str = Field(description="Reason generation stopped")
    prompt_tokens: int = Field(description="Tokens in prompt (estimated)")
    completion_tokens: int = Field(description="Tokens in completion (estimated)")
    total_tokens: int = Field(description="Total tokens used")
    cost_usd: float = Field(description="Estimated cost in USD")
    safety_ratings: Optional[List[dict]] = Field(
        default=None, description="Safety ratings if available"
    )
