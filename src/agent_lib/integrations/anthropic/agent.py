"""Anthropic (Claude) agent implementation."""

from typing import Any, Dict, Optional

from loguru import logger

from agent_lib.core.agent_block import AgentBlock
from agent_lib.core.event_emitter import EventEmitter
from agent_lib.core.execution_context import ExecutionContext

from .models import AnthropicMessage, AnthropicPrompt, AnthropicResponse
from .utils import calculate_cost
from .tool_utils import tool_definition_to_anthropic, anthropic_tool_use_to_unified


class AnthropicAgent(AgentBlock[AnthropicPrompt, AnthropicResponse]):
    """Agent for calling Anthropic's Claude API.

    This agent provides a type-safe interface to Anthropic's Claude models,
    with automatic token counting, cost estimation, and integration
    with the agent orchestration framework.

    Attributes:
        api_key: Anthropic API key
        client: Anthropic async client (created lazily)
        default_model: Default model to use if not specified in prompt

    Examples:
        Basic usage:
        >>> from agent_lib import ExecutionContext, EventEmitter
        >>> from agent_lib.integrations.anthropic import (
        ...     AnthropicAgent,
        ...     AnthropicPrompt,
        ...     AnthropicMessage
        ... )
        >>>
        >>> context = ExecutionContext()
        >>> emitter = EventEmitter()
        >>> agent = AnthropicAgent(
        ...     name="claude-agent",
        ...     api_key="sk-ant-...",
        ...     context=context,
        ...     emitter=emitter
        ... )
        >>>
        >>> prompt = AnthropicPrompt(
        ...     messages=[AnthropicMessage(role="user", content="What is Python?")],
        ...     model="claude-3-sonnet-20240229",
        ...     max_tokens=1024
        ... )
        >>> result = await agent.execute(prompt)
        >>> print(result.content)
        >>> print(f"Cost: ${result.cost_usd:.6f}")

        With system message:
        >>> prompt = AnthropicPrompt(
        ...     system="You are a helpful coding assistant",
        ...     messages=[
        ...         AnthropicMessage(role="user", content="How do I reverse a list in Python?")
        ...     ],
        ...     model="claude-3-opus-20240229",
        ...     max_tokens=2048,
        ...     temperature=0.7
        ... )
        >>> result = await agent.execute(prompt)

        In a flow:
        >>> from agent_lib import Flow
        >>> flow = Flow(context=context, emitter=emitter)
        >>> result = await flow.execute_sequential([agent], prompt)
    """

    def __init__(
        self,
        name: str,
        api_key: str,
        context: ExecutionContext,
        emitter: EventEmitter,
        default_model: str = "claude-3-sonnet-20240229",
        default_max_tokens: int = 1024,
        **kwargs: Any,
    ):
        """Initialize the Anthropic agent.

        Args:
            name: Name of the agent
            api_key: Anthropic API key
            context: Execution context for dependency injection
            emitter: Event emitter for observability
            default_model: Default model to use if not specified
            default_max_tokens: Default max tokens if not specified
            **kwargs: Additional arguments passed to AgentBlock
        """
        super().__init__(name, context, emitter, **kwargs)
        self.api_key = api_key
        self.default_model = default_model
        self.default_max_tokens = default_max_tokens
        self._client: Optional[Any] = None

    @property
    def client(self) -> Any:
        """Get or create the Anthropic async client.

        Returns:
            Anthropic AsyncAnthropic client instance

        Raises:
            ImportError: If anthropic package is not installed
        """
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError(
                    "anthropic package is required for Anthropic integration. "
                    "Install with: pip install agent-orchestration-lib[anthropic]"
                )

            self._client = AsyncAnthropic(api_key=self.api_key)
            logger.debug(f"Created Anthropic client for agent '{self.name}'")

        return self._client

    async def process(self, input_data: AnthropicPrompt) -> AnthropicResponse:
        """Process the input through Anthropic's API.

        Args:
            input_data: Anthropic prompt with messages and parameters

        Returns:
            Anthropic response with content, tokens, and cost

        Raises:
            Exception: If the API call fails
        """
        model = input_data.model or self.default_model
        max_tokens = input_data.max_tokens or self.default_max_tokens

        # Emit progress before API call
        await self.emit_progress(
            stage="calling_anthropic_api",
            progress=0.0,
            message=f"Calling Anthropic API with model '{model}'",
            details={"model": model, "messages_count": len(input_data.messages)},
        )

        # Convert Pydantic models to dicts for API
        messages_dict = [msg.model_dump() for msg in input_data.messages]

        # Build API parameters
        api_params: Dict[str, Any] = {
            "model": model,
            "messages": messages_dict,
            "max_tokens": max_tokens,
            "temperature": input_data.temperature,
        }

        # Add optional parameters
        if input_data.system is not None:
            api_params["system"] = input_data.system
        if input_data.top_p is not None:
            api_params["top_p"] = input_data.top_p
        if input_data.top_k is not None:
            api_params["top_k"] = input_data.top_k
        if input_data.stop_sequences is not None:
            api_params["stop_sequences"] = input_data.stop_sequences

        # Add tools if provided
        if input_data.tools is not None:
            api_params["tools"] = [tool_definition_to_anthropic(tool) for tool in input_data.tools]
            if input_data.tool_choice is not None:
                api_params["tool_choice"] = input_data.tool_choice

        logger.debug(
            f"Calling Anthropic API with model='{model}', "
            f"messages={len(messages_dict)}, max_tokens={max_tokens}, "
            f"temp={input_data.temperature}, "
            f"tools={len(input_data.tools) if input_data.tools else 0}"
        )

        try:
            # Call Anthropic API
            response = await self.client.messages.create(**api_params)

            # Extract response data (text and tool calls)
            content = ""
            tool_calls = []

            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text
                elif hasattr(block, "type") and block.type == "tool_use":
                    # Convert Anthropic tool_use block to unified ToolCall
                    tool_call = anthropic_tool_use_to_unified(block)
                    tool_calls.append(tool_call)

            # Set content to None if empty and we have tool calls
            if not content and tool_calls:
                content = None
            elif not content:
                content = ""

            stop_reason = response.stop_reason
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens

            # Calculate cost
            cost_usd = calculate_cost(input_tokens, output_tokens, model)

            # Emit progress after successful API call
            await self.emit_progress(
                stage="anthropic_api_complete",
                progress=1.0,
                message=f"Anthropic API call complete: {total_tokens} tokens, ${cost_usd:.6f}",
                details={
                    "tokens": total_tokens,
                    "cost": cost_usd,
                    "stop_reason": stop_reason,
                },
            )

            logger.info(
                f"Anthropic API call complete: model='{model}', "
                f"tokens={total_tokens}, cost=${cost_usd:.6f}, "
                f"stop_reason='{stop_reason}'"
            )

            return AnthropicResponse(
                content=content,
                model=model,
                stop_reason=stop_reason,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                tool_calls=tool_calls if tool_calls else None,
            )

        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            await self.emit_progress(
                stage="anthropic_api_error",
                progress=0.0,
                message=f"Anthropic API call failed: {str(e)}",
                details={"error": str(e)},
            )
            raise

    async def on_complete(self, output_data: AnthropicResponse) -> None:
        """Hook called after successful execution.

        Logs cost and token usage for tracking.

        Args:
            output_data: The generated response
        """
        logger.info(
            f"Agent '{self.name}' completed: "
            f"tokens={output_data.total_tokens}, "
            f"cost=${output_data.cost_usd:.6f}"
        )


def create_simple_prompt(
    content: str, model: str = "claude-3-sonnet-20240229", max_tokens: int = 1024, **kwargs: Any
) -> AnthropicPrompt:
    """Helper to create a simple user prompt.

    Args:
        content: The user message content
        model: Model to use
        max_tokens: Maximum tokens to generate
        **kwargs: Additional AnthropicPrompt parameters

    Returns:
        AnthropicPrompt with a single user message

    Examples:
        >>> prompt = create_simple_prompt("What is Python?")
        >>> prompt = create_simple_prompt(
        ...     "Explain AI",
        ...     model="claude-3-opus-20240229",
        ...     max_tokens=2048,
        ...     temperature=0.5
        ... )
    """
    return AnthropicPrompt(
        messages=[AnthropicMessage(role="user", content=content)],
        model=model,
        max_tokens=max_tokens,
        **kwargs,
    )


def create_system_prompt(
    system: str,
    user: str,
    model: str = "claude-3-sonnet-20240229",
    max_tokens: int = 1024,
    **kwargs: Any,
) -> AnthropicPrompt:
    """Helper to create a prompt with system and user messages.

    Args:
        system: System message content
        user: User message content
        model: Model to use
        max_tokens: Maximum tokens to generate
        **kwargs: Additional AnthropicPrompt parameters

    Returns:
        AnthropicPrompt with system and user messages

    Examples:
        >>> prompt = create_system_prompt(
        ...     "You are a helpful assistant",
        ...     "What is Python?"
        ... )
    """
    return AnthropicPrompt(
        system=system,
        messages=[AnthropicMessage(role="user", content=user)],
        model=model,
        max_tokens=max_tokens,
        **kwargs,
    )
