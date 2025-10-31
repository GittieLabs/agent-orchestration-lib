"""OpenAI agent implementation."""

from typing import Any, Dict, Optional

from loguru import logger

from agent_lib.core.agent_block import AgentBlock
from agent_lib.core.event_emitter import EventEmitter
from agent_lib.core.execution_context import ExecutionContext

from .models import OpenAIMessage, OpenAIPrompt, OpenAIResponse
from .utils import calculate_cost


class OpenAIAgent(AgentBlock[OpenAIPrompt, OpenAIResponse]):
    """Agent for calling OpenAI's chat completion API.

    This agent provides a type-safe interface to OpenAI's GPT models,
    with automatic token counting, cost estimation, and integration
    with the agent orchestration framework.

    Attributes:
        api_key: OpenAI API key
        client: OpenAI async client (created lazily)
        default_model: Default model to use if not specified in prompt

    Examples:
        Basic usage:
        >>> from agent_lib import ExecutionContext, EventEmitter
        >>> from agent_lib.integrations.openai import OpenAIAgent, OpenAIPrompt, OpenAIMessage
        >>>
        >>> context = ExecutionContext()
        >>> emitter = EventEmitter()
        >>> agent = OpenAIAgent(
        ...     name="gpt4-agent",
        ...     api_key="sk-...",
        ...     context=context,
        ...     emitter=emitter
        ... )
        >>>
        >>> prompt = OpenAIPrompt(
        ...     messages=[OpenAIMessage(role="user", content="What is Python?")],
        ...     model="gpt-4"
        ... )
        >>> result = await agent.execute(prompt)
        >>> print(result.content)
        >>> print(f"Cost: ${result.cost_usd:.4f}")

        With system message:
        >>> prompt = OpenAIPrompt(
        ...     messages=[
        ...         OpenAIMessage(role="system", content="You are a helpful coding assistant"),
        ...         OpenAIMessage(role="user", content="How do I reverse a list in Python?")
        ...     ],
        ...     model="gpt-4",
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
        default_model: str = "gpt-4",
        **kwargs: Any,
    ):
        """Initialize the OpenAI agent.

        Args:
            name: Name of the agent
            api_key: OpenAI API key
            context: Execution context for dependency injection
            emitter: Event emitter for observability
            default_model: Default model to use if not specified
            **kwargs: Additional arguments passed to AgentBlock
        """
        super().__init__(name, context, emitter, **kwargs)
        self.api_key = api_key
        self.default_model = default_model
        self._client: Optional[Any] = None

    @property
    def client(self) -> Any:
        """Get or create the OpenAI async client.

        Returns:
            OpenAI AsyncOpenAI client instance

        Raises:
            ImportError: If openai package is not installed
        """
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "openai package is required for OpenAI integration. "
                    "Install with: pip install agent-orchestration-lib[openai]"
                ) from e

            self._client = AsyncOpenAI(api_key=self.api_key)
            logger.debug(f"Created OpenAI client for agent '{self.name}'")

        return self._client

    async def process(self, input_data: OpenAIPrompt) -> OpenAIResponse:
        """Process the input through OpenAI's API.

        Args:
            input_data: OpenAI prompt with messages and parameters

        Returns:
            OpenAI response with content, tokens, and cost

        Raises:
            Exception: If the API call fails
        """
        model = input_data.model or self.default_model

        # Emit progress before API call
        self.emit_progress(
            "calling_openai_api", {"model": model, "messages_count": len(input_data.messages)}
        )

        # Convert Pydantic models to dicts for API
        messages_dict = [msg.model_dump() for msg in input_data.messages]

        # Build API parameters
        api_params: Dict[str, Any] = {
            "model": model,
            "messages": messages_dict,
            "temperature": input_data.temperature,
            "frequency_penalty": input_data.frequency_penalty,
            "presence_penalty": input_data.presence_penalty,
        }

        # Add optional parameters
        if input_data.max_tokens is not None:
            api_params["max_tokens"] = input_data.max_tokens
        if input_data.top_p is not None:
            api_params["top_p"] = input_data.top_p
        if input_data.stop is not None:
            api_params["stop"] = input_data.stop
        if input_data.response_format is not None:
            api_params["response_format"] = input_data.response_format

        logger.debug(
            f"Calling OpenAI API with model='{model}', "
            f"messages={len(messages_dict)}, temp={input_data.temperature}"
        )

        try:
            # Call OpenAI API
            response = await self.client.chat.completions.create(**api_params)

            # Extract response data
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

            # Calculate cost
            cost_usd = calculate_cost(prompt_tokens, completion_tokens, model)

            # Emit progress after successful API call
            self.emit_progress(
                "openai_api_complete",
                {
                    "tokens": total_tokens,
                    "cost": cost_usd,
                    "finish_reason": finish_reason,
                },
            )

            logger.info(
                f"OpenAI API call complete: model='{model}', "
                f"tokens={total_tokens}, cost=${cost_usd:.4f}, "
                f"finish_reason='{finish_reason}'"
            )

            return OpenAIResponse(
                content=content,
                model=model,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
            )

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            self.emit_progress("openai_api_error", {"error": str(e)})
            raise

    async def on_complete(self, input_data: OpenAIPrompt, output_data: OpenAIResponse) -> None:
        """Hook called after successful execution.

        Logs cost and token usage for tracking.

        Args:
            input_data: The input prompt
            output_data: The generated response
        """
        logger.info(
            f"Agent '{self.name}' completed: "
            f"tokens={output_data.total_tokens}, "
            f"cost=${output_data.cost_usd:.6f}"
        )


def create_simple_prompt(content: str, model: str = "gpt-4", **kwargs: Any) -> OpenAIPrompt:
    """Helper to create a simple user prompt.

    Args:
        content: The user message content
        model: Model to use
        **kwargs: Additional OpenAIPrompt parameters

    Returns:
        OpenAIPrompt with a single user message

    Examples:
        >>> prompt = create_simple_prompt("What is Python?")
        >>> prompt = create_simple_prompt("Explain AI", model="gpt-3.5-turbo", temperature=0.5)
    """
    return OpenAIPrompt(
        messages=[OpenAIMessage(role="user", content=content)], model=model, **kwargs
    )


def create_system_prompt(
    system: str, user: str, model: str = "gpt-4", **kwargs: Any
) -> OpenAIPrompt:
    """Helper to create a prompt with system and user messages.

    Args:
        system: System message content
        user: User message content
        model: Model to use
        **kwargs: Additional OpenAIPrompt parameters

    Returns:
        OpenAIPrompt with system and user messages

    Examples:
        >>> prompt = create_system_prompt(
        ...     "You are a helpful assistant",
        ...     "What is Python?"
        ... )
    """
    return OpenAIPrompt(
        messages=[
            OpenAIMessage(role="system", content=system),
            OpenAIMessage(role="user", content=user),
        ],
        model=model,
        **kwargs,
    )
