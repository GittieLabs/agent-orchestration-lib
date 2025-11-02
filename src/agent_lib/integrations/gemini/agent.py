"""Google Gemini agent implementation."""

from typing import Any

from loguru import logger

from agent_lib.core.agent_block import AgentBlock
from agent_lib.core.event_emitter import EventEmitter
from agent_lib.core.execution_context import ExecutionContext

from .models import GeminiPrompt, GeminiResponse
from .utils import calculate_cost, create_generation_config, estimate_tokens, parse_finish_reason


class GeminiAgent(AgentBlock[GeminiPrompt, GeminiResponse]):
    """Agent for calling Google's Gemini API.

    This agent provides a type-safe interface to Google's Gemini models,
    with automatic token counting, cost estimation, and integration
    with the agent orchestration framework.

    Attributes:
        api_key: Google API key for Gemini
        client: Gemini model instance (created lazily)
        default_model: Default model to use if not specified in prompt

    Examples:
        Basic usage:
        >>> import os
        >>> from agent_lib import ExecutionContext, EventEmitter
        >>> from agent_lib.integrations.gemini import GeminiAgent, GeminiPrompt
        >>>
        >>> context = ExecutionContext()
        >>> emitter = EventEmitter()
        >>> agent = GeminiAgent(
        ...     name="gemini-agent",
        ...     api_key=os.getenv("GOOGLE_API_KEY"),
        ...     context=context,
        ...     emitter=emitter
        ... )
        >>>
        >>> prompt = GeminiPrompt(
        ...     prompt="What is Python?",
        ...     model="gemini-pro"
        ... )
        >>> result = await agent.execute(prompt)
        >>> print(result.content)
        >>> print(f"Cost: ${result.cost_usd:.6f}")

        With custom parameters:
        >>> prompt = GeminiPrompt(
        ...     prompt="Explain quantum computing in simple terms",
        ...     model="gemini-1.5-flash",
        ...     temperature=0.7,
        ...     max_output_tokens=2048,
        ...     top_k=40
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
        default_model: str = "gemini-pro",
        **kwargs: Any,
    ):
        """Initialize the Gemini agent.

        Args:
            name: Name of the agent
            api_key: Google API key for Gemini
            context: Execution context for dependency injection
            emitter: Event emitter for observability
            default_model: Default model to use if not specified
            **kwargs: Additional arguments passed to AgentBlock
        """
        super().__init__(name, context, emitter, **kwargs)
        self.api_key = api_key
        self.default_model = default_model
        self._model_cache: dict = {}

    def get_model(self, model_name: str) -> Any:
        """Get or create a Gemini model instance.

        Args:
            model_name: Name of the model to use

        Returns:
            Gemini GenerativeModel instance

        Raises:
            ImportError: If google-generativeai package is not installed
        """
        if model_name not in self._model_cache:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "google-generativeai package is required for Gemini integration. "
                    "Install with: pip install agent-orchestration-lib[gemini]"
                )

            # Configure API key
            genai.configure(api_key=self.api_key)

            # Create model
            self._model_cache[model_name] = genai.GenerativeModel(model_name)
            logger.debug(f"Created Gemini model '{model_name}' for agent '{self.name}'")

        return self._model_cache[model_name]

    async def process(self, input_data: GeminiPrompt) -> GeminiResponse:
        """Process the input through Gemini's API.

        Args:
            input_data: Gemini prompt with text and parameters

        Returns:
            Gemini response with content, tokens, and cost

        Raises:
            Exception: If the API call fails
        """
        model_name = input_data.model or self.default_model

        # Emit progress before API call
        await self.emit_progress(
            stage="calling_gemini_api",
            progress=0.0,
            message=f"Calling Gemini API with model '{model_name}'",
            details={"model": model_name},
        )

        # Get model
        model = self.get_model(model_name)

        # Create generation config
        generation_config = create_generation_config(
            temperature=input_data.temperature,
            top_p=input_data.top_p,
            top_k=input_data.top_k,
            max_output_tokens=input_data.max_output_tokens,
            stop_sequences=input_data.stop_sequences,
            candidate_count=input_data.candidate_count,
        )

        logger.debug(
            f"Calling Gemini API with model='{model_name}', " f"config={generation_config}"
        )

        try:
            # Estimate prompt tokens
            prompt_tokens = estimate_tokens(input_data.prompt)

            # Call Gemini API (async)
            response = await model.generate_content_async(
                input_data.prompt, generation_config=generation_config
            )

            # Extract response data
            content = response.text if hasattr(response, "text") else ""
            finish_reason = parse_finish_reason(
                response.candidates[0].finish_reason.name if response.candidates else None
            )

            # Estimate completion tokens
            completion_tokens = estimate_tokens(content)
            total_tokens = prompt_tokens + completion_tokens

            # Get safety ratings if available
            safety_ratings = None
            if response.candidates and hasattr(response.candidates[0], "safety_ratings"):
                safety_ratings = [
                    {"category": rating.category.name, "probability": rating.probability.name}
                    for rating in response.candidates[0].safety_ratings
                ]

            # Calculate cost
            cost_usd = calculate_cost(prompt_tokens, completion_tokens, model_name)

            # Emit progress after successful API call
            await self.emit_progress(
                stage="gemini_api_complete",
                progress=1.0,
                message=f"Gemini API call complete: {total_tokens} tokens, ${cost_usd:.6f}",
                details={
                    "tokens": total_tokens,
                    "cost": cost_usd,
                    "finish_reason": finish_reason,
                },
            )

            logger.info(
                f"Gemini API call complete: model='{model_name}', "
                f"tokens={total_tokens}, cost=${cost_usd:.6f}, "
                f"finish_reason='{finish_reason}'"
            )

            return GeminiResponse(
                content=content,
                model=model_name,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                safety_ratings=safety_ratings,
            )

        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            await self.emit_progress(
                stage="gemini_api_error",
                progress=0.0,
                message=f"Gemini API call failed: {str(e)}",
                details={"error": str(e)},
            )
            raise

    async def on_complete(self, output_data: GeminiResponse) -> None:
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


def create_simple_prompt(content: str, model: str = "gemini-pro", **kwargs: Any) -> GeminiPrompt:
    """Helper to create a simple Gemini prompt.

    Args:
        content: The prompt text
        model: Model to use
        **kwargs: Additional GeminiPrompt parameters

    Returns:
        GeminiPrompt ready for execution

    Examples:
        >>> prompt = create_simple_prompt("What is Python?")
        >>> prompt = create_simple_prompt(
        ...     "Explain AI",
        ...     model="gemini-1.5-flash",
        ...     temperature=0.5,
        ...     max_output_tokens=1024
        ... )
    """
    return GeminiPrompt(prompt=content, model=model, **kwargs)
