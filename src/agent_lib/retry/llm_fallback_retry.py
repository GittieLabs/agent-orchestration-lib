"""LLMFallbackRetry - Retry strategy that falls back to alternative LLM models.

This module provides the LLMFallbackRetry class that enables automatic fallback
to alternative LLM models when the primary model fails, useful for building
resilient AI systems.
"""

from typing import List, Optional, Callable, Any
from loguru import logger

from .retry_strategy import RetryStrategy


class LLMFallbackRetry(RetryStrategy):
    """Retry strategy that falls back to alternative LLM models on failure.

    This strategy maintains a list of LLM models to try in sequence. When the
    primary model fails, it automatically retries with the next model in the list.
    Useful for building resilient AI systems that can gracefully degrade to
    alternative models.

    Example:
        ```python
        strategy = LLMFallbackRetry(
            primary_model="gpt-4",
            fallback_models=["claude-3-sonnet", "gemini-pro"],
            max_attempts=3,
            base_delay=1.0
        )

        agent = MyLLMAgent(
            name="resilient_agent",
            context=context,
            emitter=emitter,
            retry_strategy=strategy
        )

        # In agent's process method:
        model = strategy.get_current_model()
        result = await llm_manager.generate(model=model, prompt=prompt)
        ```
    """

    def __init__(
        self,
        primary_model: str,
        fallback_models: List[str],
        max_attempts: Optional[int] = None,
        base_delay: float = 1.0,
        retry_on: Optional[Callable[[Exception], bool]] = None,
    ):
        """Initialize LLM fallback retry strategy.

        Args:
            primary_model: Primary LLM model to try first
            fallback_models: List of fallback models to try in sequence
            max_attempts: Maximum attempts (defaults to number of models)
            base_delay: Base delay in seconds between retries
            retry_on: Optional predicate to determine if error should be retried
        """
        # Build complete model list
        self.models = [primary_model] + fallback_models
        self.model_count = len(self.models)

        # Default max_attempts to number of models
        if max_attempts is None:
            max_attempts = self.model_count

        super().__init__(max_attempts=max_attempts)

        if base_delay < 0:
            raise ValueError("base_delay must be non-negative")

        self.base_delay = base_delay
        self.retry_on = retry_on
        self._current_model_index = 0

        logger.debug(
            f"Initialized LLMFallbackRetry with {self.model_count} models: "
            f"{', '.join(self.models)}"
        )

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if error should be retried with next model.

        Args:
            error: The exception that occurred
            attempt: Current attempt number (1-indexed)

        Returns:
            True if should retry (have more models to try and within max_attempts)
        """
        # Check if we've exceeded max attempts
        if attempt >= self.max_attempts:
            logger.debug(
                f"Max attempts ({self.max_attempts}) reached, not retrying"
            )
            return False

        # Check if we've exhausted all models
        if self._current_model_index >= self.model_count:
            logger.debug(
                f"All {self.model_count} models exhausted, not retrying"
            )
            return False

        # If custom retry predicate provided, use it
        if self.retry_on is not None:
            should_retry = self.retry_on(error)
            if not should_retry:
                logger.debug(
                    f"Custom retry predicate returned False for {type(error).__name__}"
                )
            return should_retry

        # Default: retry all errors
        return True

    def get_wait_time(self, attempt: int) -> float:
        """Get wait time before next retry.

        Args:
            attempt: Current attempt number (1-indexed)

        Returns:
            Base delay in seconds (constant for all attempts)
        """
        return self.base_delay

    def get_current_model(self) -> str:
        """Get the current model to use for this attempt.

        Returns:
            Model name string

        Raises:
            IndexError: If called when all models are exhausted
        """
        if self._current_model_index >= self.model_count:
            raise IndexError(
                f"All {self.model_count} models exhausted, "
                f"current_index={self._current_model_index}"
            )

        model = self.models[self._current_model_index]
        logger.debug(
            f"Current model (index {self._current_model_index}): {model}"
        )
        return model

    def advance_to_next_model(self) -> Optional[str]:
        """Advance to the next model in the fallback sequence.

        Returns:
            Next model name if available, None if all models exhausted
        """
        self._current_model_index += 1

        if self._current_model_index >= self.model_count:
            logger.info(
                f"All {self.model_count} models exhausted, no more fallbacks"
            )
            return None

        next_model = self.models[self._current_model_index]
        logger.info(
            f"Advancing to fallback model (index {self._current_model_index}): {next_model}"
        )
        return next_model

    def reset(self) -> None:
        """Reset the strategy to start from the primary model again."""
        logger.debug("Resetting LLMFallbackRetry to primary model")
        self._current_model_index = 0

    def get_remaining_models(self) -> List[str]:
        """Get list of models that haven't been tried yet.

        Returns:
            List of remaining model names
        """
        return self.models[self._current_model_index + 1:]

    def __repr__(self) -> str:
        """String representation of the strategy."""
        return (
            f"LLMFallbackRetry("
            f"models={self.model_count}, "
            f"max_attempts={self.max_attempts}, "
            f"current_index={self._current_model_index})"
        )
