"""Retry strategies for agent execution.

This module provides retry strategies for handling transient failures in agent execution.
Strategies include exponential backoff and can be composed for complex retry logic.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, Any
import asyncio
import time
from loguru import logger


class RetryStrategy(ABC):
    """Base class for retry strategies.

    A retry strategy determines:
    1. Whether an error should be retried
    2. How long to wait before retrying
    3. When to give up retrying

    Example:
        ```python
        strategy = ExponentialBackoffRetry(max_attempts=3)

        attempt = 1
        while attempt <= strategy.max_attempts:
            try:
                result = await agent.execute(input_data)
                break
            except Exception as e:
                if strategy.should_retry(e, attempt):
                    wait_time = strategy.get_wait_time(attempt)
                    await asyncio.sleep(wait_time)
                    attempt += 1
                else:
                    raise
        ```
    """

    def __init__(self, max_attempts: int = 3):
        """Initialize retry strategy.

        Args:
            max_attempts: Maximum number of attempts (including initial attempt)
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.max_attempts = max_attempts

    @abstractmethod
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if an error should be retried.

        Args:
            error: The exception that was raised
            attempt: The current attempt number (1-indexed)

        Returns:
            True if the error should be retried, False otherwise
        """
        pass

    @abstractmethod
    def get_wait_time(self, attempt: int) -> float:
        """Calculate wait time before the next retry.

        Args:
            attempt: The current attempt number (1-indexed)

        Returns:
            Wait time in seconds before next retry
        """
        pass

    def __repr__(self) -> str:
        """String representation of the retry strategy."""
        return f"{self.__class__.__name__}(max_attempts={self.max_attempts})"


class NoRetry(RetryStrategy):
    """Retry strategy that never retries.

    Useful as a default or to explicitly disable retries.
    """

    def __init__(self):
        """Initialize no-retry strategy."""
        super().__init__(max_attempts=1)

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Never retry."""
        return False

    def get_wait_time(self, attempt: int) -> float:
        """No wait time since we never retry."""
        return 0.0


class ExponentialBackoffRetry(RetryStrategy):
    """Exponential backoff retry strategy.

    Wait time doubles with each attempt: base_delay * (2 ** (attempt - 1))

    Example wait times with base_delay=1.0:
    - Attempt 1 fails: wait 1.0s
    - Attempt 2 fails: wait 2.0s
    - Attempt 3 fails: wait 4.0s

    With max_delay cap:
    - Prevents unbounded waiting on later attempts
    - Useful for preventing excessive delays
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: Optional[float] = None,
        exponential_base: float = 2.0,
        retry_on: Optional[Callable[[Exception], bool]] = None,
    ):
        """Initialize exponential backoff retry strategy.

        Args:
            max_attempts: Maximum number of attempts
            base_delay: Base delay in seconds for first retry
            max_delay: Maximum delay in seconds (None for no cap)
            exponential_base: Base for exponential calculation (default: 2.0)
            retry_on: Optional function to determine if error should be retried
        """
        super().__init__(max_attempts=max_attempts)
        if base_delay <= 0:
            raise ValueError("base_delay must be positive")
        if exponential_base <= 1:
            raise ValueError("exponential_base must be greater than 1")
        if max_delay is not None and max_delay < base_delay:
            raise ValueError("max_delay must be >= base_delay")

        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retry_on = retry_on

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if error should be retried.

        Args:
            error: The exception that occurred
            attempt: Current attempt number

        Returns:
            True if should retry (not exceeded max and error is retryable)
        """
        # Check if we've exceeded max attempts
        if attempt >= self.max_attempts:
            return False

        # If custom retry predicate provided, use it
        if self.retry_on is not None:
            return self.retry_on(error)

        # Default: retry all errors
        return True

    def get_wait_time(self, attempt: int) -> float:
        """Calculate exponential backoff wait time.

        Args:
            attempt: Current attempt number (1-indexed)

        Returns:
            Wait time in seconds
        """
        # Calculate exponential delay
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))

        # Apply max_delay cap if set
        if self.max_delay is not None:
            delay = min(delay, self.max_delay)

        return delay

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ExponentialBackoffRetry("
            f"max_attempts={self.max_attempts}, "
            f"base_delay={self.base_delay}, "
            f"max_delay={self.max_delay})"
        )


class FixedDelayRetry(RetryStrategy):
    """Fixed delay retry strategy.

    Waits a constant amount of time between retries.
    Simple and predictable, useful for rate limiting.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        delay: float = 1.0,
        retry_on: Optional[Callable[[Exception], bool]] = None,
    ):
        """Initialize fixed delay retry strategy.

        Args:
            max_attempts: Maximum number of attempts
            delay: Fixed delay in seconds between retries
            retry_on: Optional function to determine if error should be retried
        """
        super().__init__(max_attempts=max_attempts)
        if delay < 0:
            raise ValueError("delay must be non-negative")
        self.delay = delay
        self.retry_on = retry_on

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if error should be retried."""
        if attempt >= self.max_attempts:
            return False

        if self.retry_on is not None:
            return self.retry_on(error)

        return True

    def get_wait_time(self, attempt: int) -> float:
        """Return fixed delay."""
        return self.delay

    def __repr__(self) -> str:
        """String representation."""
        return f"FixedDelayRetry(max_attempts={self.max_attempts}, delay={self.delay})"


class LinearBackoffRetry(RetryStrategy):
    """Linear backoff retry strategy.

    Wait time increases linearly: base_delay + (increment * (attempt - 1))

    Example with base_delay=1.0, increment=0.5:
    - Attempt 1 fails: wait 1.0s
    - Attempt 2 fails: wait 1.5s
    - Attempt 3 fails: wait 2.0s
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        increment: float = 0.5,
        max_delay: Optional[float] = None,
        retry_on: Optional[Callable[[Exception], bool]] = None,
    ):
        """Initialize linear backoff retry strategy.

        Args:
            max_attempts: Maximum number of attempts
            base_delay: Base delay in seconds for first retry
            increment: Increment in seconds for each subsequent retry
            max_delay: Maximum delay in seconds (None for no cap)
            retry_on: Optional function to determine if error should be retried
        """
        super().__init__(max_attempts=max_attempts)
        if base_delay < 0:
            raise ValueError("base_delay must be non-negative")
        if increment < 0:
            raise ValueError("increment must be non-negative")
        if max_delay is not None and max_delay < base_delay:
            raise ValueError("max_delay must be >= base_delay")

        self.base_delay = base_delay
        self.increment = increment
        self.max_delay = max_delay
        self.retry_on = retry_on

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if error should be retried."""
        if attempt >= self.max_attempts:
            return False

        if self.retry_on is not None:
            return self.retry_on(error)

        return True

    def get_wait_time(self, attempt: int) -> float:
        """Calculate linear backoff wait time."""
        delay = self.base_delay + (self.increment * (attempt - 1))

        if self.max_delay is not None:
            delay = min(delay, self.max_delay)

        return delay

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LinearBackoffRetry("
            f"max_attempts={self.max_attempts}, "
            f"base_delay={self.base_delay}, "
            f"increment={self.increment})"
        )


# Helper functions for common retry predicates
def retry_on_exception_type(*exception_types: type) -> Callable[[Exception], bool]:
    """Create a predicate that retries on specific exception types.

    Args:
        *exception_types: Exception types to retry on

    Returns:
        Predicate function for retry_on parameter

    Example:
        ```python
        strategy = ExponentialBackoffRetry(
            retry_on=retry_on_exception_type(ValueError, TypeError)
        )
        ```
    """

    def predicate(error: Exception) -> bool:
        return isinstance(error, exception_types)

    return predicate


def retry_on_error_message(substring: str) -> Callable[[Exception], bool]:
    """Create a predicate that retries when error message contains substring.

    Args:
        substring: Substring to look for in error message

    Returns:
        Predicate function for retry_on parameter

    Example:
        ```python
        strategy = ExponentialBackoffRetry(
            retry_on=retry_on_error_message("rate limit")
        )
        ```
    """

    def predicate(error: Exception) -> bool:
        return substring.lower() in str(error).lower()

    return predicate
