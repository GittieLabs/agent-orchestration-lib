"""Unit tests for RetryStrategy classes."""

import pytest
from agent_lib.retry import (
    RetryStrategy,
    NoRetry,
    ExponentialBackoffRetry,
    FixedDelayRetry,
    LinearBackoffRetry,
    retry_on_exception_type,
    retry_on_error_message,
)


class TestRetryStrategyBase:
    """Test base RetryStrategy class."""

    def test_cannot_instantiate_base_class(self):
        """Test that base RetryStrategy cannot be instantiated."""
        with pytest.raises(TypeError):
            RetryStrategy(max_attempts=3)

    def test_invalid_max_attempts(self):
        """Test that max_attempts must be at least 1."""
        with pytest.raises(ValueError, match="max_attempts must be at least 1"):
            ExponentialBackoffRetry(max_attempts=0)


class TestNoRetry:
    """Test NoRetry strategy."""

    def test_create_no_retry(self):
        """Test creating NoRetry strategy."""
        strategy = NoRetry()
        assert strategy.max_attempts == 1

    def test_never_retries(self):
        """Test that NoRetry never retries."""
        strategy = NoRetry()
        error = ValueError("test error")

        assert not strategy.should_retry(error, 1)

    def test_wait_time_is_zero(self):
        """Test that wait time is always 0."""
        strategy = NoRetry()
        assert strategy.get_wait_time(1) == 0.0


class TestExponentialBackoffRetry:
    """Test ExponentialBackoffRetry strategy."""

    def test_create_exponential_backoff(self):
        """Test creating exponential backoff strategy."""
        strategy = ExponentialBackoffRetry(max_attempts=3)
        assert strategy.max_attempts == 3
        assert strategy.base_delay == 1.0
        assert strategy.max_delay is None

    def test_exponential_backoff_calculation(self):
        """Test exponential backoff delay calculation."""
        strategy = ExponentialBackoffRetry(max_attempts=5, base_delay=1.0)

        # 1.0 * (2 ** 0) = 1.0
        assert strategy.get_wait_time(1) == 1.0
        # 1.0 * (2 ** 1) = 2.0
        assert strategy.get_wait_time(2) == 2.0
        # 1.0 * (2 ** 2) = 4.0
        assert strategy.get_wait_time(3) == 4.0
        # 1.0 * (2 ** 3) = 8.0
        assert strategy.get_wait_time(4) == 8.0

    def test_exponential_backoff_with_base(self):
        """Test exponential backoff with custom base."""
        strategy = ExponentialBackoffRetry(
            max_attempts=3, base_delay=2.0, exponential_base=3.0
        )

        # 2.0 * (3 ** 0) = 2.0
        assert strategy.get_wait_time(1) == 2.0
        # 2.0 * (3 ** 1) = 6.0
        assert strategy.get_wait_time(2) == 6.0
        # 2.0 * (3 ** 2) = 18.0
        assert strategy.get_wait_time(3) == 18.0

    def test_exponential_backoff_max_delay(self):
        """Test that max_delay caps the wait time."""
        strategy = ExponentialBackoffRetry(
            max_attempts=5, base_delay=1.0, max_delay=5.0
        )

        assert strategy.get_wait_time(1) == 1.0  # 1.0
        assert strategy.get_wait_time(2) == 2.0  # 2.0
        assert strategy.get_wait_time(3) == 4.0  # 4.0
        assert strategy.get_wait_time(4) == 5.0  # capped at 5.0 (would be 8.0)
        assert strategy.get_wait_time(5) == 5.0  # capped at 5.0 (would be 16.0)

    def test_should_retry_within_max_attempts(self):
        """Test retry decision within max attempts."""
        strategy = ExponentialBackoffRetry(max_attempts=3)
        error = ValueError("test error")

        assert strategy.should_retry(error, 1) is True
        assert strategy.should_retry(error, 2) is True
        assert strategy.should_retry(error, 3) is False  # At max, don't retry

    def test_custom_retry_predicate(self):
        """Test custom retry predicate."""

        def retry_only_value_errors(error: Exception) -> bool:
            return isinstance(error, ValueError)

        strategy = ExponentialBackoffRetry(max_attempts=3, retry_on=retry_only_value_errors)

        # Should retry ValueError
        assert strategy.should_retry(ValueError("test"), 1) is True

        # Should not retry TypeError
        assert strategy.should_retry(TypeError("test"), 1) is False

    def test_invalid_base_delay(self):
        """Test that base_delay must be positive."""
        with pytest.raises(ValueError, match="base_delay must be positive"):
            ExponentialBackoffRetry(base_delay=0)

        with pytest.raises(ValueError, match="base_delay must be positive"):
            ExponentialBackoffRetry(base_delay=-1)

    def test_invalid_exponential_base(self):
        """Test that exponential_base must be > 1."""
        with pytest.raises(ValueError, match="exponential_base must be greater than 1"):
            ExponentialBackoffRetry(exponential_base=1.0)

    def test_invalid_max_delay(self):
        """Test that max_delay must be >= base_delay."""
        with pytest.raises(ValueError, match="max_delay must be >= base_delay"):
            ExponentialBackoffRetry(base_delay=2.0, max_delay=1.0)

    def test_repr(self):
        """Test string representation."""
        strategy = ExponentialBackoffRetry(max_attempts=3, base_delay=2.0, max_delay=10.0)
        repr_str = repr(strategy)
        assert "ExponentialBackoffRetry" in repr_str
        assert "max_attempts=3" in repr_str
        assert "base_delay=2.0" in repr_str
        assert "max_delay=10.0" in repr_str


class TestFixedDelayRetry:
    """Test FixedDelayRetry strategy."""

    def test_create_fixed_delay(self):
        """Test creating fixed delay strategy."""
        strategy = FixedDelayRetry(max_attempts=3, delay=2.0)
        assert strategy.max_attempts == 3
        assert strategy.delay == 2.0

    def test_fixed_delay_calculation(self):
        """Test that delay is constant."""
        strategy = FixedDelayRetry(max_attempts=5, delay=1.5)

        assert strategy.get_wait_time(1) == 1.5
        assert strategy.get_wait_time(2) == 1.5
        assert strategy.get_wait_time(3) == 1.5
        assert strategy.get_wait_time(4) == 1.5

    def test_should_retry(self):
        """Test retry decision."""
        strategy = FixedDelayRetry(max_attempts=3, delay=1.0)
        error = ValueError("test")

        assert strategy.should_retry(error, 1) is True
        assert strategy.should_retry(error, 2) is True
        assert strategy.should_retry(error, 3) is False

    def test_custom_retry_predicate(self):
        """Test custom retry predicate."""

        def retry_timeouts_only(error: Exception) -> bool:
            return isinstance(error, TimeoutError)

        strategy = FixedDelayRetry(max_attempts=3, delay=1.0, retry_on=retry_timeouts_only)

        assert strategy.should_retry(TimeoutError("test"), 1) is True
        assert strategy.should_retry(ValueError("test"), 1) is False

    def test_invalid_delay(self):
        """Test that delay must be non-negative."""
        with pytest.raises(ValueError, match="delay must be non-negative"):
            FixedDelayRetry(delay=-1.0)

    def test_repr(self):
        """Test string representation."""
        strategy = FixedDelayRetry(max_attempts=5, delay=2.5)
        repr_str = repr(strategy)
        assert "FixedDelayRetry" in repr_str
        assert "max_attempts=5" in repr_str
        assert "delay=2.5" in repr_str


class TestLinearBackoffRetry:
    """Test LinearBackoffRetry strategy."""

    def test_create_linear_backoff(self):
        """Test creating linear backoff strategy."""
        strategy = LinearBackoffRetry(max_attempts=3, base_delay=1.0, increment=0.5)
        assert strategy.max_attempts == 3
        assert strategy.base_delay == 1.0
        assert strategy.increment == 0.5

    def test_linear_backoff_calculation(self):
        """Test linear backoff delay calculation."""
        strategy = LinearBackoffRetry(max_attempts=5, base_delay=1.0, increment=0.5)

        # 1.0 + (0.5 * 0) = 1.0
        assert strategy.get_wait_time(1) == 1.0
        # 1.0 + (0.5 * 1) = 1.5
        assert strategy.get_wait_time(2) == 1.5
        # 1.0 + (0.5 * 2) = 2.0
        assert strategy.get_wait_time(3) == 2.0
        # 1.0 + (0.5 * 3) = 2.5
        assert strategy.get_wait_time(4) == 2.5

    def test_linear_backoff_max_delay(self):
        """Test that max_delay caps the wait time."""
        strategy = LinearBackoffRetry(
            max_attempts=5, base_delay=1.0, increment=1.0, max_delay=3.0
        )

        assert strategy.get_wait_time(1) == 1.0  # 1.0
        assert strategy.get_wait_time(2) == 2.0  # 2.0
        assert strategy.get_wait_time(3) == 3.0  # capped (would be 3.0)
        assert strategy.get_wait_time(4) == 3.0  # capped (would be 4.0)
        assert strategy.get_wait_time(5) == 3.0  # capped (would be 5.0)

    def test_should_retry(self):
        """Test retry decision."""
        strategy = LinearBackoffRetry(max_attempts=3, base_delay=1.0, increment=0.5)
        error = ValueError("test")

        assert strategy.should_retry(error, 1) is True
        assert strategy.should_retry(error, 2) is True
        assert strategy.should_retry(error, 3) is False

    def test_custom_retry_predicate(self):
        """Test custom retry predicate."""

        def retry_io_errors(error: Exception) -> bool:
            return isinstance(error, IOError)

        strategy = LinearBackoffRetry(
            max_attempts=3, base_delay=1.0, increment=0.5, retry_on=retry_io_errors
        )

        assert strategy.should_retry(IOError("test"), 1) is True
        assert strategy.should_retry(ValueError("test"), 1) is False

    def test_invalid_base_delay(self):
        """Test that base_delay must be non-negative."""
        with pytest.raises(ValueError, match="base_delay must be non-negative"):
            LinearBackoffRetry(base_delay=-1.0, increment=0.5)

    def test_invalid_increment(self):
        """Test that increment must be non-negative."""
        with pytest.raises(ValueError, match="increment must be non-negative"):
            LinearBackoffRetry(base_delay=1.0, increment=-0.5)

    def test_invalid_max_delay(self):
        """Test that max_delay must be >= base_delay."""
        with pytest.raises(ValueError, match="max_delay must be >= base_delay"):
            LinearBackoffRetry(base_delay=2.0, increment=0.5, max_delay=1.0)

    def test_repr(self):
        """Test string representation."""
        strategy = LinearBackoffRetry(max_attempts=4, base_delay=1.5, increment=0.75)
        repr_str = repr(strategy)
        assert "LinearBackoffRetry" in repr_str
        assert "max_attempts=4" in repr_str
        assert "base_delay=1.5" in repr_str
        assert "increment=0.75" in repr_str


class TestRetryPredicates:
    """Test retry predicate helper functions."""

    def test_retry_on_exception_type_single(self):
        """Test retry_on_exception_type with single exception."""
        predicate = retry_on_exception_type(ValueError)

        assert predicate(ValueError("test")) is True
        assert predicate(TypeError("test")) is False

    def test_retry_on_exception_type_multiple(self):
        """Test retry_on_exception_type with multiple exceptions."""
        predicate = retry_on_exception_type(ValueError, TypeError, KeyError)

        assert predicate(ValueError("test")) is True
        assert predicate(TypeError("test")) is True
        assert predicate(KeyError("test")) is True
        assert predicate(RuntimeError("test")) is False

    def test_retry_on_exception_type_inheritance(self):
        """Test retry_on_exception_type with exception inheritance."""
        predicate = retry_on_exception_type(Exception)

        # All exceptions inherit from Exception
        assert predicate(ValueError("test")) is True
        assert predicate(TypeError("test")) is True
        assert predicate(RuntimeError("test")) is True

    def test_retry_on_error_message(self):
        """Test retry_on_error_message."""
        predicate = retry_on_error_message("rate limit")

        assert predicate(ValueError("rate limit exceeded")) is True
        assert predicate(RuntimeError("Rate Limit Error")) is True  # case-insensitive
        assert predicate(ValueError("timeout error")) is False

    def test_retry_on_error_message_case_insensitive(self):
        """Test that retry_on_error_message is case-insensitive."""
        predicate = retry_on_error_message("timeout")

        assert predicate(ValueError("TIMEOUT")) is True
        assert predicate(ValueError("Timeout")) is True
        assert predicate(ValueError("timeout")) is True
        assert predicate(ValueError("TimeOut Error")) is True


class TestRetryStrategyUsagePatterns:
    """Test common usage patterns."""

    def test_retry_loop_pattern(self):
        """Test typical retry loop pattern."""
        strategy = ExponentialBackoffRetry(max_attempts=3)
        attempt = 1
        errors = []

        while attempt <= strategy.max_attempts:
            try:
                if attempt < 3:
                    raise ValueError(f"Attempt {attempt} failed")
                # Success on attempt 3
                break
            except Exception as e:
                errors.append(str(e))
                if strategy.should_retry(e, attempt):
                    wait_time = strategy.get_wait_time(attempt)
                    assert wait_time > 0
                    attempt += 1
                else:
                    raise

        assert len(errors) == 2
        assert attempt == 3

    def test_rate_limit_retry_pattern(self):
        """Test rate limit retry pattern."""
        # Use fixed delay for rate limits
        strategy = FixedDelayRetry(
            max_attempts=3, delay=1.0, retry_on=retry_on_error_message("rate limit")
        )

        # Should retry rate limit errors
        assert strategy.should_retry(ValueError("rate limit exceeded"), 1) is True
        assert strategy.get_wait_time(1) == 1.0

        # Should not retry other errors
        assert strategy.should_retry(ValueError("other error"), 1) is False

    def test_transient_error_retry_pattern(self):
        """Test pattern for retrying transient errors."""
        # Retry specific exception types with exponential backoff
        strategy = ExponentialBackoffRetry(
            max_attempts=5,
            base_delay=0.5,
            max_delay=5.0,
            retry_on=retry_on_exception_type(TimeoutError, ConnectionError),
        )

        # Should retry transient errors
        assert strategy.should_retry(TimeoutError("timeout"), 1) is True
        assert strategy.should_retry(ConnectionError("connection lost"), 1) is True

        # Should not retry permanent errors
        assert strategy.should_retry(ValueError("bad input"), 1) is False
