"""Assertion utilities for agent testing."""

from typing import Any, Optional, Union
from agent_lib.core import ExecutionContext, EventEmitter
from .mock_agents import MockAgent, AgentSpy


class AssertionError(Exception):
    """Raised when an assertion fails."""
    pass


def assert_agent_called(
    agent: Union[MockAgent, AgentSpy],
    times: Optional[int] = None,
    at_least: Optional[int] = None,
    at_most: Optional[int] = None,
) -> None:
    """Assert that an agent was called a specific number of times.

    Args:
        agent: MockAgent or AgentSpy to check
        times: Exact number of times (optional)
        at_least: Minimum number of times (optional)
        at_most: Maximum number of times (optional)

    Raises:
        AssertionError: If the assertion fails
        ValueError: If invalid arguments are provided

    Example:
        ```python
        mock = MockAgent("test", return_value=42)
        await mock.execute({"input": "data"})

        assert_agent_called(mock, times=1)
        assert_agent_called(mock, at_least=1)
        ```
    """
    if not isinstance(agent, (MockAgent, AgentSpy)):
        raise ValueError(
            f"Expected MockAgent or AgentSpy, got {type(agent).__name__}"
        )

    call_count = agent.call_count

    if times is not None:
        if call_count != times:
            raise AssertionError(
                f"Expected agent '{agent.name}' to be called {times} time(s), "
                f"but was called {call_count} time(s)"
            )

    if at_least is not None:
        if call_count < at_least:
            raise AssertionError(
                f"Expected agent '{agent.name}' to be called at least "
                f"{at_least} time(s), but was called {call_count} time(s)"
            )

    if at_most is not None:
        if call_count > at_most:
            raise AssertionError(
                f"Expected agent '{agent.name}' to be called at most "
                f"{at_most} time(s), but was called {call_count} time(s)"
            )

    # If no constraints provided, just check that it was called at least once
    if times is None and at_least is None and at_most is None:
        if call_count == 0:
            raise AssertionError(
                f"Expected agent '{agent.name}' to be called at least once, "
                f"but it was never called"
            )


def assert_event_emitted(
    emitter: EventEmitter,
    event_type: str,
    times: Optional[int] = None,
    with_data: Optional[dict] = None,
) -> None:
    """Assert that an event was emitted.

    This function requires that the emitter was created with
    `create_test_emitter(capture_events=True)`.

    Args:
        emitter: EventEmitter to check
        event_type: Type of event to check for
        times: Exact number of times event should have been emitted (optional)
        with_data: Expected event data (optional, checks subset match)

    Raises:
        AssertionError: If the assertion fails
        AttributeError: If emitter doesn't have event capture enabled

    Example:
        ```python
        from agent_lib.testing import create_test_emitter

        emitter = create_test_emitter()
        await emitter.emit("test_event", {"key": "value"})

        assert_event_emitted(emitter, "test_event", times=1)
        assert_event_emitted(emitter, "test_event", with_data={"key": "value"})
        ```
    """
    if not hasattr(emitter, 'get_captured_events'):
        raise AttributeError(
            "EventEmitter does not have event capture enabled. "
            "Use create_test_emitter(capture_events=True) to enable."
        )

    captured = emitter.get_captured_events()
    matching_events = [
        e for e in captured
        if e["type"] == event_type
    ]

    if with_data is not None:
        # Filter to events that contain the expected data
        matching_events = [
            e for e in matching_events
            if _dict_contains(e["data"], with_data)
        ]

    count = len(matching_events)

    if times is not None:
        if count != times:
            raise AssertionError(
                f"Expected event '{event_type}' to be emitted {times} time(s), "
                f"but was emitted {count} time(s)"
            )
    else:
        # If no count specified, just check it was emitted at least once
        if count == 0:
            data_msg = f" with data {with_data}" if with_data else ""
            raise AssertionError(
                f"Expected event '{event_type}'{data_msg} to be emitted, "
                f"but it was not found"
            )


def assert_context_has(
    context: ExecutionContext,
    key: str,
    value: Optional[Any] = None,
) -> None:
    """Assert that execution context has a specific key/value.

    Args:
        context: ExecutionContext to check
        key: Key to check for
        value: Expected value (optional, if not provided just checks key exists)

    Raises:
        AssertionError: If the assertion fails

    Example:
        ```python
        from agent_lib.testing import create_test_context

        context = create_test_context({"user_id": 123})

        assert_context_has(context, "user_id")
        assert_context_has(context, "user_id", 123)
        ```
    """
    if not context.has(key):
        raise AssertionError(
            f"Expected context to have key '{key}', but it was not found"
        )

    if value is not None:
        actual_value = context.get(key)
        if actual_value != value:
            raise AssertionError(
                f"Expected context['{key}'] to be {value!r}, "
                f"but got {actual_value!r}"
            )


def _dict_contains(container: dict, subset: dict) -> bool:
    """Check if container dict contains all key-value pairs from subset.

    Args:
        container: The larger dictionary
        subset: The subset to check for

    Returns:
        True if all subset key-value pairs are in container
    """
    for key, value in subset.items():
        if key not in container:
            return False
        if isinstance(value, dict) and isinstance(container[key], dict):
            if not _dict_contains(container[key], value):
                return False
        elif container[key] != value:
            return False
    return True
