"""Test fixtures for agent testing."""

from typing import Optional, List
from agent_lib.core import ExecutionContext, EventEmitter, Flow
from agent_lib.core.agent_block import AgentBlock


def create_test_context(
    initial_data: Optional[dict] = None,
    emitter: Optional[EventEmitter] = None
) -> ExecutionContext:
    """Create a test execution context.

    Args:
        initial_data: Initial context data (default: empty dict)
        emitter: Event emitter to use (default: creates new one)

    Returns:
        ExecutionContext configured for testing

    Example:
        ```python
        context = create_test_context({"user_id": 123})
        assert context.get("user_id") == 123
        ```
    """
    if emitter is None:
        emitter = create_test_emitter()

    context = ExecutionContext(emitter=emitter)

    if initial_data:
        for key, value in initial_data.items():
            context.set(key, value)

    return context


def create_test_emitter(
    capture_events: bool = True
) -> EventEmitter:
    """Create a test event emitter.

    Args:
        capture_events: If True, stores emitted events for inspection

    Returns:
        EventEmitter configured for testing

    Example:
        ```python
        emitter = create_test_emitter()

        # Subscribe to events
        events = []
        emitter.on("test_event", lambda e: events.append(e))

        # Emit event
        await emitter.emit("test_event", {"data": "test"})

        assert len(events) == 1
        ```
    """
    emitter = EventEmitter()

    if capture_events:
        # Add event capture capability
        emitter._captured_events = []

        original_emit = emitter.emit

        async def capturing_emit(event_type: str, data: dict):
            emitter._captured_events.append({
                "type": event_type,
                "data": data
            })
            return await original_emit(event_type, data)

        emitter.emit = capturing_emit
        emitter.get_captured_events = lambda: emitter._captured_events.copy()
        emitter.clear_captured_events = lambda: emitter._captured_events.clear()

    return emitter


def create_test_flow(
    agents: Optional[List[AgentBlock]] = None,
    name: str = "test_flow",
    emitter: Optional[EventEmitter] = None
) -> Flow:
    """Create a test flow.

    Args:
        agents: List of agents to add to flow (default: empty)
        name: Flow name (default: "test_flow")
        emitter: Event emitter to use (default: creates new one)

    Returns:
        Flow configured for testing

    Example:
        ```python
        from agent_lib.testing import MockAgent

        agent1 = MockAgent("agent1", return_value={"result": 1})
        agent2 = MockAgent("agent2", return_value={"result": 2})

        flow = create_test_flow(agents=[agent1, agent2])

        result = await flow.execute_sequential({"input": "data"})
        ```
    """
    if emitter is None:
        emitter = create_test_emitter()

    flow = Flow(name=name, emitter=emitter)

    if agents:
        for agent in agents:
            flow.add_agent(agent)

    return flow
