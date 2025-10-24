"""Mock agents and spies for testing."""

from typing import Any, Optional, List, Callable
from agent_lib.core import AgentBlock, ExecutionContext


class MockAgent(AgentBlock):
    """Mock agent for testing.

    This agent allows you to control what it returns and whether it raises
    exceptions, making it easy to test agent workflows without real agents.

    Example:
        ```python
        # Create a mock that returns a fixed value
        mock = MockAgent(name="mock", return_value={"result": "success"})
        result = await mock.execute({"input": "data"})
        # result == {"result": "success"}

        # Create a mock that raises an exception
        mock = MockAgent(name="mock", side_effect=ValueError("test error"))
        # await mock.execute(...) will raise ValueError

        # Check how many times it was called
        assert mock.call_count == 1
        ```
    """

    def __init__(
        self,
        name: str,
        return_value: Any = None,
        side_effect: Optional[Exception] = None,
        delay: float = 0.0,
    ):
        """Initialize mock agent.

        Args:
            name: Agent name
            return_value: Value to return from execute (default: None)
            side_effect: Exception to raise from execute (default: None)
            delay: Simulated delay in seconds (default: 0.0)
        """
        super().__init__(name=name)
        self._return_value = return_value
        self._side_effect = side_effect
        self._delay = delay
        self._call_count = 0
        self._calls: List[tuple[Any, ExecutionContext]] = []

    async def _execute(
        self,
        input_data: Any,
        context: ExecutionContext
    ) -> Any:
        """Execute mock agent.

        Args:
            input_data: Input data
            context: Execution context

        Returns:
            The configured return value

        Raises:
            Exception: If side_effect is configured
        """
        import asyncio

        self._call_count += 1
        self._calls.append((input_data, context))

        if self._delay > 0:
            await asyncio.sleep(self._delay)

        if self._side_effect is not None:
            raise self._side_effect

        return self._return_value

    @property
    def call_count(self) -> int:
        """Get number of times this agent was called."""
        return self._call_count

    @property
    def calls(self) -> List[tuple[Any, ExecutionContext]]:
        """Get list of all calls made to this agent.

        Returns:
            List of (input_data, context) tuples
        """
        return self._calls.copy()

    @property
    def last_call(self) -> Optional[tuple[Any, ExecutionContext]]:
        """Get the last call made to this agent.

        Returns:
            Tuple of (input_data, context) or None if never called
        """
        return self._calls[-1] if self._calls else None

    def reset(self) -> None:
        """Reset call tracking."""
        self._call_count = 0
        self._calls = []

    def configure(
        self,
        return_value: Any = None,
        side_effect: Optional[Exception] = None,
        delay: float = 0.0,
    ) -> None:
        """Reconfigure the mock agent.

        Args:
            return_value: New return value
            side_effect: New side effect
            delay: New delay
        """
        self._return_value = return_value
        self._side_effect = side_effect
        self._delay = delay


class AgentSpy(AgentBlock):
    """Spy wrapper to track agent calls.

    This wrapper tracks calls to an existing agent without changing its
    behavior. Useful for verifying that an agent was called with the
    right inputs.

    Example:
        ```python
        # Wrap an existing agent
        real_agent = MyAgent(name="real")
        spy = AgentSpy(real_agent)

        # Use the spy in place of the real agent
        result = await spy.execute({"input": "data"})

        # Check tracking
        assert spy.call_count == 1
        assert spy.last_input == {"input": "data"}
        ```
    """

    def __init__(self, agent: AgentBlock):
        """Initialize agent spy.

        Args:
            agent: The agent to wrap and track
        """
        super().__init__(name=f"spy[{agent.name}]")
        self._agent = agent
        self._call_count = 0
        self._calls: List[tuple[Any, ExecutionContext, Any]] = []
        self._errors: List[tuple[Any, ExecutionContext, Exception]] = []

    async def _execute(
        self,
        input_data: Any,
        context: ExecutionContext
    ) -> Any:
        """Execute wrapped agent and track the call.

        Args:
            input_data: Input data
            context: Execution context

        Returns:
            Result from wrapped agent

        Raises:
            Exception: Any exception raised by wrapped agent
        """
        self._call_count += 1

        try:
            result = await self._agent.execute(input_data, context)
            self._calls.append((input_data, context, result))
            return result
        except Exception as e:
            self._errors.append((input_data, context, e))
            raise

    @property
    def call_count(self) -> int:
        """Get number of times the wrapped agent was called."""
        return self._call_count

    @property
    def calls(self) -> List[tuple[Any, ExecutionContext, Any]]:
        """Get list of successful calls.

        Returns:
            List of (input_data, context, result) tuples
        """
        return self._calls.copy()

    @property
    def errors(self) -> List[tuple[Any, ExecutionContext, Exception]]:
        """Get list of failed calls.

        Returns:
            List of (input_data, context, exception) tuples
        """
        return self._errors.copy()

    @property
    def last_call(self) -> Optional[tuple[Any, ExecutionContext, Any]]:
        """Get the last successful call.

        Returns:
            Tuple of (input_data, context, result) or None
        """
        return self._calls[-1] if self._calls else None

    @property
    def last_input(self) -> Optional[Any]:
        """Get the input from the last successful call."""
        return self._calls[-1][0] if self._calls else None

    @property
    def last_output(self) -> Optional[Any]:
        """Get the output from the last successful call."""
        return self._calls[-1][2] if self._calls else None

    @property
    def wrapped_agent(self) -> AgentBlock:
        """Get the wrapped agent."""
        return self._agent

    def reset(self) -> None:
        """Reset call tracking."""
        self._call_count = 0
        self._calls = []
        self._errors = []
