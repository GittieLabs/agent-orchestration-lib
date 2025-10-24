"""Unit tests for Flow orchestration."""

import pytest
from typing import Any
from agent_lib.core import Flow, AgentBlock, ExecutionContext, EventEmitter
from agent_lib.events import Event, ProgressEvent


class NumberDoubler(AgentBlock[int, int]):
    """Test agent that doubles a number."""

    async def process(self, input_data: int) -> int:
        """Double the input number."""
        return input_data * 2


class NumberAdder(AgentBlock[int, int]):
    """Test agent that adds a fixed value."""

    def __init__(self, name: str, add_value: int):
        super().__init__(name=name)
        self.add_value = add_value

    async def process(self, input_data: int) -> int:
        """Add fixed value to input."""
        return input_data + self.add_value


class StringFormatter(AgentBlock[int, str]):
    """Test agent that converts number to formatted string."""

    async def process(self, input_data: int) -> str:
        """Format number as string."""
        return f"Result: {input_data}"


class FailingAgent(AgentBlock[Any, Any]):
    """Test agent that always fails."""

    async def process(self, input_data: Any) -> Any:
        """Always raise an error."""
        raise ValueError("Intentional failure")


class EventEmittingAgent(AgentBlock[int, int]):
    """Test agent that emits progress events."""

    async def process(self, input_data: int) -> int:
        """Emit progress and return input."""
        await self.emit_progress("processing", 0.5, "Processing")
        return input_data


class ContextWritingAgent(AgentBlock[str, str]):
    """Test agent that writes to context."""

    def __init__(self, name: str, key: str, value: str):
        super().__init__(name=name)
        self.key = key
        self.value = value

    async def process(self, input_data: str) -> str:
        """Write to context and return input."""
        self.context.register(self.key, self.value)
        return input_data


class ContextReadingAgent(AgentBlock[str, str]):
    """Test agent that reads from context."""

    def __init__(self, name: str, key: str):
        super().__init__(name=name)
        self.key = key

    async def process(self, input_data: str) -> str:
        """Read from context and append to input."""
        value = self.context.get(self.key, "not_found")
        return f"{input_data}:{value}"


class TestFlowInitialization:
    """Test Flow initialization and configuration."""

    def test_create_flow_basic(self):
        """Test creating a basic flow."""
        flow = Flow(name="test_flow")
        assert flow.name == "test_flow"
        assert flow.context is not None
        assert flow.emitter is not None
        assert len(flow.agents) == 0

    def test_create_flow_with_context(self):
        """Test creating flow with custom context."""
        context = ExecutionContext()
        flow = Flow(name="test_flow", context=context)
        assert flow.context is context

    def test_create_flow_with_emitter(self):
        """Test creating flow with custom emitter."""
        emitter = EventEmitter()
        flow = Flow(name="test_flow", emitter=emitter)
        assert flow.emitter is emitter

    def test_flow_repr(self):
        """Test flow string representation."""
        flow = Flow(name="test_flow")
        flow.add_agent(NumberDoubler(name="doubler"))
        repr_str = repr(flow)
        assert "Flow" in repr_str
        assert "test_flow" in repr_str
        assert "1" in repr_str  # agent count

    def test_flow_len(self):
        """Test flow length."""
        flow = Flow(name="test_flow")
        assert len(flow) == 0
        flow.add_agent(NumberDoubler(name="doubler"))
        assert len(flow) == 1


class TestFlowAgentManagement:
    """Test Flow agent management methods."""

    def test_add_agent(self):
        """Test adding agent to flow."""
        flow = Flow(name="test_flow")
        agent = NumberDoubler(name="doubler")

        result = flow.add_agent(agent)

        assert len(flow.agents) == 1
        assert flow.agents[0] is agent
        assert result is flow  # method chaining

    def test_add_agent_shares_context(self):
        """Test that adding agent shares flow context."""
        flow = Flow(name="test_flow")
        agent = NumberDoubler(name="doubler")

        flow.add_agent(agent)

        assert agent.context is flow.context

    def test_add_agent_shares_emitter(self):
        """Test that adding agent shares flow emitter."""
        flow = Flow(name="test_flow")
        agent = NumberDoubler(name="doubler")

        flow.add_agent(agent)

        assert agent.emitter is flow.emitter

    def test_add_multiple_agents_chaining(self):
        """Test adding multiple agents with method chaining."""
        flow = Flow(name="test_flow")
        agent1 = NumberDoubler(name="doubler")
        agent2 = NumberAdder(name="adder", add_value=5)

        flow.add_agent(agent1).add_agent(agent2)

        assert len(flow.agents) == 2
        assert flow.agents[0] is agent1
        assert flow.agents[1] is agent2

    def test_clear_agents(self):
        """Test clearing all agents from flow."""
        flow = Flow(name="test_flow")
        flow.add_agent(NumberDoubler(name="doubler"))
        flow.add_agent(NumberAdder(name="adder", add_value=5))

        result = flow.clear_agents()

        assert len(flow.agents) == 0
        assert result is flow  # method chaining

    def test_get_agent_count(self):
        """Test getting agent count."""
        flow = Flow(name="test_flow")
        assert flow.get_agent_count() == 0

        flow.add_agent(NumberDoubler(name="doubler"))
        assert flow.get_agent_count() == 1

        flow.add_agent(NumberAdder(name="adder", add_value=5))
        assert flow.get_agent_count() == 2

    def test_get_agent_names(self):
        """Test getting agent names."""
        flow = Flow(name="test_flow")
        assert flow.get_agent_names() == []

        flow.add_agent(NumberDoubler(name="doubler"))
        flow.add_agent(NumberAdder(name="adder", add_value=5))

        names = flow.get_agent_names()
        assert names == ["doubler", "adder"]


class TestFlowSequentialExecution:
    """Test Flow sequential execution."""

    @pytest.mark.asyncio
    async def test_execute_sequential_single_agent(self):
        """Test sequential execution with single agent."""
        flow = Flow(name="test_flow")
        flow.add_agent(NumberDoubler(name="doubler"))

        result = await flow.execute_sequential(10)

        assert result == 20

    @pytest.mark.asyncio
    async def test_execute_sequential_multiple_agents(self):
        """Test sequential execution with multiple agents."""
        flow = Flow(name="test_flow")
        flow.add_agent(NumberDoubler(name="doubler"))  # 10 * 2 = 20
        flow.add_agent(NumberAdder(name="adder", add_value=5))  # 20 + 5 = 25

        result = await flow.execute_sequential(10)

        assert result == 25

    @pytest.mark.asyncio
    async def test_execute_sequential_chain(self):
        """Test sequential execution chains output to input."""
        flow = Flow(name="test_flow")
        flow.add_agent(NumberDoubler(name="doubler1"))  # 5 * 2 = 10
        flow.add_agent(NumberDoubler(name="doubler2"))  # 10 * 2 = 20
        flow.add_agent(NumberAdder(name="adder", add_value=3))  # 20 + 3 = 23

        result = await flow.execute_sequential(5)

        assert result == 23

    @pytest.mark.asyncio
    async def test_execute_sequential_type_transformation(self):
        """Test sequential execution with type transformation."""
        flow = Flow(name="test_flow")
        flow.add_agent(NumberDoubler(name="doubler"))  # 10 * 2 = 20
        flow.add_agent(StringFormatter(name="formatter"))  # "Result: 20"

        result = await flow.execute_sequential(10)

        assert result == "Result: 20"
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_execute_sequential_no_agents_error(self):
        """Test sequential execution fails with no agents."""
        flow = Flow(name="test_flow")

        with pytest.raises(ValueError, match="No agents in flow"):
            await flow.execute_sequential(10)

    @pytest.mark.asyncio
    async def test_execute_sequential_agent_failure(self):
        """Test sequential execution propagates agent failure."""
        flow = Flow(name="test_flow")
        flow.add_agent(NumberDoubler(name="doubler"))
        flow.add_agent(FailingAgent(name="failing"))

        with pytest.raises(ValueError, match="Intentional failure"):
            await flow.execute_sequential(10)

    @pytest.mark.asyncio
    async def test_execute_sequential_shares_context(self):
        """Test sequential execution shares context between agents."""
        flow = Flow(name="test_flow")
        flow.add_agent(ContextWritingAgent(name="writer", key="test_key", value="test_value"))
        flow.add_agent(ContextReadingAgent(name="reader", key="test_key"))

        result = await flow.execute_sequential("input")

        assert result == "input:test_value"


class TestFlowParallelExecution:
    """Test Flow parallel execution."""

    @pytest.mark.asyncio
    async def test_execute_parallel_basic(self):
        """Test parallel execution with separate inputs."""
        flow = Flow(name="test_flow")
        flow.add_agent(NumberDoubler(name="doubler1"))
        flow.add_agent(NumberDoubler(name="doubler2"))
        flow.add_agent(NumberDoubler(name="doubler3"))

        results = await flow.execute_parallel([1, 2, 3])

        assert results == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_execute_parallel_different_agents(self):
        """Test parallel execution with different agent types."""
        flow = Flow(name="test_flow")
        flow.add_agent(NumberDoubler(name="doubler"))
        flow.add_agent(NumberAdder(name="adder", add_value=10))

        results = await flow.execute_parallel([5, 5])

        assert results == [10, 15]  # 5*2=10, 5+10=15

    @pytest.mark.asyncio
    async def test_execute_parallel_no_agents_error(self):
        """Test parallel execution fails with no agents."""
        flow = Flow(name="test_flow")

        with pytest.raises(ValueError, match="No agents in flow"):
            await flow.execute_parallel([1, 2, 3])

    @pytest.mark.asyncio
    async def test_execute_parallel_input_count_mismatch(self):
        """Test parallel execution fails with wrong input count."""
        flow = Flow(name="test_flow")
        flow.add_agent(NumberDoubler(name="doubler1"))
        flow.add_agent(NumberDoubler(name="doubler2"))

        with pytest.raises(ValueError, match="Number of inputs.*must match"):
            await flow.execute_parallel([1, 2, 3])  # 3 inputs, 2 agents

    @pytest.mark.asyncio
    async def test_execute_parallel_agent_failure(self):
        """Test parallel execution propagates agent failure."""
        flow = Flow(name="test_flow")
        flow.add_agent(NumberDoubler(name="doubler"))
        flow.add_agent(FailingAgent(name="failing"))

        with pytest.raises(ValueError, match="Intentional failure"):
            await flow.execute_parallel([1, 2])

    @pytest.mark.asyncio
    async def test_execute_parallel_context_isolation(self):
        """Test parallel execution isolates agent contexts."""
        flow = Flow(name="test_flow")
        flow.add_agent(ContextWritingAgent(name="writer1", key="key", value="value1"))
        flow.add_agent(ContextWritingAgent(name="writer2", key="key", value="value2"))

        await flow.execute_parallel(["input1", "input2"])

        # Each agent should have written to its own child context
        # Parent context should not have the key (agents write to child contexts)
        assert flow.context.get("key", "not_found") == "not_found"


class TestFlowParallelSameInput:
    """Test Flow parallel execution with same input."""

    @pytest.mark.asyncio
    async def test_execute_parallel_same_input_basic(self):
        """Test parallel execution with same input for all agents."""
        flow = Flow(name="test_flow")
        flow.add_agent(NumberDoubler(name="doubler"))
        flow.add_agent(NumberAdder(name="adder", add_value=5))

        results = await flow.execute_parallel_same_input(10)

        assert results == [20, 15]  # 10*2=20, 10+5=15

    @pytest.mark.asyncio
    async def test_execute_parallel_same_input_no_agents(self):
        """Test parallel same input fails with no agents."""
        flow = Flow(name="test_flow")

        with pytest.raises(ValueError, match="No agents in flow"):
            await flow.execute_parallel_same_input(10)

    @pytest.mark.asyncio
    async def test_execute_parallel_same_input_multiple_agents(self):
        """Test parallel same input with multiple agents."""
        flow = Flow(name="test_flow")
        flow.add_agent(NumberAdder(name="adder1", add_value=1))
        flow.add_agent(NumberAdder(name="adder2", add_value=2))
        flow.add_agent(NumberAdder(name="adder3", add_value=3))

        results = await flow.execute_parallel_same_input(10)

        assert results == [11, 12, 13]


class TestFlowFanout:
    """Test Flow fan-out/fan-in pattern."""

    @pytest.mark.asyncio
    async def test_execute_with_fanout_no_merge(self):
        """Test fanout without merge function returns list."""
        flow = Flow(name="test_flow")
        flow.add_agent(NumberDoubler(name="doubler"))
        flow.add_agent(NumberAdder(name="adder", add_value=5))

        result = await flow.execute_with_fanout(10)

        assert result == [20, 15]

    @pytest.mark.asyncio
    async def test_execute_with_fanout_with_merge(self):
        """Test fanout with merge function."""
        flow = Flow(name="test_flow")
        flow.add_agent(NumberDoubler(name="doubler"))
        flow.add_agent(NumberAdder(name="adder", add_value=5))

        def sum_results(results):
            return sum(results)

        result = await flow.execute_with_fanout(10, merge_fn=sum_results)

        assert result == 35  # 20 + 15

    @pytest.mark.asyncio
    async def test_execute_with_fanout_average_merge(self):
        """Test fanout with average merge function."""
        flow = Flow(name="test_flow")
        flow.add_agent(NumberAdder(name="adder1", add_value=0))
        flow.add_agent(NumberAdder(name="adder2", add_value=10))
        flow.add_agent(NumberAdder(name="adder3", add_value=20))

        def average(results):
            return sum(results) / len(results)

        result = await flow.execute_with_fanout(10, merge_fn=average)

        assert result == 20.0  # (10 + 20 + 30) / 3

    @pytest.mark.asyncio
    async def test_execute_with_fanout_custom_merge(self):
        """Test fanout with custom merge logic."""
        flow = Flow(name="test_flow")
        flow.add_agent(NumberDoubler(name="doubler1"))
        flow.add_agent(NumberDoubler(name="doubler2"))
        flow.add_agent(NumberDoubler(name="doubler3"))

        def max_result(results):
            return max(results)

        result = await flow.execute_with_fanout(5, merge_fn=max_result)

        assert result == 10  # All return 10, max is 10


class TestFlowEventAggregation:
    """Test Flow event aggregation from multiple agents."""

    @pytest.mark.asyncio
    async def test_sequential_events_aggregated(self):
        """Test events from sequential agents are aggregated."""
        flow = Flow(name="test_flow")
        flow.add_agent(EventEmittingAgent(name="agent1"))
        flow.add_agent(EventEmittingAgent(name="agent2"))

        events = []

        def event_handler(event: Event):
            events.append(event)

        flow.emitter.subscribe("progress", event_handler)

        await flow.execute_sequential(10)

        # Should have 2 progress events (one from each agent)
        progress_events = [e for e in events if isinstance(e, ProgressEvent)]
        assert len(progress_events) == 2

    @pytest.mark.asyncio
    async def test_parallel_events_aggregated(self):
        """Test events from parallel agents are aggregated."""
        flow = Flow(name="test_flow")
        flow.add_agent(EventEmittingAgent(name="agent1"))
        flow.add_agent(EventEmittingAgent(name="agent2"))
        flow.add_agent(EventEmittingAgent(name="agent3"))

        events = []

        def event_handler(event: Event):
            events.append(event)

        flow.emitter.subscribe("progress", event_handler)

        await flow.execute_parallel([1, 2, 3])

        # Should have 3 progress events (one from each agent)
        progress_events = [e for e in events if isinstance(e, ProgressEvent)]
        assert len(progress_events) == 3


class TestFlowRealWorldPatterns:
    """Test real-world usage patterns."""

    @pytest.mark.asyncio
    async def test_data_processing_pipeline(self):
        """Test sequential data processing pipeline."""
        flow = Flow(name="processing_pipeline")
        flow.add_agent(NumberDoubler(name="normalize"))
        flow.add_agent(NumberAdder(name="offset", add_value=10))
        flow.add_agent(NumberDoubler(name="scale"))

        result = await flow.execute_sequential(5)

        # 5 * 2 = 10, 10 + 10 = 20, 20 * 2 = 40
        assert result == 40

    @pytest.mark.asyncio
    async def test_parallel_scoring_with_average(self):
        """Test parallel scoring with average aggregation."""
        flow = Flow(name="scoring")
        flow.add_agent(NumberAdder(name="scorer1", add_value=10))
        flow.add_agent(NumberAdder(name="scorer2", add_value=20))
        flow.add_agent(NumberAdder(name="scorer3", add_value=30))

        avg_score = await flow.execute_with_fanout(
            50, merge_fn=lambda r: sum(r) / len(r)
        )

        # (60 + 70 + 80) / 3 = 70
        assert avg_score == 70.0

    @pytest.mark.asyncio
    async def test_reusable_flow_with_clear(self):
        """Test reusing flow by clearing and adding new agents."""
        flow = Flow(name="reusable_flow")
        flow.add_agent(NumberDoubler(name="doubler"))

        result1 = await flow.execute_sequential(5)
        assert result1 == 10

        # Clear and add different agents
        flow.clear_agents()
        flow.add_agent(NumberAdder(name="adder", add_value=100))

        result2 = await flow.execute_sequential(5)
        assert result2 == 105

    @pytest.mark.asyncio
    async def test_nested_flow_pattern(self):
        """Test using flow output as input to another flow."""
        # First flow: preprocessing
        flow1 = Flow(name="preprocess")
        flow1.add_agent(NumberDoubler(name="doubler"))

        # Second flow: processing
        flow2 = Flow(name="process")
        flow2.add_agent(NumberAdder(name="adder", add_value=5))

        # Execute flows in sequence
        intermediate = await flow1.execute_sequential(10)
        final = await flow2.execute_sequential(intermediate)

        # 10 * 2 = 20, then 20 + 5 = 25
        assert final == 25
