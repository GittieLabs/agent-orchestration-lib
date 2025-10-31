"""Tests for SwitchStep multi-branch conditional logic."""

import pytest
from pydantic import BaseModel

from agent_lib import ExecutionContext, EventEmitter, SwitchStep
from agent_lib.testing import MockAgent


class SimpleInput(BaseModel):
    """Simple input model for testing."""

    value: str
    category: str


class SimpleOutput(BaseModel):
    """Simple output model for testing."""

    result: str


@pytest.fixture
def context():
    """Create execution context."""
    return ExecutionContext()


@pytest.fixture
def emitter():
    """Create event emitter."""
    return EventEmitter()


@pytest.fixture
def mock_agents(context, emitter):
    """Create mock agents for different cases."""
    return {
        "agent_a": MockAgent(
            name="agent_a",
            return_value=SimpleOutput(result="Processed by A"),
            context=context,
            emitter=emitter,
        ),
        "agent_b": MockAgent(
            name="agent_b",
            return_value=SimpleOutput(result="Processed by B"),
            context=context,
            emitter=emitter,
        ),
        "agent_c": MockAgent(
            name="agent_c",
            return_value=SimpleOutput(result="Processed by C"),
            context=context,
            emitter=emitter,
        ),
        "default_agent": MockAgent(
            name="default_agent",
            return_value=SimpleOutput(result="Processed by default"),
            context=context,
            emitter=emitter,
        ),
    }


class TestSwitchStepBasics:
    """Test basic SwitchStep functionality."""

    def test_create_switch_step(self, context, emitter, mock_agents):
        """Test creating a SwitchStep."""
        switch = SwitchStep(
            name="test_switch",
            selector=lambda x: x.category,
            cases={
                "A": mock_agents["agent_a"],
                "B": mock_agents["agent_b"],
            },
            context=context,
            emitter=emitter,
        )

        assert switch.name == "test_switch"
        assert len(switch.cases) == 2
        assert switch.default_agent is None

    def test_create_switch_with_default(self, context, emitter, mock_agents):
        """Test creating a SwitchStep with default agent."""
        switch = SwitchStep(
            name="test_switch",
            selector=lambda x: x.category,
            cases={"A": mock_agents["agent_a"]},
            default_agent=mock_agents["default_agent"],
            context=context,
            emitter=emitter,
        )

        assert switch.default_agent is not None
        assert switch.default_agent.name == "default_agent"

    def test_empty_cases_raises_error(self, context, emitter):
        """Test that empty cases dict raises ValueError."""
        with pytest.raises(ValueError, match="at least one case"):
            SwitchStep(
                name="empty_switch",
                selector=lambda x: x.category,
                cases={},
                context=context,
                emitter=emitter,
            )


class TestSwitchExecution:
    """Test SwitchStep execution with different cases."""

    @pytest.mark.asyncio
    async def test_execute_matching_case_a(self, context, emitter, mock_agents):
        """Test executing with a matching case A."""
        switch = SwitchStep(
            name="test_switch",
            selector=lambda x: x.category,
            cases={
                "A": mock_agents["agent_a"],
                "B": mock_agents["agent_b"],
                "C": mock_agents["agent_c"],
            },
            context=context,
            emitter=emitter,
        )

        input_data = SimpleInput(value="test", category="A")
        result = await switch.execute(input_data)

        assert result.result == "Processed by A"
        assert mock_agents["agent_a"].call_count == 1
        assert mock_agents["agent_b"].call_count == 0

    @pytest.mark.asyncio
    async def test_execute_matching_case_b(self, context, emitter, mock_agents):
        """Test executing with a matching case B."""
        switch = SwitchStep(
            name="test_switch",
            selector=lambda x: x.category,
            cases={
                "A": mock_agents["agent_a"],
                "B": mock_agents["agent_b"],
                "C": mock_agents["agent_c"],
            },
            context=context,
            emitter=emitter,
        )

        input_data = SimpleInput(value="test", category="B")
        result = await switch.execute(input_data)

        assert result.result == "Processed by B"
        assert mock_agents["agent_b"].call_count == 1
        assert mock_agents["agent_a"].call_count == 0

    @pytest.mark.asyncio
    async def test_execute_with_default_agent(self, context, emitter, mock_agents):
        """Test executing with no match but default agent provided."""
        switch = SwitchStep(
            name="test_switch",
            selector=lambda x: x.category,
            cases={
                "A": mock_agents["agent_a"],
                "B": mock_agents["agent_b"],
            },
            default_agent=mock_agents["default_agent"],
            context=context,
            emitter=emitter,
        )

        input_data = SimpleInput(value="test", category="Z")
        result = await switch.execute(input_data)

        assert result.result == "Processed by default"
        assert mock_agents["default_agent"].call_count == 1
        assert mock_agents["agent_a"].call_count == 0

    @pytest.mark.asyncio
    async def test_execute_no_match_no_default_raises_error(self, context, emitter, mock_agents):
        """Test executing with no match and no default raises ValueError."""
        switch = SwitchStep(
            name="test_switch",
            selector=lambda x: x.category,
            cases={
                "A": mock_agents["agent_a"],
                "B": mock_agents["agent_b"],
            },
            context=context,
            emitter=emitter,
        )

        input_data = SimpleInput(value="test", category="Z")

        with pytest.raises(ValueError, match="No case matched 'Z'"):
            await switch.execute(input_data)


class TestSelectorFunction:
    """Test different selector functions."""

    @pytest.mark.asyncio
    async def test_complex_selector(self, context, emitter, mock_agents):
        """Test with a complex selector function."""

        def complex_selector(data: SimpleInput) -> str:
            if data.value.startswith("high"):
                return "high_priority"
            elif data.value.startswith("low"):
                return "low_priority"
            else:
                return "normal"

        switch = SwitchStep(
            name="priority_switch",
            selector=complex_selector,
            cases={
                "high_priority": mock_agents["agent_a"],
                "low_priority": mock_agents["agent_b"],
                "normal": mock_agents["agent_c"],
            },
            context=context,
            emitter=emitter,
        )

        # Test high priority
        result = await switch.execute(SimpleInput(value="high_value", category="any"))
        assert result.result == "Processed by A"

        # Test low priority
        result = await switch.execute(SimpleInput(value="low_value", category="any"))
        assert result.result == "Processed by B"

        # Test normal
        result = await switch.execute(SimpleInput(value="normal_value", category="any"))
        assert result.result == "Processed by C"

    @pytest.mark.asyncio
    async def test_selector_exception_handling(self, context, emitter, mock_agents):
        """Test that selector exceptions are properly handled."""

        def failing_selector(data: SimpleInput) -> str:
            raise RuntimeError("Selector failed")

        switch = SwitchStep(
            name="test_switch",
            selector=failing_selector,
            cases={"A": mock_agents["agent_a"]},
            context=context,
            emitter=emitter,
        )

        input_data = SimpleInput(value="test", category="A")

        with pytest.raises(ValueError, match="Selector function failed"):
            await switch.execute(input_data)


class TestEventEmission:
    """Test event emission during switch execution."""

    @pytest.mark.asyncio
    async def test_emits_case_selected_event(self, context, emitter, mock_agents):
        """Test that case_selected event is emitted."""
        events = []
        emitter.subscribe("progress", lambda event: events.append(event))

        switch = SwitchStep(
            name="test_switch",
            selector=lambda x: x.category,
            cases={"A": mock_agents["agent_a"]},
            context=context,
            emitter=emitter,
        )

        input_data = SimpleInput(value="test", category="A")
        await switch.execute(input_data)

        # Find case_selected event
        case_selected_events = [e for e in events if e.get("stage") == "case_selected"]
        assert len(case_selected_events) > 0
        assert case_selected_events[0]["data"]["case"] == "A"

    @pytest.mark.asyncio
    async def test_emits_default_event_when_using_default(self, context, emitter, mock_agents):
        """Test that using_default event is emitted."""
        events = []
        emitter.subscribe("progress", lambda event: events.append(event))

        switch = SwitchStep(
            name="test_switch",
            selector=lambda x: x.category,
            cases={"A": mock_agents["agent_a"]},
            default_agent=mock_agents["default_agent"],
            context=context,
            emitter=emitter,
        )

        input_data = SimpleInput(value="test", category="Z")
        await switch.execute(input_data)

        # Find using_default event
        default_events = [e for e in events if e.get("stage") == "using_default"]
        assert len(default_events) > 0
        assert default_events[0]["data"]["case"] == "Z"


class TestMultipleCases:
    """Test SwitchStep with many cases."""

    @pytest.mark.asyncio
    async def test_many_cases(self, context, emitter):
        """Test SwitchStep with many cases (5+)."""
        agents = {}
        for i in range(10):
            agents[f"case_{i}"] = MockAgent(
                name=f"agent_{i}",
                return_value=SimpleOutput(result=f"Processed by {i}"),
                context=context,
                emitter=emitter,
            )

        switch = SwitchStep(
            name="many_cases_switch",
            selector=lambda x: f"case_{x.category}",
            cases=agents,
            context=context,
            emitter=emitter,
        )

        # Test different cases
        for i in [0, 3, 7, 9]:
            result = await switch.execute(SimpleInput(value="test", category=str(i)))
            assert result.result == f"Processed by {i}"

    @pytest.mark.asyncio
    async def test_switch_reuse(self, context, emitter, mock_agents):
        """Test that SwitchStep can be reused multiple times."""
        switch = SwitchStep(
            name="reusable_switch",
            selector=lambda x: x.category,
            cases={
                "A": mock_agents["agent_a"],
                "B": mock_agents["agent_b"],
            },
            context=context,
            emitter=emitter,
        )

        # Execute multiple times
        await switch.execute(SimpleInput(value="test1", category="A"))
        await switch.execute(SimpleInput(value="test2", category="B"))
        await switch.execute(SimpleInput(value="test3", category="A"))

        assert mock_agents["agent_a"].call_count == 2
        assert mock_agents["agent_b"].call_count == 1
