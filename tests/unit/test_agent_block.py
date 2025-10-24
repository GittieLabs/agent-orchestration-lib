"""Unit tests for AgentBlock."""

import pytest
from pydantic import BaseModel
from agent_lib.core import AgentBlock, ExecutionContext, EventEmitter
from agent_lib.events import (
    StartEvent,
    ProgressEvent,
    ErrorEvent,
    CompletionEvent,
)


# Test input/output models
class SimpleInput(BaseModel):
    """Simple test input model."""

    value: int


class SimpleOutput(BaseModel):
    """Simple test output model."""

    result: int


class GreetingInput(BaseModel):
    """Greeting test input model."""

    name: str


class GreetingOutput(BaseModel):
    """Greeting test output model."""

    message: str


# Test agent implementations
class SimpleAgent(AgentBlock[SimpleInput, SimpleOutput]):
    """Simple agent that doubles the input value."""

    async def process(self, input_data: SimpleInput) -> SimpleOutput:
        """Double the input value."""
        return SimpleOutput(result=input_data.value * 2)


class GreetingAgent(AgentBlock[GreetingInput, GreetingOutput]):
    """Agent that generates greetings."""

    async def process(self, input_data: GreetingInput) -> GreetingOutput:
        """Generate a greeting message."""
        return GreetingOutput(message=f"Hello, {input_data.name}!")


class ProgressReportingAgent(AgentBlock[SimpleInput, SimpleOutput]):
    """Agent that reports progress during execution."""

    async def process(self, input_data: SimpleInput) -> SimpleOutput:
        """Process with progress updates."""
        await self.emit_progress("start", 0.0, "Starting processing")
        await self.emit_progress("processing", 0.5, "Halfway done")
        await self.emit_progress("complete", 1.0, "Finishing up")
        return SimpleOutput(result=input_data.value * 2)


class FailingAgent(AgentBlock[SimpleInput, SimpleOutput]):
    """Agent that always fails."""

    async def process(self, input_data: SimpleInput) -> SimpleOutput:
        """Always raise an error."""
        raise ValueError("Intentional failure")


class ValidatingAgent(AgentBlock[SimpleInput, SimpleOutput]):
    """Agent with custom validation."""

    async def validate_input(self, input_data: SimpleInput) -> None:
        """Validate that input value is positive."""
        if input_data.value <= 0:
            raise ValueError("Input value must be positive")

    async def validate_output(self, output_data: SimpleOutput) -> None:
        """Validate that output result is positive."""
        if output_data.result <= 0:
            raise ValueError("Output result must be positive")

    async def process(self, input_data: SimpleInput) -> SimpleOutput:
        """Double the input value."""
        return SimpleOutput(result=input_data.value * 2)


class LifecycleAgent(AgentBlock[SimpleInput, SimpleOutput]):
    """Agent that tracks lifecycle hooks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lifecycle_events = []

    async def on_start(self, input_data: SimpleInput) -> None:
        """Track start event."""
        self.lifecycle_events.append("on_start")

    async def on_complete(self, output_data: SimpleOutput) -> None:
        """Track completion event."""
        self.lifecycle_events.append("on_complete")

    async def on_error(self, error: Exception, input_data: SimpleInput) -> None:
        """Track error event."""
        self.lifecycle_events.append("on_error")

    async def process(self, input_data: SimpleInput) -> SimpleOutput:
        """Double the input value."""
        self.lifecycle_events.append("process")
        return SimpleOutput(result=input_data.value * 2)


class TestAgentBlockBasics:
    """Test basic AgentBlock functionality."""

    @pytest.mark.asyncio
    async def test_create_agent(self):
        """Test creating an agent."""
        agent = SimpleAgent(name="test_agent")
        assert agent.name == "test_agent"
        assert agent.context is not None
        assert agent.emitter is not None
        assert not agent._is_running

    @pytest.mark.asyncio
    async def test_agent_with_context(self):
        """Test creating an agent with custom context."""
        context = ExecutionContext()
        context.register("test_key", "test_value")

        agent = SimpleAgent(name="test_agent", context=context)
        assert agent.context.get("test_key") == "test_value"

    @pytest.mark.asyncio
    async def test_agent_with_emitter(self):
        """Test creating an agent with custom emitter."""
        emitter = EventEmitter()
        agent = SimpleAgent(name="test_agent", emitter=emitter)
        assert agent.emitter is emitter

    @pytest.mark.asyncio
    async def test_agent_with_config(self):
        """Test creating an agent with config."""
        agent = SimpleAgent(
            name="test_agent", max_retries=3, timeout=30, debug=True
        )
        assert agent.config["max_retries"] == 3
        assert agent.config["timeout"] == 30
        assert agent.config["debug"] is True

    @pytest.mark.asyncio
    async def test_agent_repr(self):
        """Test agent string representation."""
        agent = SimpleAgent(name="test_agent")
        assert "test_agent" in repr(agent)
        assert "idle" in repr(agent)


class TestAgentBlockExecution:
    """Test agent execution."""

    @pytest.mark.asyncio
    async def test_execute_simple_agent(self):
        """Test executing a simple agent."""
        agent = SimpleAgent(name="doubler")
        result = await agent.execute(SimpleInput(value=5))

        assert isinstance(result, SimpleOutput)
        assert result.result == 10

    @pytest.mark.asyncio
    async def test_execute_greeting_agent(self):
        """Test executing a greeting agent."""
        agent = GreetingAgent(name="greeter")
        result = await agent.execute(GreetingInput(name="Alice"))

        assert isinstance(result, GreetingOutput)
        assert result.message == "Hello, Alice!"

    @pytest.mark.asyncio
    async def test_execute_multiple_times(self):
        """Test executing an agent multiple times."""
        agent = SimpleAgent(name="doubler")

        result1 = await agent.execute(SimpleInput(value=3))
        result2 = await agent.execute(SimpleInput(value=7))

        assert result1.result == 6
        assert result2.result == 14

    @pytest.mark.asyncio
    async def test_concurrent_execution_not_allowed(self):
        """Test that concurrent execution on same agent raises error."""
        agent = SimpleAgent(name="doubler")

        # Manually set running flag
        agent._is_running = True

        with pytest.raises(RuntimeError, match="already running"):
            await agent.execute(SimpleInput(value=5))


class TestAgentBlockEvents:
    """Test event emission during execution."""

    @pytest.mark.asyncio
    async def test_emits_start_event(self):
        """Test that start event is emitted."""
        agent = SimpleAgent(name="test_agent")
        events = []

        async def handler(event):
            events.append(event)

        agent.emitter.subscribe("start", handler)

        await agent.execute(SimpleInput(value=5))

        assert len(events) == 1
        assert isinstance(events[0], StartEvent)
        assert events[0].source == "test_agent"

    @pytest.mark.asyncio
    async def test_emits_completion_event(self):
        """Test that completion event is emitted."""
        agent = SimpleAgent(name="test_agent")
        events = []

        async def handler(event):
            events.append(event)

        agent.emitter.subscribe("completion", handler)

        await agent.execute(SimpleInput(value=5))

        assert len(events) == 1
        assert isinstance(events[0], CompletionEvent)
        assert events[0].success is True
        assert events[0].duration_seconds > 0

    @pytest.mark.asyncio
    async def test_emits_progress_events(self):
        """Test that progress events are emitted."""
        agent = ProgressReportingAgent(name="test_agent")
        events = []

        async def handler(event):
            events.append(event)

        agent.emitter.subscribe("progress", handler)

        await agent.execute(SimpleInput(value=5))

        assert len(events) == 3
        assert all(isinstance(e, ProgressEvent) for e in events)
        assert events[0].progress == 0.0
        assert events[1].progress == 0.5
        assert events[2].progress == 1.0

    @pytest.mark.asyncio
    async def test_emits_error_event(self):
        """Test that error event is emitted on failure."""
        agent = FailingAgent(name="test_agent")
        events = []

        async def handler(event):
            events.append(event)

        agent.emitter.subscribe("error", handler)

        with pytest.raises(ValueError):
            await agent.execute(SimpleInput(value=5))

        assert len(events) == 1
        assert isinstance(events[0], ErrorEvent)
        assert events[0].error_type == "ValueError"
        assert "Intentional failure" in events[0].error_message

    @pytest.mark.asyncio
    async def test_completion_event_on_error(self):
        """Test that completion event with success=False is emitted on error."""
        agent = FailingAgent(name="test_agent")
        events = []

        async def handler(event):
            events.append(event)

        agent.emitter.subscribe("completion", handler)

        with pytest.raises(ValueError):
            await agent.execute(SimpleInput(value=5))

        assert len(events) == 1
        assert isinstance(events[0], CompletionEvent)
        assert events[0].success is False


class TestAgentBlockValidation:
    """Test input/output validation."""

    @pytest.mark.asyncio
    async def test_input_validation_passes(self):
        """Test that valid input passes validation."""
        agent = ValidatingAgent(name="test_agent")
        result = await agent.execute(SimpleInput(value=5))
        assert result.result == 10

    @pytest.mark.asyncio
    async def test_input_validation_fails(self):
        """Test that invalid input fails validation."""
        agent = ValidatingAgent(name="test_agent")

        with pytest.raises(ValueError, match="must be positive"):
            await agent.execute(SimpleInput(value=-5))

    @pytest.mark.asyncio
    async def test_pydantic_validation(self):
        """Test that Pydantic validation works."""
        agent = SimpleAgent(name="test_agent")

        # This should fail Pydantic validation (wrong type)
        with pytest.raises(Exception):  # Pydantic ValidationError
            await agent.execute({"value": "not_an_int"})


class TestAgentBlockLifecycle:
    """Test lifecycle hooks."""

    @pytest.mark.asyncio
    async def test_lifecycle_hooks_called(self):
        """Test that lifecycle hooks are called in correct order."""
        agent = LifecycleAgent(name="test_agent")
        await agent.execute(SimpleInput(value=5))

        assert agent.lifecycle_events == ["on_start", "process", "on_complete"]

    @pytest.mark.asyncio
    async def test_on_error_hook_called(self):
        """Test that on_error hook is called on failure."""

        class FailingLifecycleAgent(LifecycleAgent):
            async def process(self, input_data: SimpleInput) -> SimpleOutput:
                self.lifecycle_events.append("process")
                raise ValueError("Test error")

        agent = FailingLifecycleAgent(name="test_agent")

        with pytest.raises(ValueError):
            await agent.execute(SimpleInput(value=5))

        assert agent.lifecycle_events == ["on_start", "process", "on_error"]


class TestAgentBlockProgressReporting:
    """Test progress reporting."""

    @pytest.mark.asyncio
    async def test_emit_progress(self):
        """Test emitting progress updates."""
        agent = SimpleAgent(name="test_agent")
        events = []

        async def handler(event):
            events.append(event)

        agent.emitter.subscribe("progress", handler)

        # Emit progress manually
        await agent.emit_progress(
            stage="testing",
            progress=0.5,
            message="Test progress",
            details={"items": 50, "total": 100},
        )

        assert len(events) == 1
        assert events[0].stage == "testing"
        assert events[0].progress == 0.5
        assert events[0].message == "Test progress"
        assert events[0].details["items"] == 50


class TestAgentBlockComplexScenarios:
    """Test complex usage scenarios."""

    @pytest.mark.asyncio
    async def test_agent_with_shared_context(self):
        """Test multiple agents sharing the same context."""
        context = ExecutionContext()
        context.register("shared_data", {"count": 0})

        agent1 = SimpleAgent(name="agent1", context=context)
        agent2 = SimpleAgent(name="agent2", context=context)

        # Both agents should have access to the same shared data
        assert agent1.context.get("shared_data") is agent2.context.get(
            "shared_data"
        )

    @pytest.mark.asyncio
    async def test_agent_with_shared_emitter(self):
        """Test multiple agents sharing the same emitter."""
        emitter = EventEmitter()
        all_events = []

        async def handler(event):
            all_events.append(event)

        emitter.subscribe("start", handler)
        emitter.subscribe("completion", handler)

        agent1 = SimpleAgent(name="agent1", emitter=emitter)
        agent2 = SimpleAgent(name="agent2", emitter=emitter)

        await agent1.execute(SimpleInput(value=3))
        await agent2.execute(SimpleInput(value=5))

        # Should have 2 start events and 2 completion events
        start_events = [e for e in all_events if isinstance(e, StartEvent)]
        completion_events = [
            e for e in all_events if isinstance(e, CompletionEvent)
        ]

        assert len(start_events) == 2
        assert len(completion_events) == 2

    @pytest.mark.asyncio
    async def test_sequential_agent_execution(self):
        """Test sequential agent execution pattern."""
        agent1 = SimpleAgent(name="doubler")
        agent2 = SimpleAgent(name="doubler2")

        # Execute agents in sequence, passing output as input
        result1 = await agent1.execute(SimpleInput(value=5))
        result2 = await agent2.execute(SimpleInput(value=result1.result))

        assert result1.result == 10
        assert result2.result == 20

    @pytest.mark.asyncio
    async def test_event_tracking_pattern(self):
        """Test pattern for tracking all events during execution."""
        agent = ProgressReportingAgent(name="test_agent")
        all_events = []

        async def track_all(event):
            all_events.append((type(event).__name__, event))

        # Subscribe to all event types
        agent.emitter.subscribe("start", track_all)
        agent.emitter.subscribe("progress", track_all)
        agent.emitter.subscribe("completion", track_all)

        await agent.execute(SimpleInput(value=5))

        # Should have start, 3 progress, and completion events
        assert len(all_events) == 5
        assert all_events[0][0] == "StartEvent"
        assert all_events[1][0] == "ProgressEvent"
        assert all_events[2][0] == "ProgressEvent"
        assert all_events[3][0] == "ProgressEvent"
        assert all_events[4][0] == "CompletionEvent"
