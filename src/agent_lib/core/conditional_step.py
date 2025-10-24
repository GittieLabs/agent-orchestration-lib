"""ConditionalStep - Execute different agents based on condition evaluation.

This module provides the ConditionalStep class that enables if/else branching logic
in agent workflows. It evaluates a condition function against input data and executes
either the true_agent or false_agent accordingly.
"""

from typing import TypeVar, Generic, Optional, Any, Callable
from loguru import logger

from pydantic import BaseModel
from .agent_block import AgentBlock
from .execution_context import ExecutionContext
from .event_emitter import EventEmitter

TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)


class ConditionalStep(AgentBlock[TInput, TOutput]):
    """Execute different agents based on condition evaluation.

    Provides if/else branching logic for agent workflows. The condition function
    is evaluated against the input data, and either true_agent or false_agent
    is executed based on the result.

    Example:
        ```python
        def high_confidence(data: AnalysisOutput) -> bool:
            return data.confidence > 0.8

        conditional = ConditionalStep(
            name="confidence_check",
            condition=high_confidence,
            true_agent=auto_approve_agent,
            false_agent=manual_review_agent,
            context=context,
            emitter=emitter
        )

        result = await conditional.execute(analysis_output)
        ```
    """

    def __init__(
        self,
        name: str,
        condition: Callable[[TInput], bool],
        true_agent: AgentBlock[TInput, TOutput],
        false_agent: Optional[AgentBlock[TInput, TOutput]] = None,
        context: Optional[ExecutionContext] = None,
        emitter: Optional[EventEmitter] = None,
        **config: Any,
    ):
        """Initialize the conditional step.

        Args:
            name: Unique identifier for this conditional step
            condition: Function that takes input and returns True/False
            true_agent: Agent to execute if condition is True
            false_agent: Agent to execute if condition is False (optional)
            context: Execution context for dependency injection
            emitter: Event emitter for progress notifications
            **config: Additional configuration options
        """
        super().__init__(name=name, context=context, emitter=emitter, **config)
        self.condition = condition
        self.true_agent = true_agent
        self.false_agent = false_agent

        # Share context and emitter with child agents
        self.true_agent.context = self.context
        self.true_agent.emitter = self.emitter

        if self.false_agent:
            self.false_agent.context = self.context
            self.false_agent.emitter = self.emitter

        logger.debug(
            f"Initialized ConditionalStep '{name}' with "
            f"true_agent='{true_agent.name}', "
            f"false_agent='{false_agent.name if false_agent else None}'"
        )

    def get_input_model(self) -> type[TInput]:
        """Get the input model from the true agent."""
        return self.true_agent.get_input_model()

    def get_output_model(self) -> type[TOutput]:
        """Get the output model from the true agent."""
        return self.true_agent.get_output_model()

    async def process(self, input_data: TInput) -> TOutput:
        """Evaluate condition and execute appropriate agent.

        Args:
            input_data: Input data to evaluate and pass to selected agent

        Returns:
            Output from the executed agent

        Raises:
            ValueError: If condition is False and no false_agent is provided
        """
        # Evaluate condition
        try:
            condition_result = self.condition(input_data)
        except Exception as e:
            logger.error(f"[{self.name}] Condition evaluation failed: {e}")
            raise ValueError(f"Condition evaluation failed: {e}") from e

        # Log the condition result
        await self.emit_progress(
            stage="condition_evaluation",
            progress=0.5,
            message=f"Condition evaluated to: {condition_result}"
        )

        # Execute appropriate agent
        if condition_result:
            logger.info(f"[{self.name}] Condition True - executing '{self.true_agent.name}'")
            return await self.true_agent.execute(input_data)
        else:
            if self.false_agent:
                logger.info(f"[{self.name}] Condition False - executing '{self.false_agent.name}'")
                return await self.false_agent.execute(input_data)
            else:
                # No false agent, pass through the input unchanged
                logger.info(f"[{self.name}] Condition False - no false_agent, passing through")
                # Return input as output (type system may need adjustment here)
                return input_data  # type: ignore

    def __repr__(self) -> str:
        """String representation of the conditional step."""
        return (
            f"ConditionalStep(name='{self.name}', "
            f"true_agent='{self.true_agent.name}', "
            f"false_agent='{self.false_agent.name if self.false_agent else None}')"
        )
