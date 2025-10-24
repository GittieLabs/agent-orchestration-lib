"""FlowAdapter - Wraps a Flow to make it executable as an AgentBlock.

This module provides the FlowAdapter class that allows treating a Flow as an AgentBlock,
enabling nested flow execution and composition of complex multi-agent workflows.
"""

from typing import TypeVar, Optional, Any
from loguru import logger

from pydantic import BaseModel
from .agent_block import AgentBlock
from .flow import Flow
from .execution_context import ExecutionContext
from .event_emitter import EventEmitter

TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)


class FlowAdapter(AgentBlock[TInput, TOutput]):
    """Adapter that wraps a Flow to make it executable as an AgentBlock.

    This enables treating an entire flow as a single agent step, allowing
    for nested flow execution and hierarchical workflow composition.

    Example:
        ```python
        # Create a sub-flow for preprocessing
        preprocess_flow = Flow("preprocessing", context, emitter)
        preprocess_flow.add_agent(clean_agent)
        preprocess_flow.add_agent(validate_agent)

        # Wrap it as an agent block
        preprocess_step = FlowAdapter(preprocess_flow)

        # Use in a larger flow
        main_flow = Flow("main_workflow", context, emitter)
        main_flow.add_agent(extract_agent)
        main_flow.add_agent(preprocess_step)  # Nested flow!
        main_flow.add_agent(analyze_agent)

        result = await main_flow.execute_sequential(input_data)
        ```
    """

    def __init__(
        self,
        flow: Flow,
        name: Optional[str] = None,
        context: Optional[ExecutionContext] = None,
        emitter: Optional[EventEmitter] = None,
        **config: Any,
    ):
        """Initialize the flow adapter.

        Args:
            flow: The Flow instance to wrap
            name: Optional name for the adapter (defaults to flow_<flow_name>)
            context: Execution context (inherits from flow if None)
            emitter: Event emitter (inherits from flow if None)
            **config: Additional configuration options
        """
        # Use flow's context/emitter if not provided
        adapter_context = context or flow.context
        adapter_emitter = emitter or flow.emitter

        # Initialize agent block
        super().__init__(
            name=name or f"flow_{flow.name}",
            context=adapter_context,
            emitter=adapter_emitter,
            **config,
        )

        self.flow = flow

        # Update flow's context and emitter to match adapter
        self.flow.context = self.context
        self.flow.emitter = self.emitter

        # Propagate context/emitter to all agents in the flow
        for agent in self.flow.agents:
            agent.context = self.context
            agent.emitter = self.emitter

        logger.debug(
            f"Initialized FlowAdapter '{self.name}' wrapping flow '{flow.name}' "
            f"with {len(flow.agents)} agents"
        )

    def get_input_model(self) -> type[TInput]:
        """Get the input model from the first agent in the flow.

        Returns:
            Input model type of the first agent

        Raises:
            ValueError: If the flow has no agents
        """
        if not self.flow.agents:
            raise ValueError(f"Flow '{self.flow.name}' has no agents")

        return self.flow.agents[0].get_input_model()

    def get_output_model(self) -> type[TOutput]:
        """Get the output model from the last agent in the flow.

        Returns:
            Output model type of the last agent

        Raises:
            ValueError: If the flow has no agents
        """
        if not self.flow.agents:
            raise ValueError(f"Flow '{self.flow.name}' has no agents")

        return self.flow.agents[-1].get_output_model()

    async def process(self, input_data: TInput) -> TOutput:
        """Execute the wrapped flow sequentially.

        Args:
            input_data: Input data for the first agent in the flow

        Returns:
            Output from the last agent in the flow

        Raises:
            ValueError: If the flow has no agents
            Exception: If any agent in the flow fails
        """
        if not self.flow.agents:
            raise ValueError(f"Flow '{self.flow.name}' has no agents")

        logger.info(
            f"[{self.name}] Executing wrapped flow '{self.flow.name}' "
            f"with {len(self.flow.agents)} agents"
        )

        # Emit progress for flow start
        await self.emit_progress(
            stage="flow_start",
            progress=0.0,
            message=f"Starting flow '{self.flow.name}'"
        )

        # Execute flow sequentially
        try:
            result = await self.flow.execute_sequential(input_data)

            # Emit progress for flow completion
            await self.emit_progress(
                stage="flow_complete",
                progress=1.0,
                message=f"Completed flow '{self.flow.name}'"
            )

            return result

        except Exception as e:
            logger.error(f"[{self.name}] Flow '{self.flow.name}' failed: {e}")
            raise

    def __repr__(self) -> str:
        """String representation of the flow adapter."""
        return (
            f"FlowAdapter(name='{self.name}', "
            f"flow='{self.flow.name}', "
            f"agents={len(self.flow.agents)})"
        )
