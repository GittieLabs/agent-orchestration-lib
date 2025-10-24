"""Flow - Multi-agent workflow orchestration.

This module provides the Flow class for orchestrating multiple agents in sequence
or parallel, with support for shared context, event aggregation, and error handling.
"""

from typing import List, Optional, Any, Dict
import asyncio
from loguru import logger

from .agent_block import AgentBlock
from .execution_context import ExecutionContext
from .event_emitter import EventEmitter
from ..events import Event, ProgressEvent, ErrorEvent, CompletionEvent


class Flow:
    """Orchestrate multiple agents in sequential or parallel execution.

    Flow manages the execution of multiple agents, providing:
    - Sequential execution (one after another)
    - Parallel execution (all at once)
    - Shared execution context
    - Event aggregation from all agents
    - Error handling and propagation

    Example:
        ```python
        # Sequential flow
        flow = Flow(name="processing_pipeline")
        flow.add_agent(parse_agent)
        flow.add_agent(validate_agent)
        flow.add_agent(transform_agent)

        result = await flow.execute_sequential(input_data)

        # Parallel flow
        flow = Flow(name="parallel_processing")
        flow.add_agent(agent1)
        flow.add_agent(agent2)
        flow.add_agent(agent3)

        results = await flow.execute_parallel([input1, input2, input3])
        ```
    """

    def __init__(
        self,
        name: str,
        context: Optional[ExecutionContext] = None,
        emitter: Optional[EventEmitter] = None,
    ):
        """Initialize the flow.

        Args:
            name: Flow identifier
            context: Shared execution context (creates new if None)
            emitter: Shared event emitter (creates new if None)
        """
        self.name = name
        self.context = context or ExecutionContext()
        self.emitter = emitter or EventEmitter()
        self.agents: List[AgentBlock] = []

        logger.debug(f"Initialized flow '{name}'")

    def add_agent(self, agent: AgentBlock) -> "Flow":
        """Add an agent to the flow.

        Args:
            agent: Agent to add to the flow

        Returns:
            Self for method chaining
        """
        # Share context and emitter with the agent
        agent.context = self.context
        agent.emitter = self.emitter

        self.agents.append(agent)
        logger.debug(f"Added agent '{agent.name}' to flow '{self.name}'")
        return self

    async def execute_sequential(self, initial_input: Any) -> Any:
        """Execute agents sequentially, passing output to next agent.

        Each agent's output becomes the input for the next agent in the chain.

        Args:
            initial_input: Input for the first agent

        Returns:
            Output from the final agent

        Raises:
            Exception: If any agent fails
        """
        if not self.agents:
            raise ValueError("No agents in flow")

        logger.info(f"[{self.name}] Starting sequential execution with {len(self.agents)} agents")

        current_output = initial_input

        for i, agent in enumerate(self.agents):
            logger.debug(f"[{self.name}] Executing agent {i+1}/{len(self.agents)}: {agent.name}")

            try:
                current_output = await agent.execute(current_output)
            except Exception as e:
                logger.error(f"[{self.name}] Agent '{agent.name}' failed: {e}")
                raise

        logger.info(f"[{self.name}] Sequential execution completed successfully")
        return current_output

    async def execute_parallel(self, inputs: List[Any]) -> List[Any]:
        """Execute agents in parallel with separate inputs.

        Each agent runs concurrently with its own input. Agents share the same
        context and emitter but have isolated execution.

        Args:
            inputs: List of inputs, one per agent

        Returns:
            List of outputs, one per agent

        Raises:
            ValueError: If number of inputs doesn't match number of agents
            Exception: If any agent fails
        """
        if not self.agents:
            raise ValueError("No agents in flow")

        if len(inputs) != len(self.agents):
            raise ValueError(
                f"Number of inputs ({len(inputs)}) must match number of agents ({len(self.agents)})"
            )

        logger.info(f"[{self.name}] Starting parallel execution with {len(self.agents)} agents")

        # Create child contexts for each agent to provide isolation
        tasks = []
        for i, (agent, input_data) in enumerate(zip(self.agents, inputs)):
            # Create child context for isolation
            child_context = self.context.create_child(agent_index=i, agent_name=agent.name)
            agent.context = child_context

            # Create task
            task = asyncio.create_task(
                self._execute_with_logging(agent, input_data, i),
                name=f"{self.name}.{agent.name}"
            )
            tasks.append(task)

        # Wait for all agents to complete
        try:
            results = await asyncio.gather(*tasks)
            logger.info(f"[{self.name}] Parallel execution completed successfully")
            return results
        except Exception as e:
            logger.error(f"[{self.name}] Parallel execution failed: {e}")
            raise

    async def _execute_with_logging(
        self, agent: AgentBlock, input_data: Any, index: int
    ) -> Any:
        """Execute agent with logging wrapper.

        Args:
            agent: Agent to execute
            input_data: Input data
            index: Agent index in parallel execution

        Returns:
            Agent output
        """
        logger.debug(f"[{self.name}] Starting agent {index}: {agent.name}")
        try:
            result = await agent.execute(input_data)
            logger.debug(f"[{self.name}] Completed agent {index}: {agent.name}")
            return result
        except Exception as e:
            logger.error(f"[{self.name}] Failed agent {index}: {agent.name} - {e}")
            raise

    async def execute_parallel_same_input(self, input_data: Any) -> List[Any]:
        """Execute all agents in parallel with the same input.

        All agents receive the same input and run concurrently.

        Args:
            input_data: Input data for all agents

        Returns:
            List of outputs, one per agent

        Raises:
            Exception: If any agent fails
        """
        if not self.agents:
            raise ValueError("No agents in flow")

        # Create list of same input for each agent
        inputs = [input_data] * len(self.agents)
        return await self.execute_parallel(inputs)

    def clear_agents(self) -> "Flow":
        """Remove all agents from the flow.

        Returns:
            Self for method chaining
        """
        self.agents.clear()
        logger.debug(f"Cleared all agents from flow '{self.name}'")
        return self

    def get_agent_count(self) -> int:
        """Get the number of agents in the flow.

        Returns:
            Number of agents
        """
        return len(self.agents)

    def get_agent_names(self) -> List[str]:
        """Get names of all agents in the flow.

        Returns:
            List of agent names
        """
        return [agent.name for agent in self.agents]

    async def execute_with_fanout(
        self, input_data: Any, merge_fn: Optional[callable] = None
    ) -> Any:
        """Execute all agents in parallel with same input, then merge results.

        This is the fan-out/fan-in pattern where one input is processed by
        multiple agents in parallel, then results are merged.

        Args:
            input_data: Input for all agents
            merge_fn: Optional function to merge results. If None, returns list.

        Returns:
            Merged result or list of results

        Example:
            ```python
            def merge_scores(results):
                return sum(r.score for r in results) / len(results)

            avg_score = await flow.execute_with_fanout(data, merge_scores)
            ```
        """
        results = await self.execute_parallel_same_input(input_data)

        if merge_fn is None:
            return results

        logger.debug(f"[{self.name}] Merging results from {len(results)} agents")
        return merge_fn(results)

    def __repr__(self) -> str:
        """String representation of the flow."""
        return f"Flow(name='{self.name}', agents={len(self.agents)})"

    def __len__(self) -> int:
        """Return number of agents in flow."""
        return len(self.agents)
