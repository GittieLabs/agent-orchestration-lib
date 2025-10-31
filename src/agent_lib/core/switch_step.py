"""Switch step for multi-branch conditional logic in agent workflows.

This module provides a switch/case pattern for routing execution to different
agents based on a value or condition, similar to switch statements in programming
languages.
"""

from typing import Any, Callable, Dict, Generic, Optional, TypeVar

from loguru import logger

from .agent_block import AgentBlock
from .event_emitter import EventEmitter
from .execution_context import ExecutionContext

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class SwitchStep(AgentBlock[TInput, TOutput], Generic[TInput, TOutput]):
    """Switch/case conditional step for multi-branch execution.

    Routes execution to different agents based on a selector function that
    returns a case key. Similar to switch/case statements in programming.

    Attributes:
        selector: Function that takes input and returns a case key
        cases: Dictionary mapping case keys to agents
        default_agent: Optional agent to execute if no case matches
        name: Name of the switch step

    Examples:
        Simple switch based on document type:
        >>> from agent_lib import ExecutionContext, EventEmitter, SwitchStep
        >>> from pydantic import BaseModel
        >>>
        >>> class Document(BaseModel):
        ...     type: str
        ...     content: str
        >>>
        >>> def select_processor(doc: Document) -> str:
        ...     return doc.type
        >>>
        >>> switch = SwitchStep(
        ...     name="document_router",
        ...     selector=select_processor,
        ...     cases={
        ...         "pdf": pdf_processor,
        ...         "docx": docx_processor,
        ...         "txt": txt_processor,
        ...     },
        ...     default_agent=unknown_format_handler,
        ...     context=context,
        ...     emitter=emitter
        ... )
        >>>
        >>> result = await switch.execute(Document(type="pdf", content="..."))

        Switch with complex condition:
        >>> class OrderData(BaseModel):
        ...     amount: float
        ...     priority: str
        ...
        >>> def categorize_order(order: OrderData) -> str:
        ...     if order.amount > 10000:
        ...         return "high_value"
        ...     elif order.priority == "urgent":
        ...         return "urgent"
        ...     elif order.amount > 1000:
        ...         return "medium_value"
        ...     else:
        ...         return "standard"
        >>>
        >>> switch = SwitchStep(
        ...     name="order_router",
        ...     selector=categorize_order,
        ...     cases={
        ...         "high_value": high_value_processor,
        ...         "urgent": urgent_processor,
        ...         "medium_value": standard_processor,
        ...         "standard": standard_processor,
        ...     },
        ...     context=context,
        ...     emitter=emitter
        ... )

        Switch without default (raises error if no match):
        >>> switch = SwitchStep(
        ...     name="strict_router",
        ...     selector=lambda x: x.category,
        ...     cases={
        ...         "A": agent_a,
        ...         "B": agent_b,
        ...         "C": agent_c,
        ...     },
        ...     # No default_agent - will raise ValueError if no match
        ...     context=context,
        ...     emitter=emitter
        ... )
    """

    def __init__(
        self,
        name: str,
        selector: Callable[[TInput], str],
        cases: Dict[str, AgentBlock[TInput, TOutput]],
        context: ExecutionContext,
        emitter: EventEmitter,
        default_agent: Optional[AgentBlock[TInput, TOutput]] = None,
        **kwargs: Any,
    ):
        """Initialize the switch step.

        Args:
            name: Name of the switch step
            selector: Function that takes input and returns a case key
            cases: Dictionary mapping case keys to agents to execute
            context: Execution context for dependency injection
            emitter: Event emitter for observability
            default_agent: Optional agent to execute if no case matches.
                          If None and no case matches, raises ValueError.
            **kwargs: Additional arguments passed to AgentBlock
        """
        super().__init__(name, context, emitter, **kwargs)
        self.selector = selector
        self.cases = cases
        self.default_agent = default_agent

        if not cases:
            raise ValueError("SwitchStep requires at least one case")

        logger.debug(
            f"Created SwitchStep '{name}' with {len(cases)} cases "
            f"and {'a' if default_agent else 'no'} default agent"
        )

    async def process(self, input_data: TInput) -> TOutput:
        """Process input by routing to appropriate agent based on selector.

        Args:
            input_data: Input to evaluate and route

        Returns:
            Output from the selected agent

        Raises:
            ValueError: If no case matches and no default agent is provided
            Exception: Any exception raised by the selected agent
        """
        # Evaluate selector to get case key
        try:
            case_key = self.selector(input_data)
        except Exception as e:
            logger.error(f"SwitchStep '{self.name}': Selector function failed: {e}")
            raise ValueError(f"Selector function failed: {e}") from e

        logger.debug(f"SwitchStep '{self.name}': Selector returned '{case_key}'")

        # Emit progress with selected case
        self.emit_progress(
            "case_selected", {"case": case_key, "available_cases": list(self.cases.keys())}
        )

        # Find matching agent
        selected_agent = self.cases.get(case_key)

        if selected_agent is None:
            if self.default_agent is not None:
                logger.info(
                    f"SwitchStep '{self.name}': No case matched '{case_key}', "
                    f"using default agent '{self.default_agent.name}'"
                )
                self.emit_progress(
                    "using_default",
                    {
                        "case": case_key,
                        "default_agent": self.default_agent.name,
                    },
                )
                selected_agent = self.default_agent
            else:
                available_cases = ", ".join(self.cases.keys())
                error_msg = (
                    f"No case matched '{case_key}' and no default agent provided. "
                    f"Available cases: {available_cases}"
                )
                logger.error(f"SwitchStep '{self.name}': {error_msg}")
                raise ValueError(error_msg)
        else:
            logger.info(
                f"SwitchStep '{self.name}': Matched case '{case_key}', "
                f"executing agent '{selected_agent.name}'"
            )
            self.emit_progress(
                "executing_case",
                {
                    "case": case_key,
                    "agent": selected_agent.name,
                },
            )

        # Execute selected agent
        try:
            result = await selected_agent.execute(input_data)
            logger.info(
                f"SwitchStep '{self.name}': Agent '{selected_agent.name}' "
                f"completed successfully"
            )
            return result
        except Exception as e:
            logger.error(f"SwitchStep '{self.name}': Agent '{selected_agent.name}' failed: {e}")
            raise


def create_switch(
    name: str,
    selector: Callable[[Any], str],
    cases: Dict[str, AgentBlock],
    context: ExecutionContext,
    emitter: EventEmitter,
    default_agent: Optional[AgentBlock] = None,
) -> SwitchStep:
    """Helper function to create a SwitchStep.

    Args:
        name: Name of the switch step
        selector: Function that evaluates input and returns a case key
        cases: Dictionary mapping case keys to agents
        context: Execution context
        emitter: Event emitter
        default_agent: Optional default agent if no case matches

    Returns:
        Configured SwitchStep instance

    Examples:
        >>> switch = create_switch(
        ...     name="router",
        ...     selector=lambda x: x.type,
        ...     cases={
        ...         "type_a": agent_a,
        ...         "type_b": agent_b,
        ...     },
        ...     context=context,
        ...     emitter=emitter,
        ...     default_agent=fallback_agent
        ... )
    """
    return SwitchStep(
        name=name,
        selector=selector,
        cases=cases,
        context=context,
        emitter=emitter,
        default_agent=default_agent,
    )
