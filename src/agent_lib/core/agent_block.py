"""AgentBlock - Base class for all agents in the orchestration framework.

This module provides the foundational AgentBlock class that implements the template
method pattern for agent execution. It integrates ExecutionContext for dependency
injection and EventEmitter for progress notifications.

Key features:
- Generic input/output types for type safety
- Template method pattern for execution lifecycle
- Automatic event emission for start, progress, completion, and errors
- Integration with ExecutionContext for shared state
- Retry support through RetryStrategy integration
- Comprehensive error handling and recovery
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Any, Dict
from datetime import datetime, timezone
import traceback
import time

from pydantic import BaseModel
from loguru import logger

from .execution_context import ExecutionContext
from .event_emitter import EventEmitter
from ..events import (
    StartEvent,
    ProgressEvent,
    ErrorEvent,
    CompletionEvent,
)


# Generic type variables for input/output
TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)


class AgentBlock(ABC, Generic[TInput, TOutput]):
    """Base class for all agents providing execution lifecycle and event emission.

    This class implements the template method pattern where the main execution flow
    is defined in execute(), but specific behavior is customized through abstract
    methods that subclasses must implement.

    Lifecycle:
    1. on_start() - Called before execution begins
    2. validate_input() - Validate the input data
    3. process() - Main processing logic (abstract, must implement)
    4. validate_output() - Validate the output data
    5. on_complete() - Called after successful execution
    6. on_error() - Called if an error occurs

    Example:
        ```python
        class GreetingInput(BaseModel):
            name: str

        class GreetingOutput(BaseModel):
            message: str

        class GreetingAgent(AgentBlock[GreetingInput, GreetingOutput]):
            async def process(self, input_data: GreetingInput) -> GreetingOutput:
                return GreetingOutput(message=f"Hello, {input_data.name}!")

        # Usage
        context = ExecutionContext()
        agent = GreetingAgent(name="greeter", context=context)
        result = await agent.execute(GreetingInput(name="Alice"))
        ```
    """

    def __init__(
        self,
        name: str,
        context: Optional[ExecutionContext] = None,
        emitter: Optional[EventEmitter] = None,
        **config: Any,
    ):
        """Initialize the agent.

        Args:
            name: Unique identifier for this agent instance
            context: Execution context for dependency injection (creates new if None)
            emitter: Event emitter for progress notifications (creates new if None)
            **config: Additional configuration options stored in self.config
        """
        self.name = name
        self.context = context or ExecutionContext()
        self.emitter = emitter or EventEmitter()
        self.config = config

        # Execution tracking
        self._start_time: Optional[float] = None
        self._is_running = False

        logger.debug(f"Initialized agent '{name}' with config: {config}")

    @abstractmethod
    async def process(self, input_data: TInput) -> TOutput:
        """Main processing logic - must be implemented by subclasses.

        This is the core method where the agent's specific logic is implemented.
        It receives validated input and should return valid output.

        Args:
            input_data: Validated input data

        Returns:
            Output data that will be validated

        Raises:
            Any exception will be caught and handled by the execute() method
        """
        pass

    async def validate_input(self, input_data: TInput) -> None:
        """Validate input data before processing.

        Override this method to add custom validation logic beyond Pydantic validation.
        Raise an exception if validation fails.

        Args:
            input_data: Input data to validate

        Raises:
            ValueError: If validation fails
        """
        # Default: Pydantic validation already occurred, no additional validation needed
        pass

    async def validate_output(self, output_data: TOutput) -> None:
        """Validate output data after processing.

        Override this method to add custom validation logic beyond Pydantic validation.
        Raise an exception if validation fails.

        Args:
            output_data: Output data to validate

        Raises:
            ValueError: If validation fails
        """
        # Default: Pydantic validation already occurred, no additional validation needed
        pass

    async def on_start(self, input_data: TInput) -> None:
        """Lifecycle hook called before processing begins.

        Override this to perform setup tasks like initializing resources,
        logging, or setting up monitoring.

        Args:
            input_data: The input that will be processed
        """
        pass

    async def on_complete(self, output_data: TOutput) -> None:
        """Lifecycle hook called after successful processing.

        Override this to perform cleanup tasks, logging, or metrics collection.

        Args:
            output_data: The output that was produced
        """
        pass

    async def on_error(self, error: Exception, input_data: TInput) -> None:
        """Lifecycle hook called when an error occurs.

        Override this to perform error-specific cleanup or logging.

        Args:
            error: The exception that was raised
            input_data: The input that was being processed
        """
        pass

    async def emit_progress(
        self,
        stage: str,
        progress: float,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a progress event.

        Call this method during processing to report progress updates.

        Args:
            stage: Current processing stage name
            progress: Progress value between 0.0 and 1.0
            message: Human-readable progress message
            details: Optional additional details about progress
        """
        event = ProgressEvent(
            source=self.name,
            stage=stage,
            progress=progress,
            message=message,
            details=details or {},
        )
        await self.emitter.emit(event)
        logger.debug(f"[{self.name}] Progress: {stage} - {progress*100:.1f}% - {message}")

    async def execute(self, input_data: TInput) -> TOutput:
        """Execute the agent with the given input.

        This is the main entry point for running the agent. It implements the
        template method pattern, orchestrating the execution lifecycle:

        1. Emit start event
        2. Call on_start() hook
        3. Validate input
        4. Call process() (abstract method)
        5. Validate output
        6. Call on_complete() hook
        7. Emit completion event

        If an error occurs at any stage, it will:
        1. Call on_error() hook
        2. Emit error event
        3. Re-raise the exception

        Args:
            input_data: Input data for processing

        Returns:
            Processed output data

        Raises:
            Exception: Any exception from processing or validation
        """
        if self._is_running:
            raise RuntimeError(f"Agent '{self.name}' is already running")

        self._is_running = True
        self._start_time = time.time()

        try:
            # Emit start event
            await self._emit_start(input_data)

            # Call start hook
            await self.on_start(input_data)

            # Validate input
            logger.debug(f"[{self.name}] Validating input")
            await self.validate_input(input_data)

            # Process
            logger.info(f"[{self.name}] Starting processing")
            output_data = await self.process(input_data)

            # Validate output
            logger.debug(f"[{self.name}] Validating output")
            await self.validate_output(output_data)

            # Call completion hook
            await self.on_complete(output_data)

            # Emit completion event
            await self._emit_completion(success=True)

            logger.info(f"[{self.name}] Completed successfully")
            return output_data

        except Exception as error:
            # Call error hook
            await self.on_error(error, input_data)

            # Emit error event
            await self._emit_error(error)

            # Emit failed completion event
            await self._emit_completion(success=False)

            logger.error(f"[{self.name}] Failed with error: {error}")
            raise

        finally:
            self._is_running = False

    async def _emit_start(self, input_data: TInput) -> None:
        """Emit start event with input summary."""
        event = StartEvent(
            source=self.name,
            input_summary=self._get_input_summary(input_data),
        )
        await self.emitter.emit(event)
        logger.debug(f"[{self.name}] Started")

    async def _emit_completion(self, success: bool) -> None:
        """Emit completion event with duration and success status."""
        duration = time.time() - (self._start_time or time.time())

        event = CompletionEvent(
            source=self.name,
            duration_seconds=duration,
            success=success,
        )
        await self.emitter.emit(event)

    async def _emit_error(self, error: Exception) -> None:
        """Emit error event with exception details."""
        event = ErrorEvent(
            source=self.name,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            recoverable=self._is_recoverable_error(error),
        )
        await self.emitter.emit(event)

    def _get_input_summary(self, input_data: TInput) -> str:
        """Generate a summary of the input data.

        Override this to provide a more meaningful summary for your specific agent.
        """
        return f"{type(input_data).__name__} input"

    def _is_recoverable_error(self, error: Exception) -> bool:
        """Determine if an error is recoverable.

        Override this to customize error recoverability logic.
        """
        # Default: most errors are not recoverable
        return False

    def __repr__(self) -> str:
        """String representation of the agent."""
        status = "running" if self._is_running else "idle"
        return f"AgentBlock(name='{self.name}', status={status})"
