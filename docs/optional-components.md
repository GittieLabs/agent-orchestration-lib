# Optional Components - Design Documentation

This document outlines the design for optional components that extend the core agent-orchestration-lib functionality.

## Table of Contents

1. [Conditional Flow Execution](#1-conditional-flow-execution)
2. [LLM Fallback Strategy](#2-llm-fallback-strategy)
3. [Integration Components](#3-integration-components)
4. [Event Adapters](#4-event-adapters)
5. [Utility Functions](#5-utility-functions)
6. [Advanced Flow Patterns](#6-advanced-flow-patterns)
7. [Performance & Monitoring](#7-performance--monitoring)
8. [Agent State Management](#8-agent-state-management)
9. [Advanced Context Features](#9-advanced-context-features)
10. [Testing Utilities](#10-testing-utilities)

---

## 1. Conditional Flow Execution

### Overview
Enable dynamic routing and conditional execution within flows based on runtime conditions.

### Design

```python
from typing import Callable, Any
from agent_lib.core import Flow, AgentBlock

class ConditionalFlow(Flow):
    """Flow with conditional branching support."""

    async def execute_conditional(
        self,
        input_data: Any,
        condition: Callable[[Any], bool],
        if_true: AgentBlock,
        if_false: AgentBlock
    ) -> Any:
        """Execute different agents based on condition."""
        pass

    async def execute_switch(
        self,
        input_data: Any,
        router: Callable[[Any], str],
        cases: Dict[str, AgentBlock],
        default: Optional[AgentBlock] = None
    ) -> Any:
        """Route to different agents based on router function."""
        pass
```

### Use Cases
- Route to different processing agents based on input type
- Implement retry/fallback logic
- Dynamic pipeline construction

### Testing Requirements
- Test if/else branching
- Test switch/case routing
- Test condition evaluation
- Test error handling in conditions

---

## 2. LLM Fallback Strategy

### Overview
Intelligent retry strategy that falls back to alternative LLM providers or models on failure.

### Design

```python
from typing import List, Optional, Callable
from agent_lib.retry import RetryStrategy

class LLMFallbackRetry(RetryStrategy):
    """Retry strategy with LLM model fallback."""

    def __init__(
        self,
        max_attempts: int = 3,
        models: List[str] = None,  # ["gpt-4", "claude-3", "gpt-3.5"]
        fallback_fn: Optional[Callable[[str], Any]] = None,
        base_delay: float = 1.0
    ):
        """Initialize LLM fallback retry strategy.

        Args:
            max_attempts: Maximum total attempts across all models
            models: List of model names to try in order
            fallback_fn: Function to switch models (receives model name)
            base_delay: Base delay between attempts
        """
        pass

    def get_next_model(self, attempt: int) -> Optional[str]:
        """Get the model to use for the given attempt."""
        pass
```

### Use Cases
- Primary model rate limited → fallback to alternative
- Primary model unavailable → fallback to backup
- Cost optimization (try cheaper model first)
- Quality fallback (try best model first, fallback to faster)

### Testing Requirements
- Test model rotation on failure
- Test model selection logic
- Test integration with AgentBlock
- Test exhaustion of all models

---

## 3. Integration Components

### Overview
Adapters for popular LLM frameworks and libraries.

### 3.1 LangChain Integration

```python
from agent_lib.integrations.langchain import LangChainAgentAdapter

class LangChainAgentAdapter(AgentBlock):
    """Adapter to use LangChain chains as agents."""

    def __init__(
        self,
        chain: Any,  # LangChain chain
        name: str,
        input_mapper: Optional[Callable] = None,
        output_mapper: Optional[Callable] = None
    ):
        """Wrap a LangChain chain as an AgentBlock."""
        pass
```

### 3.2 LlamaIndex Integration

```python
from agent_lib.integrations.llama_index import LlamaIndexAgentAdapter

class LlamaIndexAgentAdapter(AgentBlock):
    """Adapter to use LlamaIndex agents."""

    def __init__(
        self,
        query_engine: Any,  # LlamaIndex query engine
        name: str
    ):
        """Wrap a LlamaIndex query engine as an AgentBlock."""
        pass
```

### 3.3 OpenAI Helpers

```python
from agent_lib.integrations.openai import (
    OpenAIAgent,
    count_tokens,
    estimate_cost
)

class OpenAIAgent(AgentBlock):
    """Pre-configured agent for OpenAI API calls."""

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        **kwargs
    ):
        pass
```

### Testing Requirements
- Mock external library dependencies
- Test adapter input/output mapping
- Test error handling for library-specific errors
- Test configuration options

---

## 4. Event Adapters

### Overview
Adapters to route events to external systems.

### 4.1 Structured Logging Adapter

```python
from agent_lib.events.adapters import StructuredLogAdapter

class StructuredLogAdapter:
    """Adapter to send events to structured logging."""

    def __init__(
        self,
        logger: Any,  # structlog logger
        include_events: Optional[List[str]] = None,
        format_fn: Optional[Callable] = None
    ):
        pass

    def attach_to_emitter(self, emitter: EventEmitter) -> None:
        """Subscribe to emitter and forward to logger."""
        pass
```

### 4.2 Metrics Adapter

```python
from agent_lib.events.adapters import MetricsAdapter

class MetricsAdapter:
    """Adapter to send metrics to monitoring systems."""

    def __init__(
        self,
        metrics_client: Any,  # Prometheus, DataDog, etc.
        prefix: str = "agent_lib"
    ):
        pass

    def record_agent_execution(self, event: CompletionEvent) -> None:
        """Record agent execution metrics."""
        pass
```

### 4.3 Webhook Adapter

```python
from agent_lib.events.adapters import WebhookAdapter

class WebhookAdapter:
    """Send events to webhooks."""

    def __init__(
        self,
        url: str,
        event_types: Optional[List[str]] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        pass
```

### Testing Requirements
- Mock external service calls
- Test event filtering
- Test format transformations
- Test error handling for network failures

---

## 5. Utility Functions

### Overview
Common helper functions for agent development.

### 5.1 Data Transformation

```python
from agent_lib.utils import (
    flatten_dict,
    unflatten_dict,
    merge_deep,
    extract_fields
)

def flatten_dict(nested: Dict, separator: str = ".") -> Dict:
    """Flatten nested dictionary to single level."""
    pass

def extract_fields(data: Dict, fields: List[str]) -> Dict:
    """Extract specific fields from dictionary."""
    pass
```

### 5.2 Validation Helpers

```python
from agent_lib.utils import (
    validate_schema,
    coerce_types,
    sanitize_input
)

def validate_schema(data: Dict, schema: Dict) -> bool:
    """Validate data against JSON schema."""
    pass
```

### Testing Requirements
- Test edge cases (empty inputs, None, etc.)
- Test type coercion
- Test nested structures
- Test error conditions

---

## 6. Advanced Flow Patterns

### Overview
Complex orchestration patterns beyond basic sequential/parallel.

### 6.1 Map-Reduce Flow

```python
class MapReduceFlow(Flow):
    """Map-reduce pattern for parallel processing and aggregation."""

    async def execute_map_reduce(
        self,
        input_data: List[Any],
        map_agent: AgentBlock,
        reduce_agent: AgentBlock
    ) -> Any:
        """Execute map-reduce pattern."""
        pass
```

### 6.2 Loop/Iteration Flow

```python
class IterativeFlow(Flow):
    """Execute agents in a loop until condition is met."""

    async def execute_until(
        self,
        initial_input: Any,
        agent: AgentBlock,
        condition: Callable[[Any], bool],
        max_iterations: int = 10
    ) -> Any:
        """Execute agent repeatedly until condition is true."""
        pass
```

### 6.3 Error Recovery Flow

```python
class ResilientFlow(Flow):
    """Flow with automatic error recovery."""

    async def execute_with_recovery(
        self,
        input_data: Any,
        primary_agent: AgentBlock,
        recovery_agent: AgentBlock
    ) -> Any:
        """Execute with automatic fallback on errors."""
        pass
```

### Testing Requirements
- Test loop termination conditions
- Test max iterations
- Test error recovery paths
- Test state passing between iterations

---

## 7. Performance & Monitoring

### Overview
Built-in performance tracking and monitoring.

### 7.1 Performance Profiler

```python
from agent_lib.monitoring import PerformanceProfiler

class PerformanceProfiler:
    """Profile agent execution performance."""

    def attach_to_agent(self, agent: AgentBlock) -> None:
        """Attach profiler to agent."""
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected performance metrics."""
        pass
```

### 7.2 Resource Tracker

```python
from agent_lib.monitoring import ResourceTracker

class ResourceTracker:
    """Track resource usage (memory, CPU, tokens)."""

    def track_execution(self, agent: AgentBlock) -> ContextManager:
        """Context manager for tracking resource usage."""
        pass
```

### Testing Requirements
- Test metric collection
- Test timer accuracy
- Test resource measurement
- Test aggregation functions

---

## 8. Agent State Management

### Overview
Stateful agents with persistence and recovery.

### 8.1 Stateful Agent

```python
from agent_lib.state import StatefulAgent

class StatefulAgent(AgentBlock):
    """Agent with persistent state."""

    def __init__(
        self,
        name: str,
        state_backend: StateBackend,
        **kwargs
    ):
        pass

    async def save_state(self, state: Dict[str, Any]) -> None:
        """Save agent state."""
        pass

    async def load_state(self) -> Dict[str, Any]:
        """Load agent state."""
        pass
```

### 8.2 State Backends

```python
class StateBackend(ABC):
    """Abstract state storage backend."""

    @abstractmethod
    async def save(self, key: str, value: Any) -> None:
        pass

    @abstractmethod
    async def load(self, key: str) -> Any:
        pass

class MemoryStateBackend(StateBackend):
    """In-memory state storage."""
    pass

class FileStateBackend(StateBackend):
    """File-based state storage."""
    pass
```

### Testing Requirements
- Test state persistence
- Test state recovery
- Test concurrent access
- Test state versioning

---

## 9. Advanced Context Features

### Overview
Enhanced ExecutionContext capabilities.

### 9.1 Context Middleware

```python
from agent_lib.context import ContextMiddleware

class ContextMiddleware(ABC):
    """Middleware for context operations."""

    @abstractmethod
    def before_get(self, key: str, context: ExecutionContext) -> None:
        """Called before getting a value."""
        pass

    @abstractmethod
    def after_set(self, key: str, value: Any, context: ExecutionContext) -> None:
        """Called after setting a value."""
        pass
```

### 9.2 Context Serialization

```python
from agent_lib.context import serialize_context, deserialize_context

def serialize_context(context: ExecutionContext) -> Dict[str, Any]:
    """Serialize context for distributed execution."""
    pass

def deserialize_context(data: Dict[str, Any]) -> ExecutionContext:
    """Deserialize context from data."""
    pass
```

### Testing Requirements
- Test middleware chain execution
- Test serialization/deserialization
- Test context versioning
- Test backward compatibility

---

## 10. Testing Utilities

### Overview
Utilities to make testing agents easier.

### 10.1 Mock Agents

```python
from agent_lib.testing import MockAgent, AgentSpy

class MockAgent(AgentBlock):
    """Mock agent for testing."""

    def __init__(
        self,
        name: str,
        return_value: Any = None,
        side_effect: Optional[Exception] = None
    ):
        pass

class AgentSpy(AgentBlock):
    """Spy wrapper to track agent calls."""

    def __init__(self, agent: AgentBlock):
        pass

    @property
    def call_count(self) -> int:
        pass

    @property
    def calls(self) -> List[Any]:
        pass
```

### 10.2 Test Fixtures

```python
from agent_lib.testing import (
    create_test_context,
    create_test_emitter,
    create_test_flow
)

@pytest.fixture
def test_context():
    """Fixture providing test ExecutionContext."""
    return create_test_context()
```

### 10.3 Assertion Utilities

```python
from agent_lib.testing import (
    assert_agent_called,
    assert_event_emitted,
    assert_context_has
)

def assert_agent_called(agent: AgentBlock, times: int = 1) -> None:
    """Assert agent was called specified number of times."""
    pass
```

### Testing Requirements
- Test mock agent behavior
- Test spy tracking
- Test assertion helpers
- Test fixture creation

---

## Implementation Order

Based on dependencies and value, suggested implementation order:

1. **Utility Functions** - Foundation for other components
2. **Testing Utilities** - Needed for testing other components
3. **Event Adapters** - Extends observability
4. **Conditional Flow Execution** - Core flow enhancement
5. **LLM Fallback Strategy** - High-value retry enhancement
6. **Advanced Flow Patterns** - Builds on conditional flows
7. **Performance & Monitoring** - Uses event adapters
8. **Integration Components** - Requires stable core
9. **Agent State Management** - Complex feature
10. **Advanced Context Features** - Enhancement feature

---

## Success Criteria

Each component must have:

- ✅ Complete implementation
- ✅ Comprehensive unit tests (>95% coverage)
- ✅ Integration tests where applicable
- ✅ Documentation with examples
- ✅ Type hints and validation
- ✅ Error handling
- ✅ Performance considerations

