# Architecture Overview

The Agent Orchestration Library is built on production-proven patterns extracted from real-world AI agent systems.

## Design Principles

1. **Type Safety First**: Pydantic models for all inputs/outputs
2. **Explicit Dependencies**: No hidden globals or magic
3. **Event-Driven**: Observable through comprehensive event emission
4. **Composability**: Agents as building blocks for complex workflows
5. **Testability**: Each component independently testable
6. **Production-Ready**: Patterns from real systems

## Component Architecture

```
┌─────────────────────────────────────────┐
│         ExecutionContext                │
│    (Dependency Injection)               │
├─────────────────────────────────────────┤
│  - Singleton services                   │
│  - Factory functions                    │
│  - Child contexts                       │
└─────────────────────────────────────────┘
              │
              │ provides
              ▼
┌─────────────────────────────────────────┐
│            AgentBlock                   │
│       (Base Agent Class)                │
├─────────────────────────────────────────┤
│  - Input/output validation              │
│  - Event emission                       │
│  - Lifecycle hooks                      │
│  - Error handling                       │
└─────────────────────────────────────────┘
              │
              │ emits to
              ▼
┌─────────────────────────────────────────┐
│          EventEmitter                   │
│       (Pub/Sub System)                  │
├─────────────────────────────────────────┤
│  - Event subscriptions                  │
│  - Multiple handlers                    │
│  - Event filtering                      │
└─────────────────────────────────────────┘
              │
              │ consumed by
              ▼
┌─────────────────────────────────────────┐
│         Event Adapters                  │
│  (Database, WebSocket, Logging)         │
└─────────────────────────────────────────┘
```

## Key Patterns

### Template Method Pattern

`AgentBlock` defines the execution skeleton:

```python
async def execute(self, input_data):
    # 1. Validate input
    validated = self.get_input_model().validate(input_data)

    # 2. Emit start
    await self.emit_start()

    # 3. Execute (subclass implements)
    result = await self.process(validated)

    # 4. Validate output
    validated_output = self.get_output_model().validate(result)

    # 5. Emit completion
    await self.emit_complete()

    return validated_output
```

### Strategy Pattern

Pluggable retry strategies:

```python
class RetryStrategy(ABC):
    async def execute_with_retry(self, fn, *args):
        pass

class ExponentialBackoffRetry(RetryStrategy):
    # Implementation
    pass

class LLMFallbackRetry(RetryStrategy):
    # Implementation
    pass
```

### Dependency Injection

Explicit service management:

```python
# No globals
context = ExecutionContext()
context.register_singleton("db", database)

# Agents retrieve services
class MyAgent(AgentBlock):
    async def process(self, input_data):
        db = self.context.get("db")
        # Use db
```

## Data Flow

```
Input Data (dict/model)
        │
        ▼
  Pydantic Validation
        │
        ▼
   AgentBlock.execute()
        │
        ├─> Emit start event
        │
        ├─> process() [your logic]
        │
        ├─> Emit progress events
        │
        ├─> Output validation
        │
        └─> Emit completion
        │
        ▼
  Validated Output
```

## Execution Lifecycle

1. **Initialization**: Context and emitter setup
2. **Registration**: Services registered in context
3. **Agent Creation**: Agents receive context/emitter
4. **Execution**: Input → Validation → Process → Output
5. **Events**: Progress tracked via event system
6. **Cleanup**: Resources released automatically

## Testing Strategy

Each component is independently testable:

```python
# Test ExecutionContext
context = ExecutionContext()
context.register_singleton("test", "value")
assert context.get("test") == "value"

# Test AgentBlock
agent = MyAgent("test", mock_context, mock_emitter)
result = await agent.execute(test_input)
assert result == expected_output

# Test EventEmitter
emitter = EventEmitter()
called = False
def handler(e): called = True
emitter.subscribe("test", handler)
await emitter.emit(Event(type="test"))
assert called
```

## Next Steps

- [Design Patterns](patterns.md) - Deep dive into patterns
- [Best Practices](best-practices.md) - Production recommendations
- [API Reference](../api/core.md) - Complete API docs
