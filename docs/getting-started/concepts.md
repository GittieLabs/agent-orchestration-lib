# Core Concepts

Understanding these core concepts will help you build effective agent workflows.

## ExecutionContext

**Purpose**: Dependency injection container for managing shared services and state.

```python
context = ExecutionContext()
context.register_singleton("database", db_connection)
context.register_singleton("llm_client", openai_client)

# Access in agents
db = context.get("database")
```

**Key Features**:
- Singleton services (shared across agents)
- Factory functions (new instance per access)
- Child contexts (isolated for parallel execution)
- Type-safe service retrieval

[Learn more →](../guide/execution-context.md)

## AgentBlock

**Purpose**: Base class for all agent implementations with validation and lifecycle hooks.

```python
class MyAgent(AgentBlock[InputModel, OutputModel]):
    async def process(self, input_data: InputModel) -> OutputModel:
        # Your logic here
        pass
```

**Automatic Features**:
- Input/output validation with Pydantic
- Event emission for progress tracking
- Error handling and cleanup
- Pre/post execution hooks

[Learn more →](../guide/agent-block.md)

## EventEmitter

**Purpose**: Pub/sub pattern for progress tracking and real-time notifications.

```python
emitter = EventEmitter()
emitter.subscribe("progress", lambda e: print(e.message))

# In agents
await self.emit_progress("stage", 0.5, "Processing...")
```

**Use Cases**:
- Real-time progress updates
- Error notifications
- Token usage tracking
- Custom metrics

[Learn more →](../guide/event-emitter.md)

## Flow

**Purpose**: Orchestrate multiple agents in complex workflows.

```python
flow = Flow("my_workflow", context, emitter)
flow.add_agent(extractor)
flow.add_agent(parser)
flow.add_agent(analyzer)

result = await flow.execute_sequential(initial_input)
```

**Execution Modes**:
- Sequential (one after another)
- Parallel (concurrent execution)
- Conditional (branch based on data)

[Learn more →](../guide/flow.md)

## RetryStrategy

**Purpose**: Handle transient failures with configurable retry logic.

```python
retry = ExponentialBackoffRetry(
    max_attempts=3,
    base_delay=1.0
)

result = await retry.execute_with_retry(
    agent.execute,
    input_data
)
```

**Strategies**:
- Exponential backoff
- LLM fallback chains
- Custom retry conditions

[Learn more →](../guide/retry.md)

## Design Patterns

### Execution Sandwich

Every agent execution follows this pattern:

1. Validate input (Pydantic)
2. Emit start event
3. Execute logic
4. Validate output (Pydantic)
5. Emit completion/error
6. Cleanup resources

### Dependency Injection

Services are explicitly registered and retrieved:

```python
# Register
context.register_singleton("service", instance)

# Retrieve
service = context.get("service")
```

No global state, easy testing, parallel-safe.

### Event-Driven Architecture

Agents emit events, adapters handle them:

```python
# Agent emits
await self.emit_progress("stage", 0.5, "message")

# Multiple subscribers
emitter.subscribe("progress", database_adapter)
emitter.subscribe("progress", websocket_adapter)
emitter.subscribe("progress", logging_adapter)
```

## Next Steps

- [Build a workflow](../guide/flow.md)
- [Add retry logic](../guide/retry.md)
- [View examples](../examples/document-processing.md)
