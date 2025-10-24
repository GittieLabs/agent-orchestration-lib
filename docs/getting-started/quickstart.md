# Quick Start

This tutorial will walk you through creating your first agent using the Agent Orchestration Library.

## Step 1: Define Your Data Models

All agents use Pydantic models for type-safe inputs and outputs:

```python
from pydantic import BaseModel

class TextInput(BaseModel):
    text: str
    max_length: int = 100

class SummaryOutput(BaseModel):
    summary: str
    word_count: int
```

## Step 2: Create Your Agent

Extend `AgentBlock` and implement the required methods:

```python
from agent_lib import AgentBlock

class SummaryAgent(AgentBlock[TextInput, SummaryOutput]):
    def get_input_model(self):
        return TextInput

    def get_output_model(self):
        return SummaryOutput

    async def process(self, input_data: TextInput) -> SummaryOutput:
        # Your agent logic here
        words = input_data.text.split()[:input_data.max_length]
        summary = " ".join(words)

        return SummaryOutput(
            summary=summary,
            word_count=len(words)
        )
```

## Step 3: Set Up Context and Events

```python
from agent_lib import ExecutionContext, EventEmitter

# Create context for dependency injection
context = ExecutionContext()

# Create event emitter for progress tracking
emitter = EventEmitter()

# Subscribe to events
def on_progress(event):
    print(f"Progress: {event.get('message', 'Processing...')}")

emitter.subscribe("progress", on_progress)
```

## Step 4: Execute Your Agent

```python
import asyncio

async def main():
    # Create agent instance
    agent = SummaryAgent("summarizer", context, emitter)

    # Execute
    input_data = TextInput(text="Your long text here...", max_length=50)
    result = await agent.execute(input_data)

    print(f"Summary: {result.summary}")
    print(f"Words: {result.word_count}")

# Run
asyncio.run(main())
```

## Complete Example

```python
import asyncio
from pydantic import BaseModel
from agent_lib import AgentBlock, ExecutionContext, EventEmitter

# Data models
class TextInput(BaseModel):
    text: str

class SummaryOutput(BaseModel):
    summary: str

# Agent implementation
class SummaryAgent(AgentBlock[TextInput, SummaryOutput]):
    def get_input_model(self):
        return TextInput

    def get_output_model(self):
        return SummaryOutput

    async def process(self, input_data: TextInput) -> SummaryOutput:
        await self.emit_progress("summarizing", 0.5, "Creating summary...")
        summary = input_data.text[:100] + "..."
        return SummaryOutput(summary=summary)

# Execution
async def main():
    context = ExecutionContext()
    emitter = EventEmitter()

    agent = SummaryAgent("summarizer", context, emitter)
    result = await agent.execute(TextInput(text="Hello world!"))

    print(result.summary)

asyncio.run(main())
```

## Next Steps

- [Core Concepts](concepts.md) - Understand the architecture
- [ExecutionContext Guide](../guide/execution-context.md) - Learn dependency injection
- [Event System](../guide/event-emitter.md) - Track agent progress
