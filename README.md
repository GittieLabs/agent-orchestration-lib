# Agent Orchestration Library

[![PyPI version](https://badge.fury.io/py/agent-orchestration-lib.svg)](https://badge.fury.io/py/agent-orchestration-lib)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready Python framework for building **composable**, **observable**, and **maintainable** AI agent workflows. Built with type safety, dependency injection, and event-driven architecture at its core.

## Why Agent Orchestration Library?

Modern AI applications require orchestrating multiple LLM agents with:
- **Complex Dependencies**: Managing shared services (databases, APIs, LLM clients)
- **Real-Time Observability**: Tracking progress, token usage, and errors across agent chains
- **Reliability**: Retry logic, fallback strategies, and error handling
- **Type Safety**: Preventing runtime errors through Pydantic validation
- **Testability**: Isolated, mockable components for unit testing

This library provides battle-tested patterns extracted from production agent systems, eliminating the need to rebuild orchestration infrastructure for every project.

## Core Features

### 1. ExecutionContext - Dependency Injection
Manage shared state and services across agent execution with type-safe dependency injection.

```python
from agent_lib import ExecutionContext

# Register services
context = ExecutionContext()
context.register_singleton("database", db_connection)
context.register_singleton("llm_client", openai_client)

# Access in agents
db = context.get("database")
```

### 2. AgentBlock - Validated Agent Execution
Base class for agents with automatic input/output validation, error handling, and lifecycle hooks.

```python
from agent_lib import AgentBlock
from pydantic import BaseModel

class MyInput(BaseModel):
    text: str

class MyOutput(BaseModel):
    result: str

class MyAgent(AgentBlock[MyInput, MyOutput]):
    def get_input_model(self):
        return MyInput

    def get_output_model(self):
        return MyOutput

    async def process(self, input_data: MyInput) -> MyOutput:
        # Your agent logic here
        return MyOutput(result=f"Processed: {input_data.text}")
```

### 3. EventEmitter - Event-Driven Notifications
Pub/sub pattern for progress tracking, error handling, and real-time updates.

```python
from agent_lib import EventEmitter

emitter = EventEmitter()

# Subscribe to events
def on_progress(event):
    print(f"Progress: {event['progress']}% - {event['message']}")

emitter.subscribe("progress", on_progress)

# Emit from agents
await self.emit_progress("parsing", 0.5, "Parsing document...")
```

### 4. Flow - Multi-Agent Orchestration
Define complex workflows with sequential, parallel, and conditional execution.

```python
from agent_lib import Flow

flow = Flow("document_processing", context, emitter)

# Sequential execution
flow.add_agent(pdf_extraction_agent)
flow.add_agent(text_parsing_agent)

# Execute the flow
result = await flow.execute_sequential(initial_input)
```

### 5. RetryStrategy - Resilient Execution
Configurable retry logic with exponential backoff and LLM fallback chains.

```python
from agent_lib.retry import ExponentialBackoffRetry

retry_strategy = ExponentialBackoffRetry(
    max_attempts=3,
    base_delay=1.0,
    max_delay=60.0
)

result = await retry_strategy.execute_with_retry(
    agent.execute,
    input_data
)
```

## Installation

```bash
pip install agent-orchestration-lib
```

### Requirements
- Python 3.10 or higher
- Pydantic 2.0+

## Quick Start

Here's a complete example building a document analysis workflow:

```python
import asyncio
from agent_lib import ExecutionContext, EventEmitter, AgentBlock, Flow
from pydantic import BaseModel

# 1. Define data models
class DocumentInput(BaseModel):
    file_path: str

class DocumentText(BaseModel):
    text: str
    page_count: int

class AnalysisOutput(BaseModel):
    summary: str
    key_points: list[str]

# 2. Create agents
class PDFExtractionAgent(AgentBlock[DocumentInput, DocumentText]):
    def get_input_model(self):
        return DocumentInput

    def get_output_model(self):
        return DocumentText

    async def process(self, input_data: DocumentInput) -> DocumentText:
        # Extract text from PDF
        await self.emit_progress("extraction", 0.5, "Extracting text...")
        return DocumentText(text="Sample text", page_count=5)

class AnalysisAgent(AgentBlock[DocumentText, AnalysisOutput]):
    def get_input_model(self):
        return DocumentText

    def get_output_model(self):
        return AnalysisOutput

    async def process(self, input_data: DocumentText) -> AnalysisOutput:
        # Analyze the text
        llm_client = self.context.get("llm_client")
        await self.emit_progress("analysis", 0.75, "Analyzing document...")
        return AnalysisOutput(
            summary="Document summary",
            key_points=["Point 1", "Point 2"]
        )

# 3. Build and execute workflow
async def main():
    # Setup
    context = ExecutionContext()
    context.register_singleton("llm_client", your_llm_client)

    emitter = EventEmitter()
    emitter.subscribe("progress", lambda e: print(f"Progress: {e.message}"))

    # Create flow
    flow = Flow("document_analysis", context, emitter)
    flow.add_agent(PDFExtractionAgent("pdf_extractor", context, emitter))
    flow.add_agent(AnalysisAgent("analyzer", context, emitter))

    # Execute
    result = await flow.execute_sequential(
        DocumentInput(file_path="/path/to/document.pdf")
    )
    print(f"Summary: {result.summary}")

asyncio.run(main())
```

## Architecture & Design Patterns

This library implements production-proven patterns from real-world AI agent systems:

### Execution Sandwich Pattern
Every agent execution is wrapped with validation, tracking, and cleanup:
1. Validate input with Pydantic
2. Emit start event
3. Execute agent logic
4. Validate output
5. Emit completion/error events
6. Clean up resources

### Template Method Pattern
`AgentBlock` defines the execution skeleton; subclasses provide specifics:
- `get_input_model()` - Define expected input structure
- `get_output_model()` - Define output structure
- `process()` - Implement core agent logic

### Dependency Injection
`ExecutionContext` provides explicit dependency management:
- No global state
- Easy mocking for tests
- Service lifecycle control
- Parallel-safe child contexts

### Event-Driven Architecture
`EventEmitter` decouples notification logic:
- Multiple subscribers per event
- Pluggable adapters (Database, WebSocket, Logging)
- Type-safe event models

### Strategy Pattern
`RetryStrategy` provides pluggable retry logic:
- Exponential backoff
- LLM fallback chains
- Custom retry conditions

## Use Cases

- **Document Processing Pipelines**: Extract → Parse → Analyze workflows
- **Multi-Agent Research**: Parallel information gathering with aggregation
- **Customer Support Automation**: Triage → Route → Respond chains
- **Data Enrichment**: Sequential API calls with validation
- **Content Generation**: Planning → Drafting → Review → Publishing

## Documentation

- [Architecture Overview](docs/architecture.md) - Design decisions and patterns
- [API Reference](https://agent-orchestration-lib.readthedocs.io/en/latest/api/) - Complete API documentation
- [Quickstart Guide](docs/quickstart.md) - Step-by-step tutorial
- [Examples](examples/) - Production-ready examples
- [Migration Guide](docs/migration.md) - Migrating from other frameworks

## Development

```bash
# Clone repository
git clone https://github.com/GittieLabs/agent-orchestration-lib.git
cd agent-orchestration-lib

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/agent_lib

# Code formatting
black src tests
ruff check src tests --fix
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agent_lib --cov-report=html

# Run specific test file
pytest tests/unit/test_agent_block.py
```

## Design Principles

1. **Type Safety First**: Pydantic models for all inputs/outputs
2. **Explicit Dependencies**: No hidden globals or magical imports
3. **Event-Driven**: Observable through event emission
4. **Composability**: Agents as building blocks for complex workflows
5. **Testability**: Each component independently testable
6. **Production-Ready**: Patterns extracted from real systems

## Project Status

**Version**: 0.1.0 (Alpha)

This library is in active development. The API is stabilizing but may change in minor releases. Feedback and contributions are welcome!

### Roadmap

- [x] Core components (ExecutionContext, AgentBlock, EventEmitter, Flow)
- [x] Retry strategies (Exponential backoff)
- [x] Event system with adapters
- [ ] LLM fallback retry strategy
- [ ] Webhook event adapter
- [ ] Metrics collection adapter
- [ ] Advanced flow patterns (loops, fan-out/fan-in)
- [ ] Distributed tracing integration
- [ ] 1.0 stable release

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code of conduct
- Development setup
- Pull request process
- Coding standards

## License

MIT License - see [LICENSE](LICENSE) for details.

Copyright (c) 2025 GittieLabs, LLC

## Support

- **Documentation**: [https://agent-orchestration-lib.readthedocs.io](https://agent-orchestration-lib.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/GittieLabs/agent-orchestration-lib/issues)
- **Discussions**: [GitHub Discussions](https://github.com/GittieLabs/agent-orchestration-lib/discussions)

## Acknowledgments

This library was built from patterns extracted during the development of production AI agent systems. Special thanks to the teams who battle-tested these approaches in real-world applications.
