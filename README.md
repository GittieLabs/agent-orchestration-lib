# Agent Orchestration Library

A framework for building composable, observable, and maintainable agent workflows with Python.

## Features

- **ExecutionContext**: Dependency injection container for managing shared state and services
- **EventEmitter**: Event-driven architecture with pub/sub pattern for notifications
- **AgentBlock**: Base class for agent implementations with validation and error handling
- **RetryStrategy**: Configurable retry logic with exponential backoff and LLM fallbacks
- **Flow**: Multi-agent workflow orchestration with sequential, parallel, and conditional execution

## Installation

```bash
pip install agent-orchestration-lib
```

## Quick Start

```python
from agent_lib import ExecutionContext, AgentBlock, EventEmitter
from pydantic import BaseModel

# Define your input/output models
class GreetingInput(BaseModel):
    name: str

class GreetingOutput(BaseModel):
    message: str

# Create an agent
class GreetingAgent(AgentBlock[GreetingInput, GreetingOutput]):
    def get_input_model(self):
        return GreetingInput

    def get_output_model(self):
        return GreetingOutput

    async def execute_impl(self, input_data: GreetingInput) -> GreetingOutput:
        return GreetingOutput(message=f"Hello, {input_data.name}!")

# Execute the agent
async def main():
    context = ExecutionContext()
    events = EventEmitter()

    agent = GreetingAgent("greeting", context, events)
    result = await agent.execute(GreetingInput(name="World"))
    print(result.message)  # Output: Hello, World!
```

## Documentation

Full documentation available at: https://agent-orchestration-lib.readthedocs.io

## Development

```bash
# Clone the repository
git clone https://github.com/yourusername/agent-orchestration-lib.git
cd agent-orchestration-lib

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run type checking
mypy src/agent_lib

# Format code
black src tests
ruff check src tests
```

## Architecture

This library was designed based on patterns extracted from production agent systems. See the [architecture documentation](docs/architecture.md) for detailed design decisions.

Key design principles:
- **Type Safety**: Full Pydantic validation for inputs and outputs
- **Observability**: Event emission for all significant operations
- **Composability**: Agents can be combined into complex workflows
- **Testability**: Each component is independently testable
- **Flexibility**: Extensible through dependency injection

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Project Status

**Alpha** - API may change. Not recommended for production use yet.

Current version: 0.1.0
