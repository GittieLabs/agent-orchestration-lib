# Agent Orchestration Library

Welcome to the **Agent Orchestration Library** - a production-ready Python framework for building composable, observable, and maintainable AI agent workflows.

## Overview

The Agent Orchestration Library provides the foundational infrastructure needed to build reliable multi-agent systems. It eliminates the need to rebuild orchestration patterns for every project by providing battle-tested components extracted from production AI systems.

## Key Features

### :material-puzzle: Composability
Build complex workflows from simple, reusable agent blocks. Chain agents sequentially, run them in parallel, or create conditional branching logic.

### :material-eye: Observability
Track every aspect of agent execution with built-in event emission. Monitor progress, token usage, errors, and custom metrics in real-time.

### :material-shield-check: Type Safety
Leverage Pydantic for automatic input/output validation. Catch errors at development time, not in production.

### :material-link: Dependency Injection
Manage shared services and state explicitly through `ExecutionContext`. No global state, easy testing, parallel-safe execution.

### :material-refresh: Retry Strategies
Handle transient failures gracefully with configurable retry logic including exponential backoff and LLM fallback chains.

## Quick Example

```python
from agent_lib import ExecutionContext, EventEmitter, AgentBlock
from pydantic import BaseModel

# Define your data models
class AnalysisInput(BaseModel):
    text: str

class AnalysisOutput(BaseModel):
    summary: str

# Create your agent
class AnalysisAgent(AgentBlock[AnalysisInput, AnalysisOutput]):
    def get_input_model(self):
        return AnalysisInput

    def get_output_model(self):
        return AnalysisOutput

    async def process(self, input_data: AnalysisInput) -> AnalysisOutput:
        # Your agent logic here
        llm = self.context.get("llm_client")
        result = await llm.analyze(input_data.text)
        return AnalysisOutput(summary=result)

# Execute your agent
async def main():
    context = ExecutionContext()
    context.register_singleton("llm_client", your_llm)

    emitter = EventEmitter()
    emitter.subscribe("progress", lambda e: print(e.message))

    agent = AnalysisAgent("analyzer", context, emitter)
    result = await agent.execute(AnalysisInput(text="..."))
```

## Why This Library?

Modern AI applications face common challenges:

- **Dependency Management**: How do I share database connections, API clients, and configuration across agents?
- **Error Handling**: How do I implement retry logic with fallbacks to different LLMs?
- **Observability**: How do I track what's happening across a chain of agent calls?
- **Testing**: How do I test agents in isolation without hitting real APIs?
- **Type Safety**: How do I prevent runtime errors from invalid data?

This library provides proven solutions to these problems, allowing you to focus on your agent logic instead of infrastructure.

## Core Components

| Component | Purpose |
|-----------|---------|
| **ExecutionContext** | Dependency injection container for managing services and state |
| **AgentBlock** | Base class for agents with validation, events, and lifecycle hooks |
| **EventEmitter** | Pub/sub system for progress tracking and notifications |
| **Flow** | Workflow orchestration for sequential, parallel, and conditional execution |
| **RetryStrategy** | Configurable retry logic with exponential backoff |

## Getting Started

1. [Installation](getting-started/installation.md) - Install the library
2. [Quick Start](getting-started/quickstart.md) - Build your first agent
3. [Core Concepts](getting-started/concepts.md) - Understand the architecture

## Use Cases

- **Document Processing**: Extract → Parse → Analyze pipelines
- **Multi-Agent Research**: Parallel information gathering with aggregation
- **Customer Support**: Triage → Route → Respond workflows
- **Data Enrichment**: Sequential API calls with validation
- **Content Generation**: Planning → Drafting → Review chains

## Design Principles

1. **Type Safety First**: Pydantic models for all inputs/outputs
2. **Explicit Dependencies**: No hidden globals or magic
3. **Event-Driven**: Observable through comprehensive event emission
4. **Composability**: Agents as building blocks for complex workflows
5. **Testability**: Each component independently testable
6. **Production-Ready**: Patterns extracted from real-world systems

## Community

- **GitHub**: [GittieLabs/agent-orchestration-lib](https://github.com/GittieLabs/agent-orchestration-lib)
- **PyPI**: [agent-orchestration-lib](https://pypi.org/project/agent-orchestration-lib/)
- **Issues**: [Report bugs or request features](https://github.com/GittieLabs/agent-orchestration-lib/issues)

## License

MIT License - Copyright (c) 2025 GittieLabs, LLC

---

Ready to build reliable agent workflows? [Get started now](getting-started/installation.md)!
