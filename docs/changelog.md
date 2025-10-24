# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-01-24

### Added
- `ConditionalStep` for if/else branching logic in agent workflows
- `FlowAdapter` to execute flows as agent blocks (nested flows)
- `LLMFallbackRetry` strategy for automatic LLM model fallback on failures
- Exported `Flow` class for multi-agent orchestration
- Exported all retry strategies: `ExponentialBackoffRetry`, `FixedDelayRetry`, `LinearBackoffRetry`, `NoRetry`
- Helper functions: `retry_on_exception_type()`, `retry_on_error_message()`

### Features
- Conditional execution based on runtime data evaluation
- Sub-flow composition for hierarchical workflows
- LLM resilience with automatic model switching
- Complete retry strategy suite with configurable backoff policies

## [0.1.0] - 2025-01-24

### Added
- Initial release of Agent Orchestration Library
- `ExecutionContext` for dependency injection
- `AgentBlock` base class with validation and events
- `EventEmitter` pub/sub system
- `Flow` for multi-agent orchestration
- `ExponentialBackoffRetry` strategy
- Comprehensive documentation
- Full test suite
- Type hints throughout
- MIT License

### Core Features
- Type-safe agent execution with Pydantic
- Event-driven architecture
- Dependency injection container
- Sequential and parallel execution
- Retry strategies
- Progress tracking
- Error handling

[0.1.0]: https://github.com/GittieLabs/agent-orchestration-lib/releases/tag/v0.1.0
