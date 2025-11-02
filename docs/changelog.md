# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2025-11-02

### Added
- **Unified Tool/Function Calling**: Provider-agnostic tool calling interface
  - `ToolDefinition`: Universal tool definition compatible with OpenAI, Anthropic, and Gemini
  - `ToolParameter`: JSON Schema-based parameter definitions
  - `ToolCall`: Unified format for tool calls from LLMs
  - `ToolResult`: Unified format for tool execution results
  - Same tool definitions work across all LLM providers
  - Single source of truth for tools - define once, use everywhere

- **OpenAI Tool Support** (COMPLETE):
  - Updated `OpenAIPrompt` to accept `tools` and `tool_choice` parameters
  - Updated `OpenAIMessage` to support `tool` role and tool-related fields
  - Updated `OpenAIResponse` to include `tool_calls`
  - Automatic conversion between unified and OpenAI native formats
  - Multi-turn conversations with tool results
  - Agent implementation complete with tool call extraction
  - Example: `examples/tool-calling-openai.py`

- **Anthropic Tool Support** (COMPLETE):
  - Updated `AnthropicPrompt` to accept `tools` and `tool_choice` parameters
  - Updated `AnthropicMessage` to support tool result content blocks
  - Updated `AnthropicResponse` to include `tool_calls`
  - Automatic conversion between unified and Anthropic tool_use format
  - Agent implementation complete with tool_use block parsing
  - Multi-turn conversations with tool results
  - Example: `examples/tool-calling-anthropic.py`

- **Gemini Tool Support** (COMPLETE):
  - Updated `GeminiPrompt` to accept `tools` parameter
  - Updated `GeminiResponse` to include `tool_calls`
  - Automatic conversion between unified and Gemini function declaration format
  - Agent implementation complete with function_call parsing
  - Support for Gemini 1.5 Flash and Pro models
  - Example: `examples/tool-calling-gemini.py`

### Changed
- Made `content` field optional in all LLM response models (can be None when only tool calls)
- All three LLM agents now support tool calling with consistent interface

### Documentation
- Added comprehensive tool calling examples for all three providers
- Tool usage patterns and best practices
- Multi-turn conversation examples with tool results
- Examples show same tool definitions working across all providers

### Testing
- 17 unit tests for tool calling functionality (all passing)
- Model instantiation tests
- Format conversion tests
- Validation tests

## [0.3.3] - 2025-10-31

### Fixed
- **LLM Integrations**: Fixed `on_complete()` method signature in all LLM agents
  - **Issue**: Base class calls `on_complete(output_data)` with 1 argument, but LLM agents defined it with 2 parameters `(input_data, output_data)`
  - **Error**: `TypeError: on_complete() missing 1 required positional argument: 'output_data'`
  - **Fix**: Removed unused `input_data` parameter from all three LLM agents
  - **Affected agents**: OpenAI, Anthropic, Gemini

## [0.3.2] - 2025-10-31

### Fixed
- **LLM Integrations**: Fixed `emit_progress()` method calls in all LLM agents (9 total fixes)
  - **OpenAI Agent**: Fixed 3 emit_progress() calls (calling_openai_api, openai_api_complete, openai_api_error)
  - **Anthropic Agent**: Fixed 3 emit_progress() calls (calling_anthropic_api, anthropic_api_complete, anthropic_api_error)
  - **Gemini Agent**: Fixed 3 emit_progress() calls (calling_gemini_api, gemini_api_complete, gemini_api_error)
  - Added missing `await` keyword for all emit_progress() calls
  - Fixed parameters to use named arguments: `stage`, `progress`, `message`, `details`
  - Added descriptive messages for better observability
  - Set appropriate progress values (0.0 for start, 1.0 for completion, 0.0 for errors)
  - Bug: emit_progress() was being called with only 2 arguments instead of required 4 arguments

## [0.3.1] - 2025-10-31

### Fixed
- **SwitchStep**: Fixed `emit_progress()` method calls to use correct signature
  - Added missing `await` keyword for all emit_progress() calls
  - Fixed parameters to use named arguments: `stage`, `progress`, `message`, `details`
  - Added descriptive messages for better observability
  - Set appropriate progress values (0.0 for case selection, 0.5 for execution)

## [0.3.0] - 2025-10-31

### Added
- **SwitchStep**: Multi-branch conditional routing for clean handling of 3+ cases
  - Switch/case pattern similar to programming languages
  - Selector function returns case key for routing
  - Optional default/fallback agent
  - Cleaner alternative to nested ConditionalStep blocks
  - Full event emission and observability
  - Example: `examples/switch-step-routing.py`

- **OpenAI Integration**: Complete integration with GPT-4, GPT-4-turbo, and GPT-3.5-turbo models
  - `OpenAIAgent` for executing OpenAI API calls
  - Automatic token counting using tiktoken
  - Cost estimation per request
  - Support for system messages, conversation history, and JSON response format
  - Helper functions: `create_simple_prompt()`, `create_system_prompt()`

- **Anthropic Integration**: Complete integration with Claude 3 models
  - `AnthropicAgent` for Claude Opus, Sonnet, and Haiku models
  - 200K token context window support
  - System prompt support (Claude's preferred format)
  - Cost estimation per request (per 1M tokens pricing)
  - Helper functions: `create_simple_prompt()`, `create_system_prompt()`

- **Google Gemini Integration**: Complete integration with Gemini models
  - `GeminiAgent` for Gemini Pro, Gemini 1.5 Pro, and Gemini 1.5 Flash
  - Up to 1M token context window (Gemini 1.5)
  - Safety ratings extraction
  - Cost estimation per request
  - Configurable sampling parameters (temperature, top-k, top-p)
  - Helper function: `create_simple_prompt()`

- **7 Comprehensive Examples**:
  - `basic-openai-agent.py` - OpenAI integration basics
  - `anthropic-agent.py` - Anthropic Claude usage patterns
  - `gemini-agent.py` - Google Gemini demonstrations
  - `multi-llm-fallback.py` - Provider fallback strategies
  - `llm-comparison.py` - Side-by-side provider comparison
  - `llm-cost-tracking.py` - Cost tracking and budget management
  - `conversation-agent.py` - Multi-turn conversational agents

- **Optional Dependencies**: Install integrations separately
  - `pip install agent-orchestration-lib[openai]`
  - `pip install agent-orchestration-lib[anthropic]`
  - `pip install agent-orchestration-lib[gemini]`
  - `pip install agent-orchestration-lib[all-llm]`

### Features
- Token counting and cost estimation for all LLM providers
- Async/await support for all LLM API calls
- Integration with existing retry strategies (including `LLMFallbackRetry`)
- Event emission for observability
- Type-safe Pydantic models for all inputs and outputs
- Helper utilities for common prompt patterns

### Documentation
- Updated README with LLM integration examples
- Added installation instructions for optional dependencies
- Updated package docstring with LLM integration information

### Changed
- Version bumped to 0.3.0
- Updated package description to highlight LLM integrations

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
