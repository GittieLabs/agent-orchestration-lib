"""Tool/function calling example with Anthropic Claude.

This example demonstrates how to use the unified tool calling interface
with Anthropic Claude models. The same tool definitions work across OpenAI,
Anthropic, and Gemini agents.

Requirements:
    pip install agent-orchestration-lib[anthropic]
    export ANTHROPIC_API_KEY=your_key_here
"""

import asyncio
import os
from typing import Any, Dict

from agent_lib import ExecutionContext, EventEmitter
from agent_lib.integrations.anthropic import AnthropicAgent, AnthropicMessage, AnthropicPrompt
from agent_lib.integrations.common import ToolDefinition, ToolParameter, ToolResult


# Define tools using the unified interface
def create_weather_tool() -> ToolDefinition:
    """Create a weather tool definition."""
    return ToolDefinition(
        name="get_weather",
        description="Get the current weather for a specific location",
        parameters={
            "location": ToolParameter(
                type="string",
                description="City name or location (e.g., 'San Francisco, CA')",
            ),
            "units": ToolParameter(
                type="string",
                description="Temperature units",
                enum=["celsius", "fahrenheit"],
                default="fahrenheit",
            ),
        },
        required=["location"],
    )


def create_calculator_tool() -> ToolDefinition:
    """Create a calculator tool definition."""
    return ToolDefinition(
        name="calculate",
        description="Perform a mathematical calculation",
        parameters={
            "expression": ToolParameter(
                type="string",
                description="Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')",
            ),
        },
        required=["expression"],
    )


# Tool execution functions (simulate actual tool execution)
def execute_get_weather(location: str, units: str = "fahrenheit") -> Dict[str, Any]:
    """Simulate weather API call."""
    # In real implementation, call actual weather API
    return {
        "location": location,
        "temperature": 72 if units == "fahrenheit" else 22,
        "units": units,
        "condition": "sunny",
        "humidity": 45,
    }


def execute_calculate(expression: str) -> Dict[str, Any]:
    """Safely evaluate mathematical expression."""
    try:
        # Note: eval() is used here for demo only - use safe math parser in production
        result = eval(expression, {"__builtins__": {}}, {})
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


async def main() -> None:
    """Run tool calling examples."""
    # Setup
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return

    context = ExecutionContext()
    emitter = EventEmitter()
    agent = AnthropicAgent(
        name="claude_tool_agent", api_key=api_key, context=context, emitter=emitter
    )

    # Define available tools
    tools = [create_weather_tool(), create_calculator_tool()]

    print("=" * 70)
    print("Example 1: Single Tool Call")
    print("=" * 70)

    # Initial user message
    prompt1 = AnthropicPrompt(
        messages=[
            AnthropicMessage(
                role="user",
                content="What's the weather like in San Francisco?",
            )
        ],
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
    )

    response1 = await agent.execute(prompt1)
    print(f"\nAssistant response: {response1.content}")
    print(f"Stop reason: {response1.stop_reason}")

    if response1.tool_calls:
        print(f"\nTool calls made: {len(response1.tool_calls)}")
        for tool_call in response1.tool_calls:
            print(f"  - {tool_call.name}({tool_call.arguments})")

            # Execute the tool
            if tool_call.name == "get_weather":
                result = execute_get_weather(**tool_call.arguments)
                print(f"  Result: {result}")

    print("\n" + "=" * 70)
    print("Example 2: Multi-Turn Conversation with Tool Results")
    print("=" * 70)

    # Start conversation
    messages = [
        AnthropicMessage(
            role="user",
            content="What's the weather in New York and what's 15 * 23?",
        )
    ]

    prompt2 = AnthropicPrompt(
        messages=messages, model="claude-3-5-sonnet-20241022", max_tokens=1024, tools=tools
    )
    response2 = await agent.execute(prompt2)

    print(f"\n[User]: What's the weather in New York and what's 15 * 23?")
    print(f"\n[Assistant]: {response2.content or '(calling tools...)'}")

    # Handle tool calls
    if response2.tool_calls:
        print(f"\nTool calls: {len(response2.tool_calls)}")

        # Add assistant message with tool calls
        messages.append(
            AnthropicMessage(
                role="assistant",
                content=response2.content,
                tool_calls=response2.tool_calls,
            )
        )

        # Execute each tool and add results
        for tool_call in response2.tool_calls:
            print(f"  Executing: {tool_call.name}({tool_call.arguments})")

            if tool_call.name == "get_weather":
                result = execute_get_weather(**tool_call.arguments)
            elif tool_call.name == "calculate":
                result = execute_calculate(**tool_call.arguments)
            else:
                result = {"error": "Unknown tool"}

            print(f"  Result: {result}")

            # Add tool result message
            messages.append(
                AnthropicMessage(
                    role="user",
                    content=[
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call.id,
                            "content": str(result),
                        }
                    ],
                )
            )

        # Continue conversation with tool results
        prompt3 = AnthropicPrompt(
            messages=messages, model="claude-3-5-sonnet-20241022", max_tokens=1024, tools=tools
        )
        response3 = await agent.execute(prompt3)

        print(f"\n[Assistant]: {response3.content}")

    print("\n" + "=" * 70)
    print("Example 3: Force Tool Usage")
    print("=" * 70)

    # Force the model to use a specific tool
    prompt4 = AnthropicPrompt(
        messages=[
            AnthropicMessage(
                role="user",
                content="Calculate 42 divided by 7",
            )
        ],
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[create_calculator_tool()],
        tool_choice={"type": "tool", "name": "calculate"},  # Force specific tool
    )

    response4 = await agent.execute(prompt4)
    print(f"\nUser: Calculate 42 divided by 7")
    print(f"Assistant: {response4.content or '(calling calculator...)'}")

    if response4.tool_calls:
        for tool_call in response4.tool_calls:
            result = execute_calculate(**tool_call.arguments)
            print(f"Tool call: {tool_call.name}({tool_call.arguments})")
            print(f"Result: {result}")

    print("\n" + "=" * 70)
    print("Summary: Unified Tool Calling with Anthropic")
    print("=" * 70)
    print("\nKey Features:")
    print("✓ Provider-agnostic tool definitions")
    print("✓ Same ToolDefinition works across OpenAI, Anthropic, Gemini")
    print("✓ Type-safe with Pydantic validation")
    print("✓ Consistent ToolCall and ToolResult formats")
    print("✓ Multi-turn conversations with tool results")
    print("✓ Control over tool usage (auto, any, tool)")
    print("\nNext Steps:")
    print("- Use same tool definitions with OpenAIAgent or GeminiAgent")
    print("- Implement error handling for tool execution")
    print("- Add tool result validation")
    print("- Build multi-agent workflows with tool sharing")


if __name__ == "__main__":
    asyncio.run(main())
