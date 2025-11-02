"""Tool/function calling example with Google Gemini.

This example demonstrates how to use the unified tool calling interface
with Google Gemini models. The same tool definitions work across OpenAI,
Anthropic, and Gemini agents.

Requirements:
    pip install agent-orchestration-lib[gemini]
    export GOOGLE_API_KEY=your_key_here
"""

import asyncio
import os
from typing import Any, Dict

from agent_lib import ExecutionContext, EventEmitter
from agent_lib.integrations.gemini import GeminiAgent, GeminiPrompt
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
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set")
        return

    context = ExecutionContext()
    emitter = EventEmitter()
    agent = GeminiAgent(
        name="gemini_tool_agent", api_key=api_key, context=context, emitter=emitter
    )

    # Define available tools
    tools = [create_weather_tool(), create_calculator_tool()]

    print("=" * 70)
    print("Example 1: Single Tool Call")
    print("=" * 70)

    # Initial user message
    prompt1 = GeminiPrompt(
        prompt="What's the weather like in San Francisco?",
        model="gemini-1.5-flash",
        tools=tools,
    )

    response1 = await agent.execute(prompt1)
    print(f"\nAssistant response: {response1.content}")
    print(f"Finish reason: {response1.finish_reason}")

    if response1.tool_calls:
        print(f"\nTool calls made: {len(response1.tool_calls)}")
        for tool_call in response1.tool_calls:
            print(f"  - {tool_call.name}({tool_call.arguments})")

            # Execute the tool
            if tool_call.name == "get_weather":
                result = execute_get_weather(**tool_call.arguments)
                print(f"  Result: {result}")

    print("\n" + "=" * 70)
    print("Example 2: Calculate with Tool")
    print("=" * 70)

    prompt2 = GeminiPrompt(
        prompt="What's 15 multiplied by 23?",
        model="gemini-1.5-flash",
        tools=[create_calculator_tool()],
    )

    response2 = await agent.execute(prompt2)

    print(f"\n[User]: What's 15 multiplied by 23?")
    print(f"\n[Assistant]: {response2.content or '(calling calculator...)'}")

    # Handle tool calls
    if response2.tool_calls:
        print(f"\nTool calls: {len(response2.tool_calls)}")
        for tool_call in response2.tool_calls:
            print(f"  Executing: {tool_call.name}({tool_call.arguments})")

            if tool_call.name == "calculate":
                result = execute_calculate(**tool_call.arguments)
                print(f"  Result: {result}")

    print("\n" + "=" * 70)
    print("Example 3: Weather Query")
    print("=" * 70)

    # Weather query with specific units
    prompt3 = GeminiPrompt(
        prompt="What's the weather in Tokyo? Use celsius.",
        model="gemini-1.5-flash",
        tools=[create_weather_tool()],
    )

    response3 = await agent.execute(prompt3)
    print(f"\nUser: What's the weather in Tokyo? Use celsius.")
    print(f"Assistant: {response3.content or '(checking weather...)'}")

    if response3.tool_calls:
        for tool_call in response3.tool_calls:
            result = execute_get_weather(**tool_call.arguments)
            print(f"Tool call: {tool_call.name}({tool_call.arguments})")
            print(f"Result: {result}")

    print("\n" + "=" * 70)
    print("Summary: Unified Tool Calling with Gemini")
    print("=" * 70)
    print("\nKey Features:")
    print("✓ Provider-agnostic tool definitions")
    print("✓ Same ToolDefinition works across OpenAI, Anthropic, Gemini")
    print("✓ Type-safe with Pydantic validation")
    print("✓ Consistent ToolCall and ToolResult formats")
    print("✓ Automatic function declaration conversion")
    print("✓ Works with Gemini 1.5 Flash and Pro models")
    print("\nNext Steps:")
    print("- Use same tool definitions with OpenAIAgent or AnthropicAgent")
    print("- Implement error handling for tool execution")
    print("- Add tool result validation")
    print("- Build multi-agent workflows with tool sharing")
    print("\nNote:")
    print("- Gemini's tool calling uses function declarations")
    print("- All conversions handled automatically by the unified interface")
    print("- Function responses can be passed back for continued conversation")


if __name__ == "__main__":
    asyncio.run(main())
