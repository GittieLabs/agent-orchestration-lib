"""Anthropic (Claude) agent example.

This example demonstrates how to use the Anthropic integration to call Claude models
with the agent orchestration framework.

Requirements:
    pip install agent-orchestration-lib[anthropic]

Environment:
    ANTHROPIC_API_KEY: Your Anthropic API key
"""

import asyncio
import os

from agent_lib import EventEmitter, ExecutionContext
from agent_lib.integrations.anthropic import (
    AnthropicAgent,
    create_simple_prompt,
    create_system_prompt,
)


async def main() -> None:
    """Run the Anthropic agent example."""
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='sk-ant-...'")
        return

    # Create context and emitter
    context = ExecutionContext()
    emitter = EventEmitter()

    # Create Anthropic agent
    agent = AnthropicAgent(
        name="claude-agent",
        api_key=api_key,
        context=context,
        emitter=emitter,
        default_model="claude-3-sonnet-20240229",
        default_max_tokens=1024,
    )

    print("=" * 60)
    print("Example 1: Simple Question with Claude Sonnet")
    print("=" * 60)

    # Example 1: Simple prompt with Sonnet (balanced performance/cost)
    prompt1 = create_simple_prompt(
        "What is Python? Answer in 2-3 sentences.",
        model="claude-3-sonnet-20240229",
        max_tokens=1024,
    )

    result1 = await agent.execute(prompt1)

    print(f"\nPrompt: What is Python?")
    print(f"Model: {result1.model}")
    print(f"\nResponse:\n{result1.content}")
    print(f"\nTokens: {result1.total_tokens} (input: {result1.input_tokens}, output: {result1.output_tokens})")
    print(f"Cost: ${result1.cost_usd:.6f}")
    print(f"Stop reason: {result1.stop_reason}")

    print("\n" + "=" * 60)
    print("Example 2: With System Prompt (Writing Assistant)")
    print("=" * 60)

    # Example 2: System prompt (Claude's preferred way to set context)
    prompt2 = create_system_prompt(
        system="You are a professional technical writer. Write clear, concise documentation.",
        user="Write a brief docstring for a Python function that validates email addresses.",
        model="claude-3-sonnet-20240229",
        max_tokens=2048,
        temperature=0.7,
    )

    result2 = await agent.execute(prompt2)

    print(f"\nSystem: Professional technical writer")
    print(f"User: Write a docstring for email validation function")
    print(f"\nResponse:\n{result2.content}")
    print(f"\nTokens: {result2.total_tokens}")
    print(f"Cost: ${result2.cost_usd:.6f}")

    print("\n" + "=" * 60)
    print("Example 3: Claude Opus (Highest Performance)")
    print("=" * 60)

    # Example 3: Claude Opus for complex reasoning
    prompt3 = create_simple_prompt(
        "Explain the concept of recursive algorithms with a simple example in Python. "
        "Include both the code and a step-by-step explanation.",
        model="claude-3-opus-20240229",
        max_tokens=2048,
        temperature=0.5,
    )

    result3 = await agent.execute(prompt3)

    print(f"\nModel: {result3.model} (most capable)")
    print(f"\nResponse:\n{result3.content}")
    print(f"\nTokens: {result3.total_tokens}")
    print(f"Cost: ${result3.cost_usd:.6f} (premium pricing for best quality)")

    print("\n" + "=" * 60)
    print("Example 4: Claude Haiku (Fast and Cheap)")
    print("=" * 60)

    # Example 4: Claude Haiku for simple tasks
    prompt4 = create_simple_prompt(
        "List 5 popular Python web frameworks. Just the names, comma-separated.",
        model="claude-3-haiku-20240307",
        max_tokens=512,
        temperature=0.3,
    )

    result4 = await agent.execute(prompt4)

    print(f"\nModel: {result4.model} (fastest and cheapest)")
    print(f"\nResponse:\n{result4.content}")
    print(f"\nTokens: {result4.total_tokens}")
    print(f"Cost: ${result4.cost_usd:.6f} (very cost-effective!)")

    print("\n" + "=" * 60)
    print("Example 5: Long Context (200K tokens available)")
    print("=" * 60)

    # Example 5: Demonstrate Claude's long context capability
    from agent_lib.integrations.anthropic import AnthropicMessage, AnthropicPrompt

    # Simulate a long document (Claude can handle up to 200K tokens)
    long_document = "The Python programming language. " * 100  # Repeat for demo

    prompt5 = AnthropicPrompt(
        system="You are analyzing a document. Provide a brief summary.",
        messages=[
            AnthropicMessage(
                role="user",
                content=f"Here is a document:\n\n{long_document}\n\nSummarize the key topic in one sentence."
            )
        ],
        model="claude-3-sonnet-20240229",
        max_tokens=512,
    )

    result5 = await agent.execute(prompt5)

    print(f"\nDocument length: {len(long_document)} characters")
    print(f"Input tokens: {result5.input_tokens} (Claude handles long context well)")
    print(f"\nSummary:\n{result5.content}")
    print(f"\nCost: ${result5.cost_usd:.6f}")

    print("\n" + "=" * 60)
    print("Summary: Cost Comparison Across Models")
    print("=" * 60)

    total_cost = result1.cost_usd + result2.cost_usd + result3.cost_usd + result4.cost_usd + result5.cost_usd
    total_tokens = result1.total_tokens + result2.total_tokens + result3.total_tokens + result4.total_tokens + result5.total_tokens

    print(f"\nTotal tokens used: {total_tokens}")
    print(f"Total cost: ${total_cost:.6f}")
    print(f"\nPer-model breakdown:")
    print(f"  Haiku:  ${result4.cost_usd:.6f} - Fast, cheap, good for simple tasks")
    print(f"  Sonnet: ${result1.cost_usd:.6f} - Balanced performance/cost")
    print(f"  Opus:   ${result3.cost_usd:.6f} - Best quality, premium pricing")
    print(f"\nClaude advantages:")
    print(f"  - 200K token context window (handles long documents)")
    print(f"  - System prompts for clear instruction-setting")
    print(f"  - Three model tiers for different use cases")


if __name__ == "__main__":
    asyncio.run(main())
