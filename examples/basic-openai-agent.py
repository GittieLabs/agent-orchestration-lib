"""Basic OpenAI agent example.

This example demonstrates how to use the OpenAI integration to call GPT models
with the agent orchestration framework.

Requirements:
    pip install agent-orchestration-lib[openai]

Environment:
    OPENAI_API_KEY: Your OpenAI API key
"""

import asyncio
import os

from agent_lib import EventEmitter, ExecutionContext
from agent_lib.integrations.openai import OpenAIAgent, create_simple_prompt, create_system_prompt


async def main() -> None:
    """Run the basic OpenAI agent example."""
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='sk-...'")
        return

    # Create context and emitter
    context = ExecutionContext()
    emitter = EventEmitter()

    # Create OpenAI agent
    agent = OpenAIAgent(
        name="gpt4-agent",
        api_key=api_key,
        context=context,
        emitter=emitter,
        default_model="gpt-4",
    )

    print("=" * 60)
    print("Example 1: Simple Question")
    print("=" * 60)

    # Example 1: Simple prompt
    prompt1 = create_simple_prompt("What is Python? Answer in 2 sentences.", model="gpt-4")

    result1 = await agent.execute(prompt1)

    print(f"\nPrompt: What is Python?")
    print(f"\nResponse:\n{result1.content}")
    print(f"\nTokens: {result1.total_tokens} (prompt: {result1.prompt_tokens}, completion: {result1.completion_tokens})")
    print(f"Cost: ${result1.cost_usd:.6f}")
    print(f"Finish reason: {result1.finish_reason}")

    print("\n" + "=" * 60)
    print("Example 2: With System Message (Coding Assistant)")
    print("=" * 60)

    # Example 2: System message
    prompt2 = create_system_prompt(
        system="You are an expert Python developer. Provide concise, practical code examples.",
        user="How do I reverse a list in Python? Show me 2 ways.",
        model="gpt-4",
        temperature=0.7,
    )

    result2 = await agent.execute(prompt2)

    print(f"\nSystem: Expert Python developer")
    print(f"User: How do I reverse a list in Python?")
    print(f"\nResponse:\n{result2.content}")
    print(f"\nTokens: {result2.total_tokens}")
    print(f"Cost: ${result2.cost_usd:.6f}")

    print("\n" + "=" * 60)
    print("Example 3: Using GPT-3.5-turbo (cheaper, faster)")
    print("=" * 60)

    # Example 3: GPT-3.5-turbo for simpler tasks
    prompt3 = create_simple_prompt(
        "List 5 popular Python web frameworks in a comma-separated list.",
        model="gpt-3.5-turbo",
        temperature=0.3,
    )

    result3 = await agent.execute(prompt3)

    print(f"\nPrompt: List 5 popular Python web frameworks")
    print(f"Model: {result3.model}")
    print(f"\nResponse:\n{result3.content}")
    print(f"\nTokens: {result3.total_tokens}")
    print(f"Cost: ${result3.cost_usd:.6f} (much cheaper!)")

    print("\n" + "=" * 60)
    print("Example 4: JSON Response Format")
    print("=" * 60)

    # Example 4: Request JSON response
    from agent_lib.integrations.openai import OpenAIMessage, OpenAIPrompt

    prompt4 = OpenAIPrompt(
        messages=[
            OpenAIMessage(
                role="system",
                content="You are a helpful assistant that outputs JSON."
            ),
            OpenAIMessage(
                role="user",
                content='List 3 programming languages with their use cases. Format as JSON: {"languages": [{"name": "...", "use_case": "..."}]}'
            ),
        ],
        model="gpt-4",
        temperature=0.5,
        response_format={"type": "json_object"},
    )

    result4 = await agent.execute(prompt4)

    print(f"\nResponse (JSON format):\n{result4.content}")
    print(f"\nTokens: {result4.total_tokens}")
    print(f"Cost: ${result4.cost_usd:.6f}")

    print("\n" + "=" * 60)
    print("Summary: Total Cost")
    print("=" * 60)

    total_cost = result1.cost_usd + result2.cost_usd + result3.cost_usd + result4.cost_usd
    total_tokens = result1.total_tokens + result2.total_tokens + result3.total_tokens + result4.total_tokens

    print(f"\nTotal tokens used: {total_tokens}")
    print(f"Total cost: ${total_cost:.6f}")
    print(f"\nThis demonstrates the value of:")
    print(f"  - Using GPT-3.5-turbo for simple tasks (much cheaper)")
    print(f"  - Using GPT-4 for complex reasoning")
    print(f"  - Automatic token counting and cost tracking")


if __name__ == "__main__":
    asyncio.run(main())
