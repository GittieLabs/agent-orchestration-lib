"""Google Gemini agent example.

This example demonstrates how to use the Gemini integration to call Google's models
with the agent orchestration framework.

Requirements:
    pip install agent-orchestration-lib[gemini]

Environment:
    GOOGLE_API_KEY: Your Google API key for Gemini
"""

import asyncio
import os

from agent_lib import EventEmitter, ExecutionContext
from agent_lib.integrations.gemini import GeminiAgent, create_simple_prompt


async def main() -> None:
    """Run the Gemini agent example."""
    # Get API key from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set")
        print("Set it with: export GOOGLE_API_KEY='...'")
        print("Get your key from: https://makersuite.google.com/app/apikey")
        return

    # Create context and emitter
    context = ExecutionContext()
    emitter = EventEmitter()

    # Create Gemini agent
    agent = GeminiAgent(
        name="gemini-agent",
        api_key=api_key,
        context=context,
        emitter=emitter,
        default_model="gemini-pro",
    )

    print("=" * 60)
    print("Example 1: Simple Question with Gemini Pro")
    print("=" * 60)

    # Example 1: Simple prompt with Gemini Pro
    prompt1 = create_simple_prompt(
        "What is Python? Answer in 2-3 sentences.",
        model="gemini-pro",
        temperature=0.7,
    )

    result1 = await agent.execute(prompt1)

    print("\nPrompt: What is Python?")
    print(f"Model: {result1.model}")
    print(f"\nResponse:\n{result1.content}")
    print(f"\nTokens: {result1.total_tokens} (estimated)")
    print(f"Cost: ${result1.cost_usd:.6f}")
    print(f"Finish reason: {result1.finish_reason}")

    if result1.safety_ratings:
        print("\nSafety ratings:")
        for rating in result1.safety_ratings:
            print(f"  {rating['category']}: {rating['probability']}")

    print("\n" + "=" * 60)
    print("Example 2: Creative Writing Task")
    print("=" * 60)

    # Example 2: Creative task with higher temperature
    from agent_lib.integrations.gemini import GeminiPrompt

    prompt2 = GeminiPrompt(
        prompt="Write a haiku about programming in Python.",
        model="gemini-pro",
        temperature=0.9,  # Higher temperature for creativity
        max_output_tokens=100,
    )

    result2 = await agent.execute(prompt2)

    print("\nPrompt: Write a haiku about Python programming")
    print("Temperature: 0.9 (creative)")
    print(f"\nResponse:\n{result2.content}")
    print(f"\nTokens: {result2.total_tokens}")
    print(f"Cost: ${result2.cost_usd:.6f}")

    print("\n" + "=" * 60)
    print("Example 3: Technical Explanation (Low Temperature)")
    print("=" * 60)

    # Example 3: Technical explanation with low temperature
    prompt3 = create_simple_prompt(
        "Explain what a binary search tree is and why it's useful. Be concise and technical.",
        model="gemini-pro",
        temperature=0.2,  # Low temperature for consistency
        max_output_tokens=500,
    )

    result3 = await agent.execute(prompt3)

    print("\nTemperature: 0.2 (factual/deterministic)")
    print(f"\nResponse:\n{result3.content}")
    print(f"\nTokens: {result3.total_tokens}")
    print(f"Cost: ${result3.cost_usd:.6f}")

    print("\n" + "=" * 60)
    print("Example 4: Using Gemini 1.5 Flash (Fast and Cheap)")
    print("=" * 60)

    # Example 4: Gemini 1.5 Flash for simple tasks
    prompt4 = create_simple_prompt(
        "List 5 popular Python libraries for data science, comma-separated.",
        model="gemini-1.5-flash",
        temperature=0.3,
    )

    result4 = await agent.execute(prompt4)

    print(f"\nModel: {result4.model} (fastest and cheapest)")
    print(f"\nResponse:\n{result4.content}")
    print(f"\nTokens: {result4.total_tokens}")
    print(f"Cost: ${result4.cost_usd:.6f} (very cost-effective!)")

    print("\n" + "=" * 60)
    print("Example 5: Using Top-K Sampling")
    print("=" * 60)

    # Example 5: Top-K sampling for controlled randomness
    prompt5 = GeminiPrompt(
        prompt="Give me 3 creative names for a Python web scraping library.",
        model="gemini-pro",
        temperature=0.8,
        top_k=40,  # Only sample from top 40 tokens
        max_output_tokens=200,
    )

    result5 = await agent.execute(prompt5)

    print("\nPrompt: Creative names for a web scraping library")
    print("Top-K: 40 (controlled randomness)")
    print(f"\nResponse:\n{result5.content}")
    print(f"\nTokens: {result5.total_tokens}")
    print(f"Cost: ${result5.cost_usd:.6f}")

    print("\n" + "=" * 60)
    print("Example 6: Long Context (1M tokens for Gemini 1.5)")
    print("=" * 60)

    # Example 6: Long context capability
    long_text = "Python is a versatile programming language. " * 200

    prompt6 = create_simple_prompt(
        f"Here is a document:\n\n{long_text}\n\nWhat programming language is mentioned? Answer in one word.",
        model="gemini-1.5-flash",
        temperature=0.1,
    )

    result6 = await agent.execute(prompt6)

    print(f"\nDocument length: {len(long_text)} characters")
    print(f"Model: {result6.model} (supports up to 1M tokens!)")
    print(f"\nResponse:\n{result6.content}")
    print(f"\nTokens: {result6.total_tokens}")
    print(f"Cost: ${result6.cost_usd:.6f}")

    print("\n" + "=" * 60)
    print("Summary: Total Cost and Performance")
    print("=" * 60)

    total_cost = (
        result1.cost_usd
        + result2.cost_usd
        + result3.cost_usd
        + result4.cost_usd
        + result5.cost_usd
        + result6.cost_usd
    )
    total_tokens = (
        result1.total_tokens
        + result2.total_tokens
        + result3.total_tokens
        + result4.total_tokens
        + result5.total_tokens
        + result6.total_tokens
    )

    print(f"\nTotal tokens used: {total_tokens} (estimated)")
    print(f"Total cost: ${total_cost:.6f}")
    print("\nGemini advantages:")
    print("  - Very cost-effective pricing")
    print("  - Gemini 1.5 Flash is extremely fast and cheap")
    print("  - 1M token context window (Gemini 1.5)")
    print("  - Built-in safety ratings")
    print("  - Fine-grained control with top-k, top-p, temperature")


if __name__ == "__main__":
    asyncio.run(main())
