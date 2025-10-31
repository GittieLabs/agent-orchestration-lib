"""LLM comparison example.

This example compares responses from different LLM providers (OpenAI, Anthropic, Gemini)
for the same prompt, showing differences in:
- Response quality
- Cost
- Latency
- Token usage

Requirements:
    pip install agent-orchestration-lib[all-llm]

Environment:
    OPENAI_API_KEY: Your OpenAI API key
    ANTHROPIC_API_KEY: Your Anthropic API key
    GOOGLE_API_KEY: Your Google API key
"""

import asyncio
import os
import time

from agent_lib import EventEmitter, ExecutionContext
from agent_lib.integrations.anthropic import AnthropicAgent, create_simple_prompt as anthropic_prompt
from agent_lib.integrations.gemini import GeminiAgent, create_simple_prompt as gemini_prompt
from agent_lib.integrations.openai import OpenAIAgent, create_simple_prompt as openai_prompt


async def compare_llms(question: str) -> None:
    """Compare LLM responses for a given question."""
    # Get API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    # Create context and emitter
    context = ExecutionContext()
    emitter = EventEmitter()

    print(f"\nQuestion: {question}")
    print("=" * 80)

    results = []

    # Test OpenAI GPT-4
    if openai_key:
        agent = OpenAIAgent(
            name="gpt4",
            api_key=openai_key,
            context=context,
            emitter=emitter,
        )

        prompt = openai_prompt(question, model="gpt-4", temperature=0.7)

        start_time = time.time()
        result = await agent.execute(prompt)
        latency = time.time() - start_time

        results.append(
            {
                "provider": "OpenAI GPT-4",
                "response": result.content,
                "tokens": result.total_tokens,
                "cost": result.cost_usd,
                "latency": latency,
            }
        )

    # Test Anthropic Claude
    if anthropic_key:
        agent = AnthropicAgent(
            name="claude",
            api_key=anthropic_key,
            context=context,
            emitter=emitter,
        )

        prompt = anthropic_prompt(
            question, model="claude-3-sonnet-20240229", max_tokens=1024, temperature=0.7
        )

        start_time = time.time()
        result = await agent.execute(prompt)
        latency = time.time() - start_time

        results.append(
            {
                "provider": "Claude 3 Sonnet",
                "response": result.content,
                "tokens": result.total_tokens,
                "cost": result.cost_usd,
                "latency": latency,
            }
        )

    # Test Google Gemini
    if google_key:
        agent = GeminiAgent(
            name="gemini",
            api_key=google_key,
            context=context,
            emitter=emitter,
        )

        prompt = gemini_prompt(question, model="gemini-pro", temperature=0.7)

        start_time = time.time()
        result = await agent.execute(prompt)
        latency = time.time() - start_time

        results.append(
            {
                "provider": "Gemini Pro",
                "response": result.content,
                "tokens": result.total_tokens,
                "cost": result.cost_usd,
                "latency": latency,
            }
        )

    # Display results
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['provider']}")
        print("-" * 80)
        print(f"Response:\n{result['response']}\n")
        print(f"Tokens: {result['tokens']}")
        print(f"Cost: ${result['cost']:.6f}")
        print(f"Latency: {result['latency']:.2f}s")

    # Summary
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        cheapest = min(results, key=lambda x: x["cost"])
        fastest = min(results, key=lambda x: x["latency"])
        most_verbose = max(results, key=lambda x: len(x["response"]))

        print(f"\n💰 Cheapest: {cheapest['provider']} (${cheapest['cost']:.6f})")
        print(f"⚡ Fastest: {fastest['provider']} ({fastest['latency']:.2f}s)")
        print(f"📝 Most detailed: {most_verbose['provider']} ({len(most_verbose['response'])} chars)")

        total_cost = sum(r["cost"] for r in results)
        avg_latency = sum(r["latency"] for r in results) / len(results)

        print(f"\nTotal cost: ${total_cost:.6f}")
        print(f"Average latency: {avg_latency:.2f}s")


async def main() -> None:
    """Run LLM comparison examples."""
    print("=" * 80)
    print("LLM Provider Comparison")
    print("=" * 80)

    # Test 1: Simple factual question
    print("\n" + "=" * 80)
    print("TEST 1: Simple Factual Question")
    print("=" * 80)
    await compare_llms("What is the capital of France? Answer in 1-2 sentences.")

    # Test 2: Creative task
    print("\n" + "=" * 80)
    print("TEST 2: Creative Task")
    print("=" * 80)
    await compare_llms("Write a haiku about programming.")

    # Test 3: Technical explanation
    print("\n" + "=" * 80)
    print("TEST 3: Technical Explanation")
    print("=" * 80)
    await compare_llms(
        "Explain what a REST API is in 2-3 sentences. Be clear and concise."
    )

    # Test 4: Code generation
    print("\n" + "=" * 80)
    print("TEST 4: Code Generation")
    print("=" * 80)
    await compare_llms(
        "Write a Python function to check if a string is a palindrome. Include docstring."
    )

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  • Different providers have different strengths")
    print("  • Cost varies significantly between providers")
    print("  • Latency depends on provider load and model")
    print("  • Choose based on your specific needs:")
    print("    - Simple tasks → Gemini Flash (fast & cheap)")
    print("    - Balanced needs → Claude Sonnet or GPT-3.5")
    print("    - Best quality → GPT-4 or Claude Opus")


if __name__ == "__main__":
    asyncio.run(main())
