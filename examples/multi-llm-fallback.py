"""Multi-LLM fallback example.

This example demonstrates how to use the LLMFallbackRetry strategy to automatically
fall back between different LLM providers when one fails or is unavailable.

This is useful for:
- High availability (if OpenAI is down, use Anthropic)
- Cost optimization (try cheaper model first, fall back to premium)
- Rate limit handling (switch providers when rate-limited)

Requirements:
    pip install agent-orchestration-lib[all-llm]

Environment:
    OPENAI_API_KEY: Your OpenAI API key
    ANTHROPIC_API_KEY: Your Anthropic API key
    GOOGLE_API_KEY: Your Google API key
"""

import asyncio
import os

from agent_lib import EventEmitter, ExecutionContext, Flow
from agent_lib.integrations.anthropic import (
    AnthropicAgent,
)
from agent_lib.integrations.anthropic import (
    create_simple_prompt as anthropic_prompt,
)
from agent_lib.integrations.gemini import GeminiAgent
from agent_lib.integrations.gemini import create_simple_prompt as gemini_prompt
from agent_lib.integrations.openai import OpenAIAgent
from agent_lib.integrations.openai import create_simple_prompt as openai_prompt
from agent_lib.retry import LLMFallbackRetry


async def main() -> None:
    """Run the multi-LLM fallback example."""
    # Get API keys from environment
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    # Create context and emitter
    context = ExecutionContext()
    emitter = EventEmitter()

    print("=" * 60)
    print("Multi-LLM Fallback Strategy Example")
    print("=" * 60)
    print("\nThis example shows automatic fallback between providers:")
    print("  1. Try OpenAI GPT-4 first (premium quality)")
    print("  2. Fall back to Anthropic Claude if OpenAI fails")
    print("  3. Fall back to Gemini if Claude fails")
    print("=" * 60)

    # Example 1: Primary model works
    if openai_key:
        print("\n" + "=" * 60)
        print("Example 1: OpenAI Available (No Fallback Needed)")
        print("=" * 60)

        agent = OpenAIAgent(
            name="gpt4-agent",
            api_key=openai_key,
            context=context,
            emitter=emitter,
        )

        # Add fallback strategy
        fallback_strategy = LLMFallbackRetry(
            models=["gpt-4", "claude-3-sonnet-20240229", "gemini-pro"],
            max_retries=2,
        )
        agent.retry_strategy = fallback_strategy

        prompt = openai_prompt("What is the capital of France? One word answer.")

        result = await agent.execute(prompt)

        print(f"\n✓ Success with primary model: {result.model}")
        print(f"Response: {result.content}")
        print(f"Cost: ${result.cost_usd:.6f}")
        print("\nNo fallback was needed - primary model succeeded!")

    # Example 2: Fallback scenario (simulated by using wrong API key or unavailable model)
    print("\n" + "=" * 60)
    print("Example 2: Fallback in Action")
    print("=" * 60)
    print("\nIn production, fallback happens automatically when:")
    print("  - API is down or rate-limited")
    print("  - API key is invalid")
    print("  - Model is unavailable")
    print("  - Request times out")

    # Example 3: Multi-provider workflow with manual fallback
    if anthropic_key and google_key:
        print("\n" + "=" * 60)
        print("Example 3: Multi-Provider Workflow")
        print("=" * 60)

        # Create multiple agents
        claude_agent = AnthropicAgent(
            name="claude-agent",
            api_key=anthropic_key,
            context=context,
            emitter=emitter,
        )

        gemini_agent = GeminiAgent(
            name="gemini-agent",
            api_key=google_key,
            context=context,
            emitter=emitter,
        )

        # Try Claude first
        print("\nTrying Claude Sonnet...")
        claude_prompt = anthropic_prompt(
            "What is recursion in programming? Answer in one sentence.",
            model="claude-3-sonnet-20240229",
            max_tokens=512,
        )

        try:
            claude_result = await claude_agent.execute(claude_prompt)
            print(f"✓ Claude response: {claude_result.content}")
            print(f"  Cost: ${claude_result.cost_usd:.6f}")
        except Exception as e:
            print(f"✗ Claude failed: {e}")
            print("\nFalling back to Gemini...")

            gemini_fallback_prompt = gemini_prompt(
                "What is recursion in programming? Answer in one sentence.",
                model="gemini-pro",
            )

            gemini_result = await gemini_agent.execute(gemini_fallback_prompt)
            print(f"✓ Gemini response: {gemini_result.content}")
            print(f"  Cost: ${gemini_result.cost_usd:.6f}")

    # Example 4: Cost-optimized fallback (try cheap model first)
    if google_key and anthropic_key:
        print("\n" + "=" * 60)
        print("Example 4: Cost-Optimized Fallback")
        print("=" * 60)
        print("\nStrategy: Try cheapest model first, fall back to premium if needed")

        # Try Gemini Flash first (cheapest)
        gemini_agent = GeminiAgent(
            name="gemini-cheap",
            api_key=google_key,
            context=context,
            emitter=emitter,
        )

        print("\n1. Trying Gemini 1.5 Flash (cheapest)...")
        cheap_prompt = gemini_prompt(
            "Solve: What is 2+2? Just the number.",
            model="gemini-1.5-flash",
        )

        try:
            cheap_result = await gemini_agent.execute(cheap_prompt)
            print(f"   ✓ Success: {cheap_result.content}")
            print(f"   Cost: ${cheap_result.cost_usd:.6f} (very cheap!)")
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            print("\n2. Falling back to Claude Sonnet (mid-tier)...")

            claude_agent = AnthropicAgent(
                name="claude-midtier",
                api_key=anthropic_key,
                context=context,
                emitter=emitter,
            )

            midtier_prompt = anthropic_prompt(
                "Solve: What is 2+2? Just the number.",
                model="claude-3-sonnet-20240229",
                max_tokens=512,
            )

            midtier_result = await claude_agent.execute(midtier_prompt)
            print(f"   ✓ Success: {midtier_result.content}")
            print(f"   Cost: ${midtier_result.cost_usd:.6f}")

    # Example 5: Parallel execution across providers (comparison, not fallback)
    if openai_key and anthropic_key and google_key:
        print("\n" + "=" * 60)
        print("Example 5: Parallel Execution (Comparison)")
        print("=" * 60)
        print("\nAsking the same question to all three providers simultaneously...")

        flow = Flow(context=context, emitter=emitter)

        # Create agents
        gpt_agent = OpenAIAgent(
            name="gpt",
            api_key=openai_key,
            context=context,
            emitter=emitter,
        )

        claude_agent = AnthropicAgent(
            name="claude",
            api_key=anthropic_key,
            context=context,
            emitter=emitter,
        )

        gemini_agent = GeminiAgent(
            name="gemini",
            api_key=google_key,
            context=context,
            emitter=emitter,
        )

        # Create prompts (same question)
        question = "What is the meaning of 'async' in Python? One sentence."

        # Note: Different prompt formats for different providers
        # In a real implementation, you'd need a wrapper to normalize this
        print(f"\nQuestion: {question}")
        print("\nResponses:")

        # Execute in parallel
        results = await flow.execute_parallel_same_input(
            agents=[gpt_agent, claude_agent, gemini_agent],
            input_data=openai_prompt(question),  # This works for this demo
        )

        for i, result in enumerate(results):
            agent_name = ["GPT-4", "Claude", "Gemini"][i]
            print(f"\n{agent_name}:")
            print(f"  {result.content[:100]}...")
            print(f"  Cost: ${result.cost_usd:.6f}")

    print("\n" + "=" * 60)
    print("Summary: Fallback Strategy Benefits")
    print("=" * 60)
    print("\n✓ High Availability: Automatic failover between providers")
    print("✓ Cost Optimization: Try cheaper models first")
    print("✓ Rate Limit Handling: Switch when rate-limited")
    print("✓ Resilience: No single point of failure")
    print("\nBest Practice:")
    print("  - Always configure at least 2 providers for production")
    print("  - Order by: reliability > cost > performance")
    print("  - Monitor fallback frequency to detect issues")


if __name__ == "__main__":
    asyncio.run(main())
