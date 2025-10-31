"""LLM cost tracking example.

This example demonstrates how to track and manage LLM costs across workflows using
the built-in cost estimation features.

Requirements:
    pip install agent-orchestration-lib[openai]

Environment:
    OPENAI_API_KEY: Your OpenAI API key
"""

import asyncio
import os
from typing import List

from agent_lib import EventEmitter, ExecutionContext
from agent_lib.integrations.openai import OpenAIAgent, create_simple_prompt


class CostTracker:
    """Simple cost tracking utility."""

    def __init__(self) -> None:
        self.total_cost = 0.0
        self.total_tokens = 0
        self.requests: List[dict] = []

    def track(self, name: str, result: any) -> None:
        """Track a request's cost and tokens."""
        self.total_cost += result.cost_usd
        self.total_tokens += result.total_tokens
        self.requests.append(
            {
                "name": name,
                "tokens": result.total_tokens,
                "cost": result.cost_usd,
                "model": result.model,
            }
        )

    def report(self) -> None:
        """Print cost report."""
        print("\n" + "=" * 60)
        print("COST TRACKING REPORT")
        print("=" * 60)

        print(f"\nTotal Requests: {len(self.requests)}")
        print(f"Total Tokens: {self.total_tokens:,}")
        print(f"Total Cost: ${self.total_cost:.6f}")

        if self.requests:
            avg_cost = self.total_cost / len(self.requests)
            avg_tokens = self.total_tokens / len(self.requests)
            print(f"Average Cost/Request: ${avg_cost:.6f}")
            print(f"Average Tokens/Request: {avg_tokens:.0f}")

            print("\nPer-Request Breakdown:")
            for i, req in enumerate(self.requests, 1):
                print(
                    f"  {i}. {req['name'][:40]:<40} | "
                    f"{req['tokens']:>6} tokens | ${req['cost']:.6f} | {req['model']}"
                )


async def main() -> None:
    """Run cost tracking example."""
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    # Create context and emitter
    context = ExecutionContext()
    emitter = EventEmitter()

    # Create cost tracker
    tracker = CostTracker()

    print("=" * 60)
    print("LLM Cost Tracking Example")
    print("=" * 60)
    print("\nThis example demonstrates:")
    print("  • Tracking costs across multiple requests")
    print("  • Comparing costs between models")
    print("  • Budget management")
    print("  • Cost optimization strategies")

    # Create agents for different models
    gpt4_agent = OpenAIAgent(
        name="gpt4",
        api_key=api_key,
        context=context,
        emitter=emitter,
        default_model="gpt-4",
    )

    gpt35_agent = OpenAIAgent(
        name="gpt35",
        api_key=api_key,
        context=context,
        emitter=emitter,
        default_model="gpt-3.5-turbo",
    )

    # Example 1: Simple tasks with GPT-3.5 (cheaper)
    print("\n" + "=" * 60)
    print("Example 1: Use Cheaper Models for Simple Tasks")
    print("=" * 60)

    simple_tasks = [
        "What is 2+2?",
        "Name 3 colors",
        "What day comes after Monday?",
        "Translate 'hello' to Spanish",
        "What is the capital of Japan?",
    ]

    print("\nExecuting 5 simple tasks with GPT-3.5-turbo...")
    for task in simple_tasks:
        prompt = create_simple_prompt(task, model="gpt-3.5-turbo")
        result = await gpt35_agent.execute(prompt)
        tracker.track(task, result)
        print(f"  ✓ {task[:40]:<40} ${result.cost_usd:.6f}")

    # Example 2: Complex tasks with GPT-4 (better quality)
    print("\n" + "=" * 60)
    print("Example 2: Use Premium Models for Complex Tasks")
    print("=" * 60)

    complex_tasks = [
        "Explain the concept of recursion with a Python example",
        "Write a function to implement binary search",
    ]

    print("\nExecuting 2 complex tasks with GPT-4...")
    for task in complex_tasks:
        prompt = create_simple_prompt(task, model="gpt-4", max_tokens=500)
        result = await gpt4_agent.execute(prompt)
        tracker.track(task, result)
        print(f"  ✓ {task[:40]:<40} ${result.cost_usd:.6f}")

    # Example 3: Workflow with mixed models
    print("\n" + "=" * 60)
    print("Example 3: Mixed Model Workflow")
    print("=" * 60)

    print("\nWorkflow: Document Processing Pipeline")
    print("  1. Extract keywords (GPT-3.5) - cheap")
    print("  2. Analyze sentiment (GPT-3.5) - cheap")
    print("  3. Generate summary (GPT-4) - quality matters")

    document = "Python is an amazing programming language. It's easy to learn and very powerful."

    # Step 1: Extract keywords (cheap model)
    prompt1 = create_simple_prompt(
        f"Extract 3 keywords from: {document}", model="gpt-3.5-turbo", max_tokens=50
    )
    result1 = await gpt35_agent.execute(prompt1)
    tracker.track("Extract keywords", result1)
    print(f"  ✓ Keywords: {result1.content} (${result1.cost_usd:.6f})")

    # Step 2: Analyze sentiment (cheap model)
    prompt2 = create_simple_prompt(
        f"Sentiment (positive/negative/neutral): {document}",
        model="gpt-3.5-turbo",
        max_tokens=10,
    )
    result2 = await gpt35_agent.execute(prompt2)
    tracker.track("Analyze sentiment", result2)
    print(f"  ✓ Sentiment: {result2.content} (${result2.cost_usd:.6f})")

    # Step 3: Generate summary (premium model for quality)
    prompt3 = create_simple_prompt(
        f"Write a professional summary: {document}",
        model="gpt-4",
        max_tokens=100,
    )
    result3 = await gpt4_agent.execute(prompt3)
    tracker.track("Generate summary", result3)
    print(f"  ✓ Summary: {result3.content[:50]}... (${result3.cost_usd:.6f})")

    print(f"\nWorkflow total cost: ${result1.cost_usd + result2.cost_usd + result3.cost_usd:.6f}")

    # Display final report
    tracker.report()

    # Cost optimization tips
    print("\n" + "=" * 60)
    print("COST OPTIMIZATION TIPS")
    print("=" * 60)

    simple_cost = sum(r["cost"] for r in tracker.requests if "gpt-3.5" in r["model"])
    complex_cost = sum(r["cost"] for r in tracker.requests if "gpt-4" in r["model"])

    print("\nCost breakdown:")
    print(
        f"  GPT-3.5-turbo requests: ${simple_cost:.6f} ({len([r for r in tracker.requests if 'gpt-3.5' in r['model']])} requests)"
    )
    print(
        f"  GPT-4 requests: ${complex_cost:.6f} ({len([r for r in tracker.requests if 'gpt-4' in r['model']])} requests)"
    )

    print("\n✓ Optimization strategies:")
    print("  1. Use GPT-3.5 for simple tasks (75% cheaper)")
    print("  2. Use GPT-4 only when quality matters")
    print("  3. Set max_tokens to limit costs")
    print("  4. Cache responses when possible")
    print("  5. Batch similar requests together")

    # Budget management example
    print("\n" + "=" * 60)
    print("BUDGET MANAGEMENT")
    print("=" * 60)

    budget_limit = 0.10  # $0.10 budget
    print(f"\nBudget limit: ${budget_limit:.2f}")
    print(f"Current spend: ${tracker.total_cost:.6f}")

    if tracker.total_cost <= budget_limit:
        remaining = budget_limit - tracker.total_cost
        print(f"✓ Within budget! Remaining: ${remaining:.6f}")
    else:
        overage = tracker.total_cost - budget_limit
        print(f"✗ Over budget by ${overage:.6f}")

    # Estimate future costs
    avg_cost_per_request = tracker.total_cost / len(tracker.requests)
    remaining_requests = int((budget_limit - tracker.total_cost) / avg_cost_per_request)

    print(f"\nWith remaining budget, you can make ~{max(0, remaining_requests)} more requests")
    print(f"(based on average cost of ${avg_cost_per_request:.6f}/request)")


if __name__ == "__main__":
    asyncio.run(main())
