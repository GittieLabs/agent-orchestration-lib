"""Conversation agent example.

This example demonstrates how to build a conversational agent that maintains
context across multiple turns using the LLM integrations.

Requirements:
    pip install agent-orchestration-lib[openai]

Environment:
    OPENAI_API_KEY: Your OpenAI API key
"""

import asyncio
import os
from typing import List

from agent_lib import EventEmitter, ExecutionContext
from agent_lib.integrations.openai import OpenAIAgent, OpenAIMessage, OpenAIPrompt


class ConversationAgent:
    """A conversational agent that maintains message history."""

    def __init__(
        self,
        agent: OpenAIAgent,
        system_prompt: str = "You are a helpful assistant.",
        max_history: int = 10,
    ):
        self.agent = agent
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.messages: List[OpenAIMessage] = []

    async def chat(self, user_message: str) -> str:
        """Send a message and get a response, maintaining conversation history."""
        # Add user message to history
        self.messages.append(OpenAIMessage(role="user", content=user_message))

        # Build prompt with full conversation history
        prompt = OpenAIPrompt(
            messages=[OpenAIMessage(role="system", content=self.system_prompt)] + self.messages,
            model="gpt-4",
            temperature=0.7,
        )

        # Get response
        result = await self.agent.execute(prompt)

        # Add assistant response to history
        self.messages.append(OpenAIMessage(role="assistant", content=result.content))

        # Trim history if it gets too long (keep last N exchanges)
        if len(self.messages) > self.max_history * 2:  # *2 because each exchange is 2 messages
            self.messages = self.messages[-self.max_history * 2 :]

        return result.content

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages = []

    def get_history(self) -> List[OpenAIMessage]:
        """Get conversation history."""
        return self.messages.copy()


async def main() -> None:
    """Run conversation agent example."""
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    # Create context and emitter
    context = ExecutionContext()
    emitter = EventEmitter()

    # Create base agent
    base_agent = OpenAIAgent(
        name="gpt4-conversational",
        api_key=api_key,
        context=context,
        emitter=emitter,
    )

    print("=" * 60)
    print("Conversation Agent Example")
    print("=" * 60)

    # Example 1: Simple multi-turn conversation
    print("\n" + "=" * 60)
    print("Example 1: Multi-Turn Conversation")
    print("=" * 60)

    conversation = ConversationAgent(
        agent=base_agent,
        system_prompt="You are a helpful Python programming tutor.",
    )

    conversation_turns = [
        "What is a list in Python?",
        "How do I add an item to it?",
        "What about removing items?",
        "Can you show me a complete example using what we discussed?",
    ]

    for i, user_msg in enumerate(conversation_turns, 1):
        print(f"\n[Turn {i}]")
        print(f"User: {user_msg}")

        response = await conversation.chat(user_msg)
        print(f"Assistant: {response}\n")

    # Show that context was maintained
    print("=" * 60)
    print("Notice how the assistant:")
    print("  • Remembered 'it' refers to a list")
    print("  • Used previous context about adding items")
    print("  • Built on earlier explanations")
    print("=" * 60)

    # Example 2: Specialized assistant
    print("\n" + "=" * 60)
    print("Example 2: Specialized Code Review Assistant")
    print("=" * 60)

    code_reviewer = ConversationAgent(
        agent=base_agent,
        system_prompt="You are an expert code reviewer. Provide concise, actionable feedback.",
    )

    code_snippet = """
def calculate_average(numbers):
    total = 0
    for n in numbers:
        total = total + n
    return total / len(numbers)
"""

    print(f"User: Please review this code:\n{code_snippet}")
    response1 = await code_reviewer.chat(f"Please review this code:\n{code_snippet}")
    print(f"Assistant: {response1}\n")

    print("\nUser: What if the list is empty?")
    response2 = await code_reviewer.chat("What if the list is empty?")
    print(f"Assistant: {response2}\n")

    print("\nUser: Can you show me the improved version?")
    response3 = await code_reviewer.chat("Can you show me the improved version?")
    print(f"Assistant: {response3}")

    # Example 3: Interactive Q&A
    print("\n" + "=" * 60)
    print("Example 3: Context-Aware Q&A")
    print("=" * 60)

    qa_agent = ConversationAgent(
        agent=base_agent,
        system_prompt="You are a knowledgeable assistant. Keep answers concise.",
        max_history=5,
    )

    qa_sequence = [
        "Who wrote 'To Kill a Mockingbird'?",
        "When was it published?",
        "What's it about?",
        "Has it won any awards?",
    ]

    for question in qa_sequence:
        print(f"\nQ: {question}")
        answer = await qa_agent.chat(question)
        print(f"A: {answer}")

    # Example 4: Conversation with history management
    print("\n" + "=" * 60)
    print("Example 4: History Management")
    print("=" * 60)

    agent_with_history = ConversationAgent(
        agent=base_agent,
        system_prompt="You are a helpful assistant.",
        max_history=3,  # Keep only last 3 exchanges
    )

    print("\nDemonstrating history limit (max 3 exchanges)...")

    for i in range(5):
        user_msg = f"This is message number {i+1}. Remember this number!"
        print(f"\nSending: {user_msg}")
        response = await agent_with_history.chat(user_msg)
        print(f"Response: {response[:50]}...")

    print("\nChecking memory...")
    test_response = await agent_with_history.chat("What were the first two numbers I mentioned?")
    print(f"\nQ: What were the first two numbers I mentioned?")
    print(f"A: {test_response}")
    print("\n(The agent should only remember the last 3, not the first 2)")

    # Example 5: Starting fresh
    print("\n" + "=" * 60)
    print("Example 5: Clearing History")
    print("=" * 60)

    reset_agent = ConversationAgent(
        agent=base_agent,
        system_prompt="You are a helpful assistant.",
    )

    print("\n1. First conversation:")
    await reset_agent.chat("My name is Alice")
    response = await reset_agent.chat("What's my name?")
    print(f"Q: What's my name?")
    print(f"A: {response}")

    print("\n2. Clearing history...")
    reset_agent.clear_history()

    print("\n3. After reset:")
    response = await reset_agent.chat("What's my name?")
    print(f"Q: What's my name?")
    print(f"A: {response}")
    print("(Should not remember Alice)")

    # Summary
    print("\n" + "=" * 60)
    print("Summary: Conversation Agent Features")
    print("=" * 60)
    print("\n✓ Maintains conversation context across turns")
    print("✓ System prompts for specialization")
    print("✓ History management (auto-trim old messages)")
    print("✓ Ability to clear history")
    print("✓ Works with any LLM provider")

    print("\nBest Practices:")
    print("  • Set appropriate max_history to manage token usage")
    print("  • Use system prompts to set agent behavior")
    print("  • Clear history when starting new topics")
    print("  • Monitor token usage in long conversations")


if __name__ == "__main__":
    asyncio.run(main())
