"""Testing utilities for agent-orchestration-lib.

This module provides utilities to make testing agents easier, including:
- Mock agents for testing
- Agent spies for tracking calls
- Test fixtures for common test scenarios
- Assertion helpers for agent testing
"""

from .mock_agents import MockAgent, AgentSpy
from .fixtures import (
    create_test_context,
    create_test_emitter,
    create_test_flow,
)
from .assertions import (
    assert_agent_called,
    assert_event_emitted,
    assert_context_has,
)

__all__ = [
    # Mock agents
    "MockAgent",
    "AgentSpy",
    # Fixtures
    "create_test_context",
    "create_test_emitter",
    "create_test_flow",
    # Assertions
    "assert_agent_called",
    "assert_event_emitted",
    "assert_context_has",
]
