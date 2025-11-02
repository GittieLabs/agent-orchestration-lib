"""Unit tests for unified tool calling functionality."""

import pytest
from pydantic import ValidationError

from agent_lib.integrations.common import (
    ToolDefinition,
    ToolParameter,
    ToolCall,
    ToolResult,
)
from agent_lib.integrations.openai.models import OpenAIMessage, OpenAIPrompt, OpenAIResponse
from agent_lib.integrations.openai.tool_utils import tool_definition_to_openai


class TestToolModels:
    """Test unified tool models."""

    def test_tool_parameter_basic(self):
        """Test basic ToolParameter creation."""
        param = ToolParameter(
            type="string",
            description="A test parameter",
        )
        assert param.type == "string"
        assert param.description == "A test parameter"
        assert param.enum is None
        assert param.default is None

    def test_tool_parameter_with_enum(self):
        """Test ToolParameter with enum values."""
        param = ToolParameter(
            type="string",
            description="Units",
            enum=["celsius", "fahrenheit"],
            default="celsius",
        )
        assert param.enum == ["celsius", "fahrenheit"]
        assert param.default == "celsius"

    def test_tool_parameter_object_type(self):
        """Test ToolParameter with object type."""
        param = ToolParameter(
            type="object",
            description="Coordinates",
            properties={
                "lat": {"type": "number"},
                "lon": {"type": "number"},
            },
            required=["lat", "lon"],
        )
        assert param.type == "object"
        assert "lat" in param.properties
        assert param.required == ["lat", "lon"]

    def test_tool_definition_basic(self):
        """Test basic ToolDefinition creation."""
        tool = ToolDefinition(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "location": ToolParameter(type="string", description="City name"),
            },
            required=["location"],
        )
        assert tool.name == "get_weather"
        assert "location" in tool.parameters
        assert tool.required == ["location"]

    def test_tool_definition_multiple_params(self):
        """Test ToolDefinition with multiple parameters."""
        tool = ToolDefinition(
            name="search",
            description="Search database",
            parameters={
                "query": ToolParameter(type="string", description="Search query"),
                "limit": ToolParameter(type="integer", description="Max results"),
                "offset": ToolParameter(type="integer", description="Start offset"),
            },
            required=["query"],
        )
        assert len(tool.parameters) == 3
        assert tool.required == ["query"]

    def test_tool_call_creation(self):
        """Test ToolCall creation."""
        call = ToolCall(
            id="call_123",
            name="get_weather",
            arguments={"location": "San Francisco", "units": "celsius"},
        )
        assert call.id == "call_123"
        assert call.name == "get_weather"
        assert call.arguments["location"] == "San Francisco"

    def test_tool_result_success(self):
        """Test ToolResult for successful execution."""
        result = ToolResult(
            tool_call_id="call_123",
            name="get_weather",
            result={"temperature": 22, "condition": "sunny"},
        )
        assert result.tool_call_id == "call_123"
        assert result.error is None
        assert result.result["temperature"] == 22

    def test_tool_result_error(self):
        """Test ToolResult for failed execution."""
        result = ToolResult(
            tool_call_id="call_456",
            name="broken_tool",
            result=None,
            error="Tool execution failed: Connection timeout",
        )
        assert result.error is not None
        assert "timeout" in result.error


class TestOpenAIToolIntegration:
    """Test OpenAI-specific tool integration."""

    def test_openai_message_with_tool_role(self):
        """Test OpenAIMessage with tool role."""
        msg = OpenAIMessage(
            role="tool",
            content="Weather data: 72F",
            tool_call_id="call_123",
            name="get_weather",
        )
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_123"
        assert msg.name == "get_weather"

    def test_openai_message_with_tool_calls(self):
        """Test OpenAIMessage with tool_calls."""
        tool_call = ToolCall(
            id="call_456",
            name="calculate",
            arguments={"expression": "2 + 2"},
        )
        msg = OpenAIMessage(
            role="assistant",
            content=None,
            tool_calls=[tool_call],
        )
        assert msg.role == "assistant"
        assert msg.content is None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "calculate"

    def test_openai_prompt_with_tools(self):
        """Test OpenAIPrompt with tools."""
        tool = ToolDefinition(
            name="get_weather",
            description="Get weather",
            parameters={
                "location": ToolParameter(type="string", description="City"),
            },
            required=["location"],
        )
        prompt = OpenAIPrompt(
            messages=[OpenAIMessage(role="user", content="Hello")],
            tools=[tool],
            tool_choice="auto",
        )
        assert len(prompt.tools) == 1
        assert prompt.tool_choice == "auto"

    def test_openai_response_with_tool_calls(self):
        """Test OpenAIResponse with tool_calls."""
        tool_call = ToolCall(
            id="call_789",
            name="get_weather",
            arguments={"location": "NYC"},
        )
        response = OpenAIResponse(
            content=None,
            model="gpt-4",
            finish_reason="tool_calls",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost_usd=0.001,
            tool_calls=[tool_call],
        )
        assert response.content is None
        assert response.finish_reason == "tool_calls"
        assert len(response.tool_calls) == 1

    def test_tool_definition_to_openai_conversion(self):
        """Test conversion of ToolDefinition to OpenAI format."""
        tool = ToolDefinition(
            name="get_weather",
            description="Get the current weather",
            parameters={
                "location": ToolParameter(type="string", description="City name"),
                "units": ToolParameter(
                    type="string",
                    description="Temperature units",
                    enum=["celsius", "fahrenheit"],
                ),
            },
            required=["location"],
        )

        openai_tool = tool_definition_to_openai(tool)

        assert openai_tool["type"] == "function"
        assert openai_tool["function"]["name"] == "get_weather"
        assert openai_tool["function"]["description"] == "Get the current weather"
        assert openai_tool["function"]["parameters"]["type"] == "object"
        assert "location" in openai_tool["function"]["parameters"]["properties"]
        assert "units" in openai_tool["function"]["parameters"]["properties"]
        assert openai_tool["function"]["parameters"]["required"] == ["location"]

        # Verify enum is preserved
        units_param = openai_tool["function"]["parameters"]["properties"]["units"]
        assert units_param["enum"] == ["celsius", "fahrenheit"]


class TestToolValidation:
    """Test tool model validation."""

    def test_tool_parameter_requires_type(self):
        """Test that ToolParameter requires type field."""
        with pytest.raises(ValidationError):
            ToolParameter(description="Missing type")

    def test_tool_definition_requires_name(self):
        """Test that ToolDefinition requires name field."""
        with pytest.raises(ValidationError):
            ToolDefinition(
                description="Missing name",
                parameters={},
            )

    def test_tool_call_requires_all_fields(self):
        """Test that ToolCall requires all fields."""
        with pytest.raises(ValidationError):
            ToolCall(name="test")  # Missing id and arguments

    def test_tool_result_requires_tool_call_id(self):
        """Test that ToolResult requires tool_call_id."""
        with pytest.raises(ValidationError):
            ToolResult(name="test", result={})  # Missing tool_call_id
