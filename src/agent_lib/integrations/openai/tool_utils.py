"""Utilities for converting between unified and OpenAI tool formats."""

import json
from typing import Any, Dict, List

from ..common.tool_models import ToolCall, ToolDefinition, ToolParameter


def tool_definition_to_openai(tool: ToolDefinition) -> Dict[str, Any]:
    """Convert unified ToolDefinition to OpenAI function format.

    Args:
        tool: Unified tool definition

    Returns:
        OpenAI function definition dict

    Examples:
        >>> tool = ToolDefinition(
        ...     name="get_weather",
        ...     description="Get weather for a location",
        ...     parameters={
        ...         "location": ToolParameter(type="string", description="City name")
        ...     },
        ...     required=["location"]
        ... )
        >>> openai_tool = tool_definition_to_openai(tool)
        >>> openai_tool["type"]
        'function'
        >>> openai_tool["function"]["name"]
        'get_weather'
    """
    # Convert parameters to JSON Schema format
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
    }

    for param_name, param_def in tool.parameters.items():
        param_schema: Dict[str, Any] = {
            "type": param_def.type,
            "description": param_def.description,
        }

        if param_def.enum is not None:
            param_schema["enum"] = param_def.enum
        if param_def.items is not None:
            param_schema["items"] = param_def.items
        if param_def.properties is not None:
            param_schema["properties"] = param_def.properties
        if param_def.required is not None:
            param_schema["required"] = param_def.required
        if param_def.default is not None:
            param_schema["default"] = param_def.default

        parameters_schema["properties"][param_name] = param_schema

    if tool.required:
        parameters_schema["required"] = tool.required

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters_schema,
        },
    }


def openai_tool_call_to_unified(openai_tool_call: Any) -> ToolCall:
    """Convert OpenAI tool call to unified ToolCall.

    Args:
        openai_tool_call: OpenAI tool call object from API response

    Returns:
        Unified ToolCall

    Examples:
        >>> # Assuming openai_tool_call is from API response
        >>> unified = openai_tool_call_to_unified(openai_tool_call)
        >>> unified.name
        'get_weather'
        >>> unified.arguments
        {'location': 'San Francisco'}
    """
    # Parse arguments JSON string to dict
    arguments = json.loads(openai_tool_call.function.arguments)

    return ToolCall(
        id=openai_tool_call.id,
        name=openai_tool_call.function.name,
        arguments=arguments,
    )
