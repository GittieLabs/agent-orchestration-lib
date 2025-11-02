"""Utilities for converting between unified and Anthropic tool formats."""

import json
from typing import Any, Dict, List

from ..common.tool_models import ToolCall, ToolDefinition


def tool_definition_to_anthropic(tool: ToolDefinition) -> Dict[str, Any]:
    """Convert unified ToolDefinition to Anthropic tool format.

    Args:
        tool: Unified tool definition

    Returns:
        Anthropic tool definition dict

    Examples:
        >>> tool = ToolDefinition(
        ...     name="get_weather",
        ...     description="Get weather for a location",
        ...     parameters={
        ...         "location": ToolParameter(type="string", description="City name")
        ...     },
        ...     required=["location"]
        ... )
        >>> anthropic_tool = tool_definition_to_anthropic(tool)
        >>> anthropic_tool["name"]
        'get_weather'
    """
    # Build input schema
    input_schema: Dict[str, Any] = {
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

        input_schema["properties"][param_name] = param_schema

    if tool.required:
        input_schema["required"] = tool.required

    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": input_schema,
    }


def anthropic_tool_use_to_unified(tool_use_block: Any) -> ToolCall:
    """Convert Anthropic tool_use block to unified ToolCall.

    Args:
        tool_use_block: Anthropic tool_use content block from API response

    Returns:
        Unified ToolCall

    Examples:
        >>> # Assuming tool_use_block is from API response
        >>> unified = anthropic_tool_use_to_unified(tool_use_block)
        >>> unified.name
        'get_weather'
    """
    return ToolCall(
        id=tool_use_block.id,
        name=tool_use_block.name,
        arguments=tool_use_block.input,
    )
