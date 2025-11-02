"""Utilities for converting between unified and Gemini tool formats."""

from typing import Any, Dict

from ..common.tool_models import ToolCall, ToolDefinition


def tool_definition_to_gemini(tool: ToolDefinition) -> Dict[str, Any]:
    """Convert unified ToolDefinition to Gemini function declaration format.

    Args:
        tool: Unified tool definition

    Returns:
        Gemini function declaration dict

    Examples:
        >>> tool = ToolDefinition(
        ...     name="get_weather",
        ...     description="Get weather for a location",
        ...     parameters={
        ...         "location": ToolParameter(type="string", description="City name")
        ...     },
        ...     required=["location"]
        ... )
        >>> gemini_func = tool_definition_to_gemini(tool)
        >>> gemini_func["name"]
        'get_weather'
    """
    # Build parameters schema
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
    }

    for param_name, param_def in tool.parameters.items():
        param_schema: Dict[str, Any] = {
            "type": param_def.type.upper(),  # Gemini uses uppercase types
            "description": param_def.description,
        }

        if param_def.enum is not None:
            param_schema["enum"] = param_def.enum
        if param_def.items is not None:
            param_schema["items"] = param_def.items
        if param_def.properties is not None:
            param_schema["properties"] = param_def.properties

        parameters_schema["properties"][param_name] = param_schema

    if tool.required:
        parameters_schema["required"] = tool.required

    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": parameters_schema,
    }


def gemini_function_call_to_unified(function_call: Any, call_id: str) -> ToolCall:
    """Convert Gemini function call to unified ToolCall.

    Args:
        function_call: Gemini function call object from API response
        call_id: Generated ID for this tool call (Gemini doesn't provide IDs)

    Returns:
        Unified ToolCall

    Examples:
        >>> # Assuming function_call is from API response
        >>> unified = gemini_function_call_to_unified(function_call, "call_1")
        >>> unified.name
        'get_weather'
    """
    return ToolCall(
        id=call_id,
        name=function_call.name,
        arguments=dict(function_call.args),
    )
