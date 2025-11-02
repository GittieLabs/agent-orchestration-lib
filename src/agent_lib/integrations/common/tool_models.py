"""Universal tool/function calling models for LLM integrations.

These models provide a provider-agnostic interface for tool calling across
OpenAI, Anthropic, and Gemini. Each provider's agent translates between
these unified models and provider-specific formats.

Key Design Principles:
- Single source of truth for tool definitions
- Consistent format for tool calls and results across providers
- Type-safe with Pydantic validation
- Based on JSON Schema for parameter definitions

Examples:
    Define a tool:
    >>> tool = ToolDefinition(
    ...     name="get_weather",
    ...     description="Get weather for a location",
    ...     parameters={
    ...         "location": ToolParameter(
    ...             type="string",
    ...             description="City name"
    ...         ),
    ...         "units": ToolParameter(
    ...             type="string",
    ...             description="Temperature units",
    ...             enum=["celsius", "fahrenheit"]
    ...         )
    ...     },
    ...     required=["location"]
    ... )

    Handle tool call from LLM:
    >>> tool_call = ToolCall(
    ...     id="call_123",
    ...     name="get_weather",
    ...     arguments={"location": "San Francisco", "units": "celsius"}
    ... )

    Return tool result:
    >>> result = ToolResult(
    ...     tool_call_id="call_123",
    ...     name="get_weather",
    ...     result={"temperature": 18, "condition": "sunny"}
    ... )
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """Parameter definition for a tool (based on JSON Schema).

    Attributes:
        type: JSON Schema type (string, number, integer, boolean, object, array, null)
        description: Human-readable description of what this parameter does
        enum: List of allowed values for this parameter (optional)
        items: Schema for array items if type is "array" (optional)
        properties: Schema for object properties if type is "object" (optional)
        required: List of required property names if type is "object" (optional)
        default: Default value for this parameter (optional)

    Examples:
        String parameter with enum:
        >>> param = ToolParameter(
        ...     type="string",
        ...     description="Temperature units",
        ...     enum=["celsius", "fahrenheit"],
        ...     default="celsius"
        ... )

        Object parameter with nested properties:
        >>> param = ToolParameter(
        ...     type="object",
        ...     description="Location coordinates",
        ...     properties={
        ...         "lat": {"type": "number", "description": "Latitude"},
        ...         "lon": {"type": "number", "description": "Longitude"}
        ...     },
        ...     required=["lat", "lon"]
        ... )

        Array parameter:
        >>> param = ToolParameter(
        ...     type="array",
        ...     description="List of city names",
        ...     items={"type": "string"}
        ... )
    """

    type: str = Field(
        description="JSON Schema type (string, number, integer, boolean, object, array, null)"
    )
    description: str = Field(description="Description of the parameter")
    enum: Optional[List[Any]] = Field(default=None, description="Allowed values")
    items: Optional[Dict[str, Any]] = Field(default=None, description="Array item schema")
    properties: Optional[Dict[str, Any]] = Field(default=None, description="Object properties")
    required: Optional[List[str]] = Field(default=None, description="Required properties")
    default: Optional[Any] = Field(default=None, description="Default value")


class ToolDefinition(BaseModel):
    """Universal tool definition compatible with all LLM providers.

    This model represents a tool/function that an LLM can call. It uses JSON Schema
    for parameter definitions, which all major LLM providers support.

    Attributes:
        name: Tool name (must be alphanumeric with underscores, no spaces)
        description: Clear description of what the tool does and when to use it
        parameters: Dictionary mapping parameter names to their schemas
        required: List of required parameter names (optional, can also be specified per-parameter)

    Examples:
        Simple tool with required parameters:
        >>> tool = ToolDefinition(
        ...     name="search_database",
        ...     description="Search the product database for items matching a query",
        ...     parameters={
        ...         "query": ToolParameter(
        ...             type="string",
        ...             description="Search query"
        ...         ),
        ...         "limit": ToolParameter(
        ...             type="integer",
        ...             description="Maximum results to return",
        ...             default=10
        ...         )
        ...     },
        ...     required=["query"]
        ... )

        Tool with complex parameters:
        >>> tool = ToolDefinition(
        ...     name="book_flight",
        ...     description="Book a flight with specified parameters",
        ...     parameters={
        ...         "origin": ToolParameter(type="string", description="Departure airport code"),
        ...         "destination": ToolParameter(type="string", description="Arrival airport code"),
        ...         "date": ToolParameter(type="string", description="Flight date (YYYY-MM-DD)"),
        ...         "passengers": ToolParameter(
        ...             type="integer",
        ...             description="Number of passengers",
        ...             default=1
        ...         ),
        ...         "class": ToolParameter(
        ...             type="string",
        ...             description="Cabin class",
        ...             enum=["economy", "business", "first"],
        ...             default="economy"
        ...         )
        ...     },
        ...     required=["origin", "destination", "date"]
        ... )
    """

    name: str = Field(description="Tool name (alphanumeric + underscores only)")
    description: str = Field(description="What the tool does and when to use it")
    parameters: Dict[str, ToolParameter] = Field(
        description="Parameter definitions (JSON Schema format)"
    )
    required: Optional[List[str]] = Field(
        default=None, description="List of required parameter names"
    )


class ToolCall(BaseModel):
    """A tool call requested by the LLM.

    When an LLM decides to call a tool, it returns a ToolCall with the tool name
    and arguments. The client should execute the tool and return a ToolResult.

    Attributes:
        id: Unique identifier for this tool call (used to match with results)
        name: Name of the tool to call (must match a ToolDefinition name)
        arguments: Tool arguments as a dictionary (validated against tool schema)

    Examples:
        >>> tool_call = ToolCall(
        ...     id="call_abc123",
        ...     name="get_weather",
        ...     arguments={"location": "New York", "units": "fahrenheit"}
        ... )

        >>> tool_call = ToolCall(
        ...     id="call_def456",
        ...     name="calculate",
        ...     arguments={"expression": "2 + 2"}
        ... )
    """

    id: str = Field(description="Unique ID for this tool call")
    name: str = Field(description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(description="Tool arguments as JSON object")


class ToolResult(BaseModel):
    """Result from executing a tool.

    After executing a tool, return a ToolResult to the LLM so it can continue
    the conversation with the tool's output.

    Attributes:
        tool_call_id: ID of the tool call this responds to (matches ToolCall.id)
        name: Name of the tool that was called
        result: The tool's return value (can be any JSON-serializable type)
        error: Error message if the tool execution failed (optional)

    Examples:
        Successful tool execution:
        >>> result = ToolResult(
        ...     tool_call_id="call_abc123",
        ...     name="get_weather",
        ...     result={"temperature": 72, "condition": "sunny", "humidity": 45}
        ... )

        Failed tool execution:
        >>> result = ToolResult(
        ...     tool_call_id="call_def456",
        ...     name="database_query",
        ...     result=None,
        ...     error="Database connection timeout after 30 seconds"
        ... )

        Tool returning simple value:
        >>> result = ToolResult(
        ...     tool_call_id="call_ghi789",
        ...     name="calculate",
        ...     result=4
        ... )
    """

    tool_call_id: str = Field(description="ID of the tool call this responds to")
    name: str = Field(description="Name of the tool that was called")
    result: Any = Field(description="Result from the tool execution")
    error: Optional[str] = Field(default=None, description="Error message if tool failed")
