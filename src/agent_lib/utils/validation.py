"""Validation utilities for agent-orchestration-lib.

This module provides utility functions for validating and sanitizing data
used in agent workflows.
"""

from typing import Dict, Any, Optional, List, Union, Type
import re
from pydantic import BaseModel, ValidationError


def validate_pydantic(
    data: Dict[str, Any],
    model: Type[BaseModel],
    strict: bool = True
) -> tuple[bool, Optional[Dict[str, Any]], Optional[List[str]]]:
    """Validate data against a Pydantic model.

    Args:
        data: Data to validate
        model: Pydantic model class
        strict: If True, validation errors raise exception (default: True)

    Returns:
        Tuple of (is_valid, validated_data, errors)
        - is_valid: True if validation passed
        - validated_data: Validated and coerced data (None if invalid)
        - errors: List of error messages (None if valid)

    Example:
        ```python
        from pydantic import BaseModel

        class User(BaseModel):
            name: str
            age: int

        is_valid, data, errors = validate_pydantic(
            {"name": "Alice", "age": "30"},
            User
        )
        # is_valid=True, data={"name": "Alice", "age": 30}, errors=None
        ```
    """
    try:
        validated = model(**data)
        return True, validated.model_dump(), None
    except ValidationError as e:
        if strict:
            raise
        errors = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
        return False, None, errors


def coerce_types(
    data: Dict[str, Any],
    type_map: Dict[str, Type],
    strict: bool = False
) -> Dict[str, Any]:
    """Coerce dictionary values to specified types.

    Args:
        data: Data to coerce
        type_map: Mapping of field names to target types
        strict: If True, raise error on coercion failure (default: False)

    Returns:
        Dictionary with coerced values

    Example:
        ```python
        data = {"age": "30", "score": "95.5", "active": "true"}
        type_map = {"age": int, "score": float, "active": bool}
        result = coerce_types(data, type_map)
        # Result: {"age": 30, "score": 95.5, "active": True}
        ```
    """
    result = data.copy()

    for field, target_type in type_map.items():
        if field not in result:
            continue

        value = result[field]

        try:
            # Handle special cases for boolean
            if target_type == bool:
                if isinstance(value, str):
                    result[field] = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    result[field] = bool(value)
            else:
                result[field] = target_type(value)
        except (ValueError, TypeError) as e:
            if strict:
                raise ValueError(
                    f"Cannot coerce field '{field}' to {target_type.__name__}: {e}"
                )
            # Leave value unchanged if coercion fails

    return result


def sanitize_string(
    text: str,
    max_length: Optional[int] = None,
    allowed_chars: Optional[str] = None,
    strip: bool = True,
    lowercase: bool = False
) -> str:
    """Sanitize a string value.

    Args:
        text: String to sanitize
        max_length: Maximum length (truncate if longer)
        allowed_chars: Regex pattern of allowed characters
        strip: Remove leading/trailing whitespace (default: True)
        lowercase: Convert to lowercase (default: False)

    Returns:
        Sanitized string

    Example:
        ```python
        text = "  Hello, World!  "
        result = sanitize_string(text, max_length=10, strip=True)
        # Result: "Hello, Wor"

        # Allow only alphanumeric and spaces:
        result = sanitize_string(
            "Hello@World!",
            allowed_chars=r"[^a-zA-Z0-9 ]"
        )
        # Result: "HelloWorld"
        ```
    """
    if strip:
        text = text.strip()

    if lowercase:
        text = text.lower()

    if allowed_chars:
        text = re.sub(allowed_chars, '', text)

    if max_length and len(text) > max_length:
        text = text[:max_length]

    return text


def sanitize_dict(
    data: Dict[str, Any],
    string_fields: Optional[List[str]] = None,
    max_length: int = 1000,
    remove_null: bool = False,
    remove_empty: bool = False
) -> Dict[str, Any]:
    """Sanitize dictionary values.

    Args:
        data: Dictionary to sanitize
        string_fields: List of field names to sanitize as strings
        max_length: Maximum length for string fields
        remove_null: Remove keys with None values (default: False)
        remove_empty: Remove keys with empty values (default: False)

    Returns:
        Sanitized dictionary

    Example:
        ```python
        data = {
            "name": "  Alice  ",
            "bio": "A" * 2000,
            "age": 30,
            "email": None,
            "tags": []
        }
        result = sanitize_dict(
            data,
            string_fields=["name", "bio"],
            max_length=100,
            remove_null=True,
            remove_empty=True
        )
        # Result: {"name": "Alice", "bio": "A"*100, "age": 30}
        ```
    """
    result = {}

    for key, value in data.items():
        # Skip null values if requested
        if remove_null and value is None:
            continue

        # Skip empty values if requested
        if remove_empty and not value and value != 0 and value is not False:
            continue

        # Sanitize string fields
        if string_fields and key in string_fields and isinstance(value, str):
            value = sanitize_string(value, max_length=max_length)

        result[key] = value

    return result


def validate_required_fields(
    data: Dict[str, Any],
    required: List[str]
) -> tuple[bool, List[str]]:
    """Validate that required fields are present and non-empty.

    Args:
        data: Data to validate
        required: List of required field names

    Returns:
        Tuple of (is_valid, missing_fields)

    Example:
        ```python
        data = {"name": "Alice", "email": ""}
        is_valid, missing = validate_required_fields(
            data,
            ["name", "email", "age"]
        )
        # is_valid=False, missing=["email", "age"]
        ```
    """
    missing = []

    for field in required:
        if field not in data or not data[field]:
            missing.append(field)

    return len(missing) == 0, missing


def validate_field_types(
    data: Dict[str, Any],
    type_map: Dict[str, Union[Type, tuple]]
) -> tuple[bool, Dict[str, str]]:
    """Validate field types.

    Args:
        data: Data to validate
        type_map: Mapping of field names to expected types

    Returns:
        Tuple of (is_valid, type_errors)
        type_errors is a dict mapping field names to error messages

    Example:
        ```python
        data = {"name": "Alice", "age": "30"}
        type_map = {"name": str, "age": int}
        is_valid, errors = validate_field_types(data, type_map)
        # is_valid=False, errors={"age": "Expected int, got str"}
        ```
    """
    errors = {}

    for field, expected_type in type_map.items():
        if field not in data:
            continue

        value = data[field]

        if not isinstance(value, expected_type):
            if isinstance(expected_type, tuple):
                type_names = " or ".join(t.__name__ for t in expected_type)
                actual_type = type(value).__name__
                errors[field] = f"Expected {type_names}, got {actual_type}"
            else:
                errors[field] = f"Expected {expected_type.__name__}, got {type(value).__name__}"

    return len(errors) == 0, errors


def validate_string_pattern(
    text: str,
    pattern: str,
    pattern_name: str = "pattern"
) -> tuple[bool, Optional[str]]:
    r"""Validate string matches a regex pattern.

    Args:
        text: String to validate
        pattern: Regex pattern
        pattern_name: Name of pattern for error message

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        ```python
        is_valid, error = validate_string_pattern(
            "alice@example.com",
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "email"
        )
        # is_valid=True, error=None
        ```
    """
    if re.match(pattern, text):
        return True, None
    return False, f"String does not match {pattern_name} pattern"


def validate_range(
    value: Union[int, float],
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    inclusive: bool = True
) -> tuple[bool, Optional[str]]:
    """Validate numeric value is within range.

    Args:
        value: Value to validate
        min_value: Minimum allowed value (None for no minimum)
        max_value: Maximum allowed value (None for no maximum)
        inclusive: Include min/max in valid range (default: True)

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        ```python
        is_valid, error = validate_range(5, min_value=0, max_value=10)
        # is_valid=True, error=None

        is_valid, error = validate_range(15, min_value=0, max_value=10)
        # is_valid=False, error="Value 15 exceeds maximum 10"
        ```
    """
    if min_value is not None:
        if inclusive and value < min_value:
            return False, f"Value {value} is less than minimum {min_value}"
        elif not inclusive and value <= min_value:
            return False, f"Value {value} must be greater than {min_value}"

    if max_value is not None:
        if inclusive and value > max_value:
            return False, f"Value {value} exceeds maximum {max_value}"
        elif not inclusive and value >= max_value:
            return False, f"Value {value} must be less than {max_value}"

    return True, None


def validate_list_length(
    items: List[Any],
    min_length: Optional[int] = None,
    max_length: Optional[int] = None
) -> tuple[bool, Optional[str]]:
    """Validate list length is within bounds.

    Args:
        items: List to validate
        min_length: Minimum allowed length (None for no minimum)
        max_length: Maximum allowed length (None for no maximum)

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        ```python
        is_valid, error = validate_list_length([1, 2, 3], min_length=1, max_length=5)
        # is_valid=True, error=None
        ```
    """
    length = len(items)

    if min_length is not None and length < min_length:
        return False, f"List length {length} is less than minimum {min_length}"

    if max_length is not None and length > max_length:
        return False, f"List length {length} exceeds maximum {max_length}"

    return True, None
