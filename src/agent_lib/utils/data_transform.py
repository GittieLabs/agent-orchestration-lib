"""Data transformation utilities for agent-orchestration-lib.

This module provides utility functions for transforming and manipulating data
structures commonly used in agent workflows.
"""

from typing import Dict, Any, List, Optional, Union
from collections.abc import Mapping


def flatten_dict(
    nested: Dict[str, Any],
    separator: str = ".",
    parent_key: str = ""
) -> Dict[str, Any]:
    """Flatten a nested dictionary to a single level.

    Converts nested dictionary structures into a flat dictionary with keys
    representing the path to each value.

    Args:
        nested: Nested dictionary to flatten
        separator: String to use for joining keys (default: ".")
        parent_key: Internal parameter for recursion (do not use)

    Returns:
        Flattened dictionary with composite keys

    Example:
        ```python
        nested = {
            "user": {
                "name": "Alice",
                "address": {"city": "NYC", "zip": "10001"}
            },
            "active": True
        }
        flat = flatten_dict(nested)
        # Result: {
        #     "user.name": "Alice",
        #     "user.address.city": "NYC",
        #     "user.address.zip": "10001",
        #     "active": True
        # }
        ```
    """
    items: List[tuple] = []

    for key, value in nested.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key

        if isinstance(value, Mapping) and value:
            # Recursively flatten nested dictionaries
            items.extend(flatten_dict(value, separator, new_key).items())
        else:
            items.append((new_key, value))

    return dict(items)


def unflatten_dict(
    flat: Dict[str, Any],
    separator: str = "."
) -> Dict[str, Any]:
    """Unflatten a dictionary with composite keys into nested structure.

    Converts a flat dictionary with composite keys back into a nested
    dictionary structure.

    Args:
        flat: Flat dictionary with composite keys
        separator: String used to separate key components (default: ".")

    Returns:
        Nested dictionary

    Example:
        ```python
        flat = {
            "user.name": "Alice",
            "user.address.city": "NYC",
            "user.address.zip": "10001",
            "active": True
        }
        nested = unflatten_dict(flat)
        # Result: {
        #     "user": {
        #         "name": "Alice",
        #         "address": {"city": "NYC", "zip": "10001"}
        #     },
        #     "active": True
        # }
        ```
    """
    result: Dict[str, Any] = {}

    for key, value in flat.items():
        parts = key.split(separator)
        current = result

        # Navigate/create nested structure
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the final value
        current[parts[-1]] = value

    return result


def merge_deep(
    dict1: Dict[str, Any],
    dict2: Dict[str, Any],
    overwrite: bool = True
) -> Dict[str, Any]:
    """Deep merge two dictionaries.

    Recursively merges dict2 into dict1, combining nested dictionaries
    rather than replacing them.

    Args:
        dict1: Base dictionary
        dict2: Dictionary to merge into dict1
        overwrite: If True, dict2 values overwrite dict1 values.
                   If False, dict1 values are preserved. (default: True)

    Returns:
        New merged dictionary

    Example:
        ```python
        dict1 = {"a": 1, "b": {"c": 2, "d": 3}}
        dict2 = {"b": {"c": 4, "e": 5}, "f": 6}
        merged = merge_deep(dict1, dict2)
        # Result: {"a": 1, "b": {"c": 4, "d": 3, "e": 5}, "f": 6}
        ```
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_deep(result[key], value, overwrite)
        elif overwrite or key not in result:
            # Overwrite or add new key
            result[key] = value

    return result


def extract_fields(
    data: Dict[str, Any],
    fields: List[str],
    flatten: bool = False,
    separator: str = "."
) -> Dict[str, Any]:
    """Extract specific fields from a dictionary.

    Supports dot notation for nested field access.

    Args:
        data: Source dictionary
        fields: List of field names to extract (supports dot notation)
        flatten: If True, flatten nested results (default: False)
        separator: Separator for dot notation (default: ".")

    Returns:
        Dictionary containing only the specified fields

    Example:
        ```python
        data = {
            "user": {"name": "Alice", "age": 30},
            "status": "active",
            "metadata": {"created": "2024-01-01"}
        }
        result = extract_fields(data, ["user.name", "status"])
        # Result: {"user": {"name": "Alice"}, "status": "active"}

        # With flatten=True:
        result = extract_fields(data, ["user.name", "status"], flatten=True)
        # Result: {"user.name": "Alice", "status": "active"}
        ```
    """
    result: Dict[str, Any] = {}

    for field in fields:
        parts = field.split(separator)
        current = data

        # Navigate to the field
        try:
            for part in parts:
                current = current[part]

            # Store the extracted value
            if flatten:
                result[field] = current
            else:
                # Reconstruct nested structure
                target = result
                for part in parts[:-1]:
                    if part not in target:
                        target[part] = {}
                    target = target[part]
                target[parts[-1]] = current

        except (KeyError, TypeError):
            # Field doesn't exist, skip it
            continue

    return result


def transform_keys(
    data: Dict[str, Any],
    transform_fn: callable,
    recursive: bool = False
) -> Dict[str, Any]:
    """Transform dictionary keys using a custom function.

    Args:
        data: Dictionary to transform
        transform_fn: Function to apply to each key
        recursive: If True, apply transformation to nested dictionaries

    Returns:
        Dictionary with transformed keys

    Example:
        ```python
        data = {"first_name": "Alice", "last_name": "Smith"}
        result = transform_keys(data, str.upper)
        # Result: {"FIRST_NAME": "Alice", "LAST_NAME": "Smith"}

        # Convert snake_case to camelCase:
        def to_camel(key):
            parts = key.split('_')
            return parts[0] + ''.join(p.capitalize() for p in parts[1:])

        result = transform_keys(data, to_camel)
        # Result: {"firstName": "Alice", "lastName": "Smith"}
        ```
    """
    result: Dict[str, Any] = {}

    for key, value in data.items():
        new_key = transform_fn(key)

        if recursive and isinstance(value, dict):
            result[new_key] = transform_keys(value, transform_fn, recursive)
        else:
            result[new_key] = value

    return result


def pick(data: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """Pick specific keys from a dictionary.

    Similar to extract_fields but only works with top-level keys.

    Args:
        data: Source dictionary
        keys: List of keys to pick

    Returns:
        Dictionary containing only the specified keys

    Example:
        ```python
        data = {"a": 1, "b": 2, "c": 3}
        result = pick(data, ["a", "c"])
        # Result: {"a": 1, "c": 3}
        ```
    """
    return {key: data[key] for key in keys if key in data}


def omit(data: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """Omit specific keys from a dictionary.

    Args:
        data: Source dictionary
        keys: List of keys to omit

    Returns:
        Dictionary without the specified keys

    Example:
        ```python
        data = {"a": 1, "b": 2, "c": 3}
        result = omit(data, ["b"])
        # Result: {"a": 1, "c": 3}
        ```
    """
    return {key: value for key, value in data.items() if key not in keys}


def map_values(
    data: Dict[str, Any],
    map_fn: callable,
    recursive: bool = False
) -> Dict[str, Any]:
    """Transform dictionary values using a custom function.

    Args:
        data: Dictionary to transform
        map_fn: Function to apply to each value
        recursive: If True, apply transformation to nested dictionaries

    Returns:
        Dictionary with transformed values

    Example:
        ```python
        data = {"a": 1, "b": 2, "c": 3}
        result = map_values(data, lambda x: x * 2)
        # Result: {"a": 2, "b": 4, "c": 6}
        ```
    """
    result: Dict[str, Any] = {}

    for key, value in data.items():
        if recursive and isinstance(value, dict):
            result[key] = map_values(value, map_fn, recursive)
        else:
            result[key] = map_fn(value)

    return result
