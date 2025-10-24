"""Utility functions and helpers."""

from .data_transform import (
    flatten_dict,
    unflatten_dict,
    merge_deep,
    extract_fields,
    transform_keys,
    pick,
    omit,
    map_values,
)

from .validation import (
    validate_pydantic,
    coerce_types,
    sanitize_string,
    sanitize_dict,
    validate_required_fields,
    validate_field_types,
    validate_string_pattern,
    validate_range,
    validate_list_length,
)

__all__ = [
    # Data transformation
    "flatten_dict",
    "unflatten_dict",
    "merge_deep",
    "extract_fields",
    "transform_keys",
    "pick",
    "omit",
    "map_values",
    # Validation
    "validate_pydantic",
    "coerce_types",
    "sanitize_string",
    "sanitize_dict",
    "validate_required_fields",
    "validate_field_types",
    "validate_string_pattern",
    "validate_range",
    "validate_list_length",
]
