"""Comprehensive unit tests for data transformation utilities."""

import pytest
from agent_lib.utils import (
    flatten_dict,
    unflatten_dict,
    merge_deep,
    extract_fields,
    transform_keys,
    pick,
    omit,
    map_values,
)


class TestFlattenDict:
    """Test flatten_dict function."""

    def test_flatten_simple_nested(self):
        """Test flattening a simple nested dictionary."""
        nested = {"a": {"b": {"c": 1}}}
        result = flatten_dict(nested)
        assert result == {"a.b.c": 1}

    def test_flatten_mixed_depth(self):
        """Test flattening mixed depth dictionary."""
        nested = {
            "user": {"name": "Alice", "address": {"city": "NYC", "zip": "10001"}},
            "active": True
        }
        result = flatten_dict(nested)
        assert result == {
            "user.name": "Alice",
            "user.address.city": "NYC",
            "user.address.zip": "10001",
            "active": True
        }

    def test_flatten_custom_separator(self):
        """Test flattening with custom separator."""
        nested = {"a": {"b": 1}}
        result = flatten_dict(nested, separator="/")
        assert result == {"a/b": 1}

    def test_flatten_empty_dict(self):
        """Test flattening empty dictionary."""
        result = flatten_dict({})
        assert result == {}

    def test_flatten_flat_dict(self):
        """Test flattening already flat dictionary."""
        flat = {"a": 1, "b": 2}
        result = flatten_dict(flat)
        assert result == flat

    def test_flatten_with_list_values(self):
        """Test that list values are not flattened."""
        nested = {"a": {"b": [1, 2, 3]}}
        result = flatten_dict(nested)
        assert result == {"a.b": [1, 2, 3]}


class TestUnflattenDict:
    """Test unflatten_dict function."""

    def test_unflatten_simple(self):
        """Test unflattening simple flat dictionary."""
        flat = {"a.b.c": 1}
        result = unflatten_dict(flat)
        assert result == {"a": {"b": {"c": 1}}}

    def test_unflatten_mixed(self):
        """Test unflattening mixed flat dictionary."""
        flat = {
            "user.name": "Alice",
            "user.address.city": "NYC",
            "active": True
        }
        result = unflatten_dict(flat)
        expected = {
            "user": {"name": "Alice", "address": {"city": "NYC"}},
            "active": True
        }
        assert result == expected

    def test_unflatten_custom_separator(self):
        """Test unflattening with custom separator."""
        flat = {"a/b": 1}
        result = unflatten_dict(flat, separator="/")
        assert result == {"a": {"b": 1}}

    def test_unflatten_empty(self):
        """Test unflattening empty dictionary."""
        result = unflatten_dict({})
        assert result == {}

    def test_flatten_unflatten_roundtrip(self):
        """Test that flatten and unflatten are inverse operations."""
        original = {
            "user": {"name": "Alice", "age": 30},
            "settings": {"theme": "dark"}
        }
        flattened = flatten_dict(original)
        unflattened = unflatten_dict(flattened)
        assert unflattened == original


class TestMergeDeep:
    """Test merge_deep function."""

    def test_merge_simple(self):
        """Test merging simple dictionaries."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"c": 3}
        result = merge_deep(dict1, dict2)
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_merge_nested(self):
        """Test merging nested dictionaries."""
        dict1 = {"a": 1, "b": {"c": 2, "d": 3}}
        dict2 = {"b": {"c": 4, "e": 5}, "f": 6}
        result = merge_deep(dict1, dict2)
        assert result == {"a": 1, "b": {"c": 4, "d": 3, "e": 5}, "f": 6}

    def test_merge_no_overwrite(self):
        """Test merging without overwriting."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 3, "c": 4}
        result = merge_deep(dict1, dict2, overwrite=False)
        assert result == {"a": 1, "b": 2, "c": 4}

    def test_merge_empty(self):
        """Test merging with empty dictionaries."""
        dict1 = {"a": 1}
        dict2 = {}
        assert merge_deep(dict1, dict2) == {"a": 1}
        assert merge_deep({}, dict1) == {"a": 1}

    def test_merge_preserves_originals(self):
        """Test that merge doesn't modify original dictionaries."""
        dict1 = {"a": 1}
        dict2 = {"b": 2}
        result = merge_deep(dict1, dict2)
        assert dict1 == {"a": 1}
        assert dict2 == {"b": 2}
        assert result == {"a": 1, "b": 2}


class TestExtractFields:
    """Test extract_fields function."""

    def test_extract_simple_fields(self):
        """Test extracting simple fields."""
        data = {"a": 1, "b": 2, "c": 3}
        result = extract_fields(data, ["a", "c"])
        assert result == {"a": 1, "c": 3}

    def test_extract_nested_fields(self):
        """Test extracting nested fields."""
        data = {
            "user": {"name": "Alice", "age": 30},
            "status": "active"
        }
        result = extract_fields(data, ["user.name", "status"])
        assert result == {"user": {"name": "Alice"}, "status": "active"}

    def test_extract_with_flatten(self):
        """Test extracting fields with flatten option."""
        data = {"user": {"name": "Alice", "age": 30}}
        result = extract_fields(data, ["user.name"], flatten=True)
        assert result == {"user.name": "Alice"}

    def test_extract_missing_fields(self):
        """Test that missing fields are skipped."""
        data = {"a": 1, "b": 2}
        result = extract_fields(data, ["a", "c"])
        assert result == {"a": 1}

    def test_extract_empty_list(self):
        """Test extracting with empty field list."""
        data = {"a": 1, "b": 2}
        result = extract_fields(data, [])
        assert result == {}


class TestTransformKeys:
    """Test transform_keys function."""

    def test_transform_to_upper(self):
        """Test transforming keys to uppercase."""
        data = {"name": "Alice", "age": 30}
        result = transform_keys(data, str.upper)
        assert result == {"NAME": "Alice", "AGE": 30}

    def test_transform_recursive(self):
        """Test recursive key transformation."""
        data = {"user": {"name": "Alice"}, "status": "active"}
        result = transform_keys(data, str.upper, recursive=True)
        assert result == {"USER": {"NAME": "Alice"}, "STATUS": "active"}

    def test_transform_custom_function(self):
        """Test transformation with custom function."""
        def add_prefix(key):
            return f"prefix_{key}"

        data = {"a": 1, "b": 2}
        result = transform_keys(data, add_prefix)
        assert result == {"prefix_a": 1, "prefix_b": 2}


class TestPickAndOmit:
    """Test pick and omit functions."""

    def test_pick_existing_keys(self):
        """Test picking existing keys."""
        data = {"a": 1, "b": 2, "c": 3}
        result = pick(data, ["a", "c"])
        assert result == {"a": 1, "c": 3}

    def test_pick_missing_keys(self):
        """Test picking with some missing keys."""
        data = {"a": 1, "b": 2}
        result = pick(data, ["a", "c"])
        assert result == {"a": 1}

    def test_omit_keys(self):
        """Test omitting keys."""
        data = {"a": 1, "b": 2, "c": 3}
        result = omit(data, ["b"])
        assert result == {"a": 1, "c": 3}

    def test_omit_missing_keys(self):
        """Test omitting keys that don't exist."""
        data = {"a": 1, "b": 2}
        result = omit(data, ["c", "d"])
        assert result == {"a": 1, "b": 2}


class TestMapValues:
    """Test map_values function."""

    def test_map_simple(self):
        """Test mapping values with simple function."""
        data = {"a": 1, "b": 2, "c": 3}
        result = map_values(data, lambda x: x * 2)
        assert result == {"a": 2, "b": 4, "c": 6}

    def test_map_recursive(self):
        """Test recursive value mapping."""
        data = {"a": 1, "nested": {"b": 2}}
        result = map_values(data, lambda x: x * 2, recursive=True)
        assert result == {"a": 2, "nested": {"b": 4}}

    def test_map_type_conversion(self):
        """Test mapping for type conversion."""
        data = {"a": "1", "b": "2"}
        result = map_values(data, int)
        assert result == {"a": 1, "b": 2}
