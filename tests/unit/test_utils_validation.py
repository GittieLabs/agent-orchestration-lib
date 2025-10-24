"""Comprehensive unit tests for validation utilities."""

import pytest
from pydantic import BaseModel, ValidationError
from agent_lib.utils import (
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


class TestValidatePydantic:
    """Test validate_pydantic function."""

    def test_validate_valid_data(self):
        """Test validation with valid data."""
        class User(BaseModel):
            name: str
            age: int

        is_valid, data, errors = validate_pydantic(
            {"name": "Alice", "age": 30},
            User,
            strict=False
        )
        assert is_valid is True
        assert data == {"name": "Alice", "age": 30}
        assert errors is None

    def test_validate_with_coercion(self):
        """Test validation with type coercion."""
        class User(BaseModel):
            name: str
            age: int

        is_valid, data, errors = validate_pydantic(
            {"name": "Alice", "age": "30"},
            User,
            strict=False
        )
        assert is_valid is True
        assert data == {"name": "Alice", "age": 30}

    def test_validate_invalid_data(self):
        """Test validation with invalid data."""
        class User(BaseModel):
            name: str
            age: int

        is_valid, data, errors = validate_pydantic(
            {"name": "Alice", "age": "invalid"},
            User,
            strict=False
        )
        assert is_valid is False
        assert data is None
        assert errors is not None
        assert len(errors) > 0

    def test_validate_strict_mode(self):
        """Test strict mode raises exception."""
        class User(BaseModel):
            name: str
            age: int

        with pytest.raises(ValidationError):
            validate_pydantic(
                {"name": "Alice", "age": "invalid"},
                User,
                strict=True
            )


class TestCoerceTypes:
    """Test coerce_types function."""

    def test_coerce_int(self):
        """Test coercing to int."""
        data = {"age": "30", "score": "95"}
        type_map = {"age": int, "score": int}
        result = coerce_types(data, type_map)
        assert result == {"age": 30, "score": 95}

    def test_coerce_float(self):
        """Test coercing to float."""
        data = {"price": "19.99"}
        type_map = {"price": float}
        result = coerce_types(data, type_map)
        assert result == {"price": 19.99}

    def test_coerce_bool(self):
        """Test coercing to bool."""
        data = {
            "a": "true", "b": "false", "c": "1",
            "d": "0", "e": "yes", "f": "no"
        }
        type_map = {k: bool for k in data.keys()}
        result = coerce_types(data, type_map)
        assert result["a"] is True
        assert result["b"] is False
        assert result["c"] is True
        assert result["d"] is False
        assert result["e"] is True
        assert result["f"] is False

    def test_coerce_missing_field(self):
        """Test coercion skips missing fields."""
        data = {"age": "30"}
        type_map = {"age": int, "score": int}
        result = coerce_types(data, type_map)
        assert result == {"age": 30}

    def test_coerce_failure_non_strict(self):
        """Test coercion failure in non-strict mode."""
        data = {"age": "invalid"}
        type_map = {"age": int}
        result = coerce_types(data, type_map, strict=False)
        assert result == {"age": "invalid"}  # Unchanged

    def test_coerce_failure_strict(self):
        """Test coercion failure in strict mode."""
        data = {"age": "invalid"}
        type_map = {"age": int}
        with pytest.raises(ValueError):
            coerce_types(data, type_map, strict=True)


class TestSanitizeString:
    """Test sanitize_string function."""

    def test_sanitize_strip(self):
        """Test stripping whitespace."""
        result = sanitize_string("  hello  ")
        assert result == "hello"

    def test_sanitize_max_length(self):
        """Test maximum length truncation."""
        result = sanitize_string("hello world", max_length=5)
        assert result == "hello"

    def test_sanitize_lowercase(self):
        """Test lowercase conversion."""
        result = sanitize_string("HELLO", lowercase=True)
        assert result == "hello"

    def test_sanitize_allowed_chars(self):
        """Test filtering allowed characters."""
        result = sanitize_string(
            "Hello@World!",
            allowed_chars=r"[^a-zA-Z ]"
        )
        assert result == "HelloWorld"

    def test_sanitize_combined(self):
        """Test combined sanitization."""
        result = sanitize_string(
            "  Hello@World!  ",
            max_length=10,
            allowed_chars=r"[^a-zA-Z]",
            strip=True,
            lowercase=True
        )
        assert result == "helloworld"


class TestSanitizeDict:
    """Test sanitize_dict function."""

    def test_sanitize_strings(self):
        """Test sanitizing string fields."""
        data = {"name": "  Alice  ", "bio": "A" * 200}
        result = sanitize_dict(
            data,
            string_fields=["name", "bio"],
            max_length=100
        )
        assert result["name"] == "Alice"
        assert len(result["bio"]) == 100

    def test_remove_null(self):
        """Test removing null values."""
        data = {"a": 1, "b": None, "c": 3}
        result = sanitize_dict(data, remove_null=True)
        assert result == {"a": 1, "c": 3}

    def test_remove_empty(self):
        """Test removing empty values."""
        data = {"a": 1, "b": "", "c": [], "d": 0, "e": False}
        result = sanitize_dict(data, remove_empty=True)
        assert result == {"a": 1, "d": 0, "e": False}


class TestValidateRequiredFields:
    """Test validate_required_fields function."""

    def test_all_required_present(self):
        """Test all required fields present."""
        data = {"name": "Alice", "email": "alice@example.com"}
        is_valid, missing = validate_required_fields(data, ["name", "email"])
        assert is_valid is True
        assert missing == []

    def test_missing_required_fields(self):
        """Test missing required fields."""
        data = {"name": "Alice"}
        is_valid, missing = validate_required_fields(
            data,
            ["name", "email", "age"]
        )
        assert is_valid is False
        assert set(missing) == {"email", "age"}

    def test_empty_value_counts_as_missing(self):
        """Test that empty values count as missing."""
        data = {"name": "Alice", "email": ""}
        is_valid, missing = validate_required_fields(data, ["name", "email"])
        assert is_valid is False
        assert missing == ["email"]


class TestValidateFieldTypes:
    """Test validate_field_types function."""

    def test_valid_types(self):
        """Test validation with correct types."""
        data = {"name": "Alice", "age": 30}
        type_map = {"name": str, "age": int}
        is_valid, errors = validate_field_types(data, type_map)
        assert is_valid is True
        assert errors == {}

    def test_invalid_types(self):
        """Test validation with incorrect types."""
        data = {"name": "Alice", "age": "30"}
        type_map = {"name": str, "age": int}
        is_valid, errors = validate_field_types(data, type_map)
        assert is_valid is False
        assert "age" in errors

    def test_multiple_allowed_types(self):
        """Test validation with multiple allowed types."""
        data = {"value": 30}
        type_map = {"value": (int, str)}
        is_valid, errors = validate_field_types(data, type_map)
        assert is_valid is True

    def test_skip_missing_fields(self):
        """Test that missing fields are skipped."""
        data = {"name": "Alice"}
        type_map = {"name": str, "age": int}
        is_valid, errors = validate_field_types(data, type_map)
        assert is_valid is True


class TestValidateStringPattern:
    """Test validate_string_pattern function."""

    def test_valid_pattern(self):
        """Test string matching pattern."""
        is_valid, error = validate_string_pattern(
            "alice@example.com",
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "email"
        )
        assert is_valid is True
        assert error is None

    def test_invalid_pattern(self):
        """Test string not matching pattern."""
        is_valid, error = validate_string_pattern(
            "not-an-email",
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "email"
        )
        assert is_valid is False
        assert "email" in error


class TestValidateRange:
    """Test validate_range function."""

    def test_value_in_range(self):
        """Test value within range."""
        is_valid, error = validate_range(5, min_value=0, max_value=10)
        assert is_valid is True
        assert error is None

    def test_value_below_min(self):
        """Test value below minimum."""
        is_valid, error = validate_range(-1, min_value=0)
        assert is_valid is False
        assert "minimum" in error

    def test_value_above_max(self):
        """Test value above maximum."""
        is_valid, error = validate_range(15, max_value=10)
        assert is_valid is False
        assert "maximum" in error

    def test_value_at_boundary_inclusive(self):
        """Test value at boundary with inclusive range."""
        assert validate_range(0, min_value=0, max_value=10, inclusive=True)[0]
        assert validate_range(10, min_value=0, max_value=10, inclusive=True)[0]

    def test_value_at_boundary_exclusive(self):
        """Test value at boundary with exclusive range."""
        assert not validate_range(0, min_value=0, inclusive=False)[0]
        assert not validate_range(10, max_value=10, inclusive=False)[0]


class TestValidateListLength:
    """Test validate_list_length function."""

    def test_length_in_range(self):
        """Test list length within range."""
        is_valid, error = validate_list_length([1, 2, 3], min_length=1, max_length=5)
        assert is_valid is True
        assert error is None

    def test_length_below_min(self):
        """Test list length below minimum."""
        is_valid, error = validate_list_length([], min_length=1)
        assert is_valid is False
        assert "minimum" in error

    def test_length_above_max(self):
        """Test list length above maximum."""
        is_valid, error = validate_list_length([1, 2, 3, 4, 5, 6], max_length=5)
        assert is_valid is False
        assert "maximum" in error

    def test_no_constraints(self):
        """Test validation with no constraints."""
        is_valid, error = validate_list_length([1, 2, 3])
        assert is_valid is True
