"""Unit tests for the constrained decoding rules (no Qwen inference)."""

from typing import cast
from unittest.mock import Mock

import pytest
from llm_sdk import Small_LLM_Model

from src.constraints import ConstrainedDecoder
from src.models import FunctionModel, ParamType


def _decoder() -> ConstrainedDecoder:
    """Create a decoder without running the model-dependent initializer."""
    return object.__new__(ConstrainedDecoder)


def _boolean_decoder(true_score: float, false_score: float) -> ConstrainedDecoder:
    """Supply controlled logits instead of executing a real LLM."""
    decoder = _decoder()
    model = Mock()
    model.get_logits_from_input_ids.return_value = [
        0.0, true_score, false_score,
    ]
    decoder.model = cast(Small_LLM_Model, model)
    decoder.bool_tokens = [[1], [2]]
    return decoder


def _validation_decoder() -> ConstrainedDecoder:
    """Prepare one function definition for output validation tests."""
    decoder = _decoder()
    decoder.functions = [
        FunctionModel(
            name="set_volume",
            description="Set the volume level",
            parameters={
                "level": ParamType(type="integer"),
                "enabled": ParamType(type="boolean"),
            },
            returns=ParamType(type="boolean"),
        )
    ]
    return decoder


# Integer prefix rules: 6 cases
@pytest.mark.parametrize(
    ("prefix", "token", "expected"),
    [
        ("12", "3", True),
        ("", "-", True),
        ("-", "8", True),
        ("0", "5", False),
        ("12", "a", False),
        ("12", ".5", False),
    ],
)
def test_integer_prefix(prefix: str, token: str, expected: bool) -> None:
    decoder = _decoder()
    assert decoder.is_valid_integer_prefix(prefix, token) is expected


# Decimal prefix rules: 6 cases
@pytest.mark.parametrize(
    ("prefix", "token", "expected"),
    [
        ("12", ".5", True),
        ("-", "2.5", True),
        ("1.", "2", True),
        ("", ".5", False),
        ("1.2", ".3", False),
        ("0", "5", False),
    ],
)
def test_number_prefix(prefix: str, token: str, expected: bool) -> None:
    decoder = _decoder()
    assert decoder.is_valid_number_prefix(prefix, token) is expected


# JSON string state machine: 8 cases
@pytest.mark.parametrize(
    ("state", "token", "expected"),
    [
        ("NORMAL", "Hello", (True, "NORMAL")),
        ("NORMAL", "\\", (True, "ESCAPE")),
        ("ESCAPE", "n", (True, "NORMAL")),
        ("ESCAPE", "x", (False, "ESCAPE")),
        ("NORMAL", '"', (True, "CLOSED")),
        ("NORMAL", '"x', (False, "CLOSED")),
        ("NORMAL", "\\u0041", (True, "NORMAL")),
        ("NORMAL", "\n", (False, "NORMAL")),
    ],
)
def test_string_token(
    state: str,
    token: str,
    expected: tuple[bool, str],
) -> None:
    decoder = _decoder()
    assert decoder.check_str_token(state, token) == expected


# Function selection: 3 cases
def test_matching_function_prefix() -> None:
    decoder = _decoder()
    decoder.function_tokens = [[10, 20], [10, 30], [40]]

    assert decoder.get_matching_functions([10]) == [[10, 20], [10, 30]]


def test_no_matching_function_prefix() -> None:
    decoder = _decoder()
    decoder.function_tokens = [[10, 20], [10, 30]]

    assert decoder.get_matching_functions([99]) == []


def test_single_function_name_is_forced() -> None:
    decoder = _decoder()
    decoder.function_tokens = [[11, 12, 13]]
    decoder.demo = False
    model = Mock()
    model.decode.return_value = 'write_note_"'
    decoder.model = cast(Small_LLM_Model, model)
    input_ids = [99]

    result = decoder.generate_function_name(input_ids)

    assert result == [11, 12, 13]
    assert input_ids == [99, 11, 12, 13]
    model.get_logits_from_input_ids.assert_not_called()


# Parameter dispatch: 3 cases
def test_boolean_parameter_true() -> None:
    decoder = _boolean_decoder(true_score=10.0, false_score=1.0)
    input_ids: list[int] = []

    assert decoder.generate_param_value(input_ids, "boolean", "}") is True
    assert input_ids == [1]


def test_boolean_parameter_false() -> None:
    decoder = _boolean_decoder(true_score=1.0, false_score=10.0)
    input_ids: list[int] = []

    assert decoder.generate_param_value(input_ids, "boolean", "}") is False
    assert input_ids == [2]


def test_unsupported_parameter_type() -> None:
    decoder = _decoder()

    with pytest.raises(ValueError, match="Unsupported parameter type"):
        decoder.generate_param_value([], "array", "}")


# Final output validation: 4 cases
def test_valid_output() -> None:
    decoder = _validation_decoder()
    result = {
        "prompt": "Set volume to three",
        "name": "set_volume",
        "parameters": {"level": 3, "enabled": True},
    }

    assert decoder.validate_output(result) is None


def test_output_with_extra_field() -> None:
    decoder = _validation_decoder()
    result = {
        "prompt": "Set volume to three",
        "name": "set_volume",
        "parameters": {"level": 3, "enabled": True},
        "extra": "not allowed",
    }

    with pytest.raises(ValueError, match="Output must contain"):
        decoder.validate_output(result)


def test_output_with_unknown_function() -> None:
    decoder = _validation_decoder()
    result = {
        "prompt": "Do something else",
        "name": "unknown_function",
        "parameters": {"level": 3, "enabled": True},
    }

    with pytest.raises(ValueError, match="Unknown function"):
        decoder.validate_output(result)


def test_output_with_wrong_parameter_type() -> None:
    decoder = _validation_decoder()
    result = {
        "prompt": "Set volume to three",
        "name": "set_volume",
        "parameters": {"level": True, "enabled": True},
    }

    with pytest.raises(ValueError, match="Incorrect type"):
        decoder.validate_output(result)
