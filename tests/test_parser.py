from src.json_parser import load_prompts, load_functions
import pytest
from pathlib import Path


def test_valid_prompts(tmp_path) -> None:
    file = tmp_path / "prompts.json"
    file.write_text('[{"prompt": "Hello"}]')
    prompts = load_prompts(str(file))
    assert len(prompts) == 1


def test_empty_prompts(tmp_path) -> None:
    file = tmp_path / "prompts.json"
    file.write_text('[{"prompt": ""}]')
    with pytest.raises(ValueError):
        load_prompts(str(file))


def test_missing_prompt_field(tmp_path) -> None:
    file = tmp_path / "prompts.json"
    file.write_text('[{"": "Hello"}]')
    with pytest.raises(ValueError):
        load_prompts(str(file))


def test_invalid_prompt_type(tmp_path) -> None:
    file = tmp_path / "prompts.json"
    file.write_text('[{"prompt": 123}]')
    with pytest.raises(ValueError):
        load_prompts(str(file))


def test_invalid_json_prompt(tmp_path) -> None:
    file = tmp_path / "prompts.json"
    file.write_text('[{"prompt": "Hello"}')
    with pytest.raises(ValueError):
        load_prompts(str(file))



def test_valid_functions(tmp_path: Path) -> None:
    file = tmp_path / "functions.json"
    file.write_text(
        '[{"name": "greet", '
        '"description": "Greet someone", '
        '"parameters": {}, '
        '"returns": {"type": "string"}}]'
    )

    functions = load_functions(str(file))

    assert len(functions) == 1
    assert functions[0].name == "greet"


def test_empty_functions_list(tmp_path: Path) -> None:
    file = tmp_path / "functions.json"
    file.write_text("[]")

    with pytest.raises(ValueError):
        load_functions(str(file))


def test_duplicate_function_names(tmp_path: Path) -> None:
    file = tmp_path / "functions.json"
    file.write_text(
        '[{"name": "greet", '
        '"description": "First", '
        '"parameters": {}, '
        '"returns": {"type": "string"}}, '
        '{"name": "greet", '
        '"description": "Second", '
        '"parameters": {}, '
        '"returns": {"type": "string"}}]'
    )

    with pytest.raises(ValueError, match="Duplicate function name"):
        load_functions(str(file))


def test_unsupported_parameter_type(tmp_path: Path) -> None:
    file = tmp_path / "functions.json"
    file.write_text(
        '[{"name": "greet", '
        '"description": "Greet someone", '
        '"parameters": {"age": {"type": "array"}}, '
        '"returns": {"type": "string"}}]'
    )

    with pytest.raises(ValueError, match="Unsupported type"):
        load_functions(str(file))


def test_invalid_json_functions(tmp_path: Path) -> None:
    file = tmp_path / "functions.json"
    file.write_text('[{"name": "greet"')

    with pytest.raises(ValueError):
        load_functions(str(file))


def test_missing_functions_file(tmp_path: Path) -> None:
    file = tmp_path / "missing.json"

    with pytest.raises(OSError):
        load_functions(str(file))
