"""Tests for the real retry loop without running Qwen."""

import json
import sys
from pathlib import Path
from typing import cast
import pytest

from llm_sdk import Small_LLM_Model
from src import __main__ as app
from src.constraints import ConstrainedDecoder
from src.models import FunctionModel


class FakeEncoding:
    """Return token IDs in the shape expected by the application."""

    def __init__(self, text: str) -> None:
        self.text = text

    def tolist(self) -> list[list[int]]:
        return [[ord(char) for char in self.text]]


class FakeModel:
    """Encode/decode text without loading a language model."""

    def encode(self, text: str) -> FakeEncoding:
        return FakeEncoding(text)

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class FakeDecoder(ConstrainedDecoder):
    """Only replace generation; keep the application's real validation."""

    failures = 0
    calls = 0
    contexts: list[str] = []

    def __init__(
        self, model: Small_LLM_Model, functions: list[FunctionModel]
    ) -> None:
        self.model = model
        self.functions = functions
        self.demo = False

    def generate_function_name(self, input_ids: list[int]) -> list[int]:
        FakeDecoder.calls += 1
        FakeDecoder.contexts.append(self.model.decode(input_ids))

        if FakeDecoder.calls <= FakeDecoder.failures:
            raise ValueError(f"forced failure {FakeDecoder.calls}")

        name_ids = cast(list[int], self.model.encode('greet"').tolist()[0])
        input_ids.extend(name_ids)
        return name_ids

    def generate_parameters(
        self, input_ids: list[int], selected_function: FunctionModel
    ) -> dict[str, str | int | float | bool]:
        return {"name": "Alice"}


@pytest.fixture
def recovery_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Prepare minimal valid inputs and substitute only Qwen-dependent work."""
    prompts_file = tmp_path / "prompts.json"
    functions_file = tmp_path / "functions.json"
    output_file = tmp_path / "output.json"

    prompts_file.write_text('[{"prompt": "Greet Alice"}]')
    functions_file.write_text(
        '[{"name": "greet", "description": "Greet a person", '
        '"parameters": {"name": {"type": "string"}}, '
        '"returns": {"type": "string"}}]'
    )

    FakeDecoder.failures = 0
    FakeDecoder.calls = 0
    FakeDecoder.contexts = []

    monkeypatch.setattr(app, "Small_LLM_Model", FakeModel)
    monkeypatch.setattr(app, "ConstrainedDecoder", FakeDecoder)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "callme",
            "--input", str(prompts_file),
            "--functions_definition", str(functions_file),
            "--output", str(output_file),
        ],
    )
    return output_file


def test_recovery_not_needed(recovery_env: Path) -> None:
    """One successful generation must not trigger a retry."""
    app.main()

    assert FakeDecoder.calls == 1
    result = json.loads(recovery_env.read_text())
    assert result[0]["name"] == "greet"


def test_recovery_success(recovery_env: Path) -> None:
    """One recoverable error must lead to a successful second attempt."""
    FakeDecoder.failures = 1

    app.main()

    assert FakeDecoder.calls == 2
    result = json.loads(recovery_env.read_text())
    assert result[0]["parameters"] == {"name": "Alice"}


def test_recovery_double_failure(recovery_env: Path) -> None:
    """A second ValueError must propagate instead of retrying forever."""
    FakeDecoder.failures = 2

    with pytest.raises(ValueError, match="forced failure 2"):
        app.main()

    assert FakeDecoder.calls == 2
    assert not recovery_env.exists()


def test_recovery_feedback(recovery_env: Path) -> None:
    """The second attempt must include feedback and the original prompt."""
    FakeDecoder.failures = 1

    app.main()

    assert len(FakeDecoder.contexts) == 2
    assert "Previous generation failed" not in FakeDecoder.contexts[0]
    assert (
        "Previous generation failed: forced failure 1"
        in FakeDecoder.contexts[1]
    )
    assert "Greet Alice" in FakeDecoder.contexts[1]
