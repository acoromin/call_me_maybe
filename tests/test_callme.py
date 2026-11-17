"""Checks for the main features of the function calling program."""

import json
import re
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar, cast
from unittest.mock import MagicMock, patch

from src.__main__ import main
from src.json_parser import load_functions, load_prompts
from src.models import InputPrompt


ROOT = Path(__file__).resolve().parents[1]


class TestCallMe(unittest.TestCase):
    """Check real results produced by Qwen."""

    public: ClassVar[list[dict[str, Any]]]
    extra: ClassVar[list[dict[str, Any]]]

    @classmethod
    def setUpClass(cls) -> None:
        """Run each input set once, then share its results between tests."""
        cls.public = cls.run_case(
            ROOT / "data/input/function_calling_tests.json",
            ROOT / "data/input/functions_definition.json",
        )
        cls.extra = cls.run_case(
            ROOT / "tests/prompts_test.json",
            ROOT / "tests/function_test.json",
        )

    @staticmethod
    def run_case(prompts: Path, functions: Path) -> list[dict]:
        """Run the program and read its JSON output."""
        with TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            command = [
                sys.executable, "-m", "src",
                "--input", str(prompts),
                "--functions_definition", str(functions),
                "--output", str(output),
            ]
            result = subprocess.run(
                command, cwd=ROOT, capture_output=True,
                text=True, timeout=300, check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            return cast(
                list[dict[str, Any]],
                json.loads(output.read_text(encoding="utf-8")),
            )

    def test_boolean(self) -> None:
        """Check both true and false."""
        self.assertEqual(self.extra[6]["parameters"], {"enabled": True})
        self.assertEqual(self.extra[7]["parameters"], {"enabled": False})

    def test_float(self) -> None:
        """Check that number parameters are floats with the right values."""
        self.assertEqual(self.public[0]["name"], "fn_add_numbers")
        self.assertEqual(self.public[0]["parameters"], {"a": 2.0, "b": 3.0})
        self.assertIs(type(self.public[0]["parameters"]["a"]), float)

    def test_integer(self) -> None:
        """Check positive and negative integers."""
        self.assertEqual(self.extra[0]["parameters"], {"age": 25})
        self.assertEqual(self.extra[2]["parameters"], {"temperature": -15})
        self.assertEqual(self.extra[5]["parameters"], {"count": -123456})
        self.assertIs(type(self.extra[2]["parameters"]["temperature"]), int)

    def test_string(self) -> None:
        """Check a string argument."""
        self.assertEqual(self.public[2]["name"], "fn_greet")
        self.assertEqual(self.public[2]["parameters"], {"name": "shrek"})

    def test_input_files(self) -> None:
        """Check valid input files and a malformed JSON file."""
        functions = load_functions(
            str(ROOT / "data/input/functions_definition.json")
        )
        prompts = load_prompts(
            str(ROOT / "data/input/function_calling_tests.json")
        )
        self.assertEqual(len(functions), 5)
        self.assertEqual(len(prompts), 11)
        with TemporaryDirectory() as directory:
            broken = Path(directory) / "broken.json"
            broken.write_text("[{", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_prompts(str(broken))

    def test_output_format(self) -> None:
        """Check one JSON object per prompt and the required keys."""
        self.assertEqual(len(self.public), 11)
        self.assertEqual(len(self.extra), 8)
        for call in self.public + self.extra:
            self.assertEqual(set(call), {"prompt", "name", "parameters"})
            self.assertIsInstance(call["prompt"], str)
            self.assertIsInstance(call["name"], str)
            self.assertIsInstance(call["parameters"], dict)

    def test_regex(self) -> None:
        """Check that the regex call produces the requested substitution."""
        call = self.public[9]
        self.assertEqual(call["name"], "fn_substitute_string_with_regex")
        args = call["parameters"]
        self.assertEqual(args["replacement"], "*")
        self.assertEqual(
            re.sub(args["regex"], args["replacement"], args["source_string"]),
            "Pr*gr*mm*ng *s f*n",
        )


class TestRetry(unittest.TestCase):
    """Check the retry flow without loading Qwen."""

    def test_retry_uses_feedback_and_resets_context(self) -> None:
        """Retry one failure, then start the next prompt without feedback."""
        model = MagicMock()
        model.encode.return_value.tolist.return_value = [[1]]
        model.decode.return_value = "fn_greet"
        decoder = MagicMock()
        decoder.generate_function_name.return_value = [1]
        decoder.generate_parameters.side_effect = [
            ValueError("String generation exceeded token limit"),
            {"name": "Alice"},
            {"name": "Bob"},
        ]
        prompts = [InputPrompt(prompt="Greet Alice"),
                   InputPrompt(prompt="Greet Bob")]

        with TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            log = StringIO()
            with (
                patch("src.__main__.Small_LLM_Model", return_value=model),
                patch("src.__main__.ConstrainedDecoder", return_value=decoder),
                patch("src.__main__.load_prompts", return_value=prompts),
                patch.object(sys, "argv", ["src", "--output", str(output)]),
                redirect_stdout(log),
                redirect_stderr(StringIO()),
            ):
                main()

            contexts = [
                call.args[0] for call in model.encode.call_args_list
                if call.args[0].startswith("<|im_start|>system")
            ]
            results = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(len(contexts), 3)
        self.assertNotIn("Previous generation failed", contexts[0])
        self.assertIn("String generation exceeded token limit", contexts[1])
        self.assertNotIn("Previous generation failed", contexts[2])
        self.assertEqual(decoder.generate_parameters.call_count, 3)
        self.assertEqual(
            [result["parameters"]["name"] for result in results],
            ["Alice", "Bob"],
        )
        self.assertIn("Generation failed", log.getvalue())


if __name__ == "__main__":
    unittest.main()
