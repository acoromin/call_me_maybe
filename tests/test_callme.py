"""Checks for the main features of the function calling program."""

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar, cast

from src.json_parser import load_functions, load_prompts


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


if __name__ == "__main__":
    unittest.main()
