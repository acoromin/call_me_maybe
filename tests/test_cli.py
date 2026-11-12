import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.__main__ import main


def function_definition() -> dict:
    return {
        "name": "fn_example",
        "description": "Example function",
        "parameters": {"value": {"type": "string"}},
        "returns": {"type": "string"},
    }


class TestCommandLine(unittest.TestCase):
    def test_creates_output_and_overwrites_existing_file(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            functions = root / "functions.json"
            prompts = root / "prompts.json"
            output = root / "nested" / "results.json"
            functions.write_text(
                json.dumps([function_definition()]), encoding="utf-8"
            )
            prompts.write_text("[]", encoding="utf-8")
            args = [
                "src", "--input", str(prompts),
                "--functions_definition", str(functions),
                "--output", str(output),
            ]
            with patch.object(sys, "argv", args), \
                    patch("src.__main__.Small_LLM_Model"), \
                    patch("src.__main__.ConstrainedDecoder"):
                main()
                self.assertEqual(json.loads(output.read_text()), [])
                output.write_text("stale", encoding="utf-8")
                main()
                self.assertEqual(json.loads(output.read_text()), [])

    def test_reports_missing_input(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            functions = root / "functions.json"
            functions.write_text(
                json.dumps([function_definition()]), encoding="utf-8"
            )
            args = [
                "src", "--input", str(root / "missing.json"),
                "--functions_definition", str(functions),
            ]
            with patch.object(sys, "argv", args):
                with self.assertRaisesRegex(SystemExit, "Could not read file"):
                    main()

    def test_reports_unwritable_output_path(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            functions = root / "functions.json"
            prompts = root / "prompts.json"
            functions.write_text(
                json.dumps([function_definition()]), encoding="utf-8"
            )
            prompts.write_text("[]", encoding="utf-8")
            args = [
                "src", "--input", str(prompts),
                "--functions_definition", str(functions),
                "--output", str(root),
            ]
            with patch.object(sys, "argv", args), \
                    patch("src.__main__.Small_LLM_Model"), \
                    patch("src.__main__.ConstrainedDecoder"):
                with self.assertRaisesRegex(
                    SystemExit, "Could not write output"
                ):
                    main()
