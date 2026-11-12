import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.json_parser import load_functions, load_prompts


def function_definition(name="fn_example", param_type="string"):
    return {
        "name": name,
        "description": "Example function",
        "parameters": {"value": {"type": param_type}},
        "returns": {"type": "string"},
    }


class TestLoadPrompts(unittest.TestCase):
    def test_rejects_missing_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "missing.json"
            with self.assertRaises(OSError):
                load_prompts(str(path))

    def test_rejects_broken_json(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text("[{", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_prompts(str(path))

    def test_loads_valid_prompts(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(
                '[{"prompt": "Greet Alice"}]',
                encoding="utf-8",
            )

            prompts = load_prompts(str(path))

            self.assertEqual(len(prompts), 1)
            self.assertEqual(prompts[0].prompt, "Greet Alice")

    def test_rejects_empty_prompt(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(
                '[{"prompt": ""}]',
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_prompts(str(path))

    def test_accepts_empty_prompt_list(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text("[]", encoding="utf-8")
            self.assertEqual(load_prompts(str(path)), [])


class TestLoadFunctions(unittest.TestCase):
    def test_loads_valid_function(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "functions.json"
            path.write_text(
                json.dumps([function_definition()]), encoding="utf-8"
            )
            functions = load_functions(str(path))
            self.assertEqual(functions[0].name, "fn_example")
            self.assertEqual(functions[0].parameters["value"].type, "string")

    def test_rejects_empty_function_list(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "functions.json"
            path.write_text("[]", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_functions(str(path))

    def test_rejects_duplicate_names(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "functions.json"
            path.write_text(
                json.dumps([function_definition(), function_definition()]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate function name"):
                load_functions(str(path))

    def test_rejects_invalid_type_in_first_function(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "functions.json"
            path.write_text(
                json.dumps([
                    function_definition("fn_bad", "array"),
                    function_definition("fn_good"),
                ]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unsupported type"):
                load_functions(str(path))

    def test_rejects_unreadable_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "functions.json"
            path.write_text(
                json.dumps([function_definition()]), encoding="utf-8"
            )
            path.chmod(0)
            try:
                if os.access(path, os.R_OK):
                    self.skipTest("This user can read chmod 000 files")
                with self.assertRaises(OSError):
                    load_functions(str(path))
            finally:
                path.chmod(0o600)
