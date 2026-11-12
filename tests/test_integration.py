import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    os.environ.get("CALLME_RUN_MODEL_TESTS") == "1",
    "Set CALLME_RUN_MODEL_TESTS=1 to load Qwen",
)
class TestModelIntegration(unittest.TestCase):
    def test_generates_valid_function_calls(self) -> None:
        prompts = [
            "What is the sum of 2 and 3?",
            "Greet shrek",
            'Replace all numbers in "Hello 34 I\'m 233 years old" '
            'with NUMBERS',
        ]
        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "prompts.json"
            output_path = Path(directory) / "results.json"
            input_path.write_text(
                json.dumps([{"prompt": prompt} for prompt in prompts]),
                encoding="utf-8",
            )
            command = [
                sys.executable, "-m", "src",
                "--input", str(input_path),
                "--functions_definition",
                str(ROOT / "data/input/functions_definition.json"),
                "--output", str(output_path),
            ]
            completed = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True,
                timeout=300, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            results = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(len(results), len(prompts))
        self.assertEqual([row["prompt"] for row in results], prompts)
        for row in results:
            self.assertEqual(set(row), {"prompt", "name", "parameters"})
        self.assertEqual(results[0]["name"], "fn_add_numbers")
        self.assertEqual(results[0]["parameters"], {"a": 2.0, "b": 3.0})
        self.assertEqual(results[1]["name"], "fn_greet")
        self.assertEqual(results[1]["parameters"], {"name": "shrek"})
        self.assertEqual(
            results[2]["name"], "fn_substitute_string_with_regex"
        )
        parameters = results[2]["parameters"]
        self.assertEqual(parameters["replacement"], "NUMBERS")
        self.assertEqual(
            re.sub(
                parameters["regex"], parameters["replacement"],
                parameters["source_string"],
            ),
            "Hello NUMBERS I'm NUMBERS years old",
        )
