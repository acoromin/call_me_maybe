*This project has been created as part of the 42 curriculum by acoromin.*

# Call Me Maybe

## Description

Call Me Maybe turns natural-language requests into JSON function calls. Given a list of
available functions and a list of prompts, it selects a function and extracts its
arguments. It does **not** execute the selected function. The default model is
`Qwen/Qwen3-0.6B`, accessed through the supplied `llm_sdk` public API.

The program generates values under token constraints instead of asking the model to
write an entire JSON document freely. It supports `string`, `integer`, `number`, and
`boolean` parameters. Each result has exactly three top-level keys: `prompt`, `name`,
and `parameters`.

## Instructions

Use Python 3.13 or newer and install [uv](https://docs.astral.sh/uv/). From the project
root, install the locked dependencies and run the program:

```sh
uv sync
uv run python -m src
```

The first run may download the model. By default, the program reads
`data/input/function_calling_tests.json` and `data/input/functions_definition.json`,
then writes `data/output/function_calling_results.json`. The output directory is
created if needed. Supply other paths with:

```sh
uv run python -m src \
  --input data/input/function_calling_tests.json \
  --functions_definition data/input/functions_definition.json \
  --output data/output/function_calling_results.json
```

`make install` and `make run` provide shortcuts. To pass paths through Make, use:

```sh
make run ARGS="--input path/to/prompts.json --functions_definition path/to/functions.json --output path/to/results.json"
```

`make debug` runs the program under `pdb`; `make clean` removes Python caches;
`make lint` runs flake8 and mypy; and `make test` runs the pytest unit tests.

The Makefile defaults to `GOINFRE=1`, storing the virtual environment, uv cache,
and Hugging Face cache outside the project under
`/home/acoromin/goinfre/call_me_maybe`. Set `GOINFRE_ROOT` for a different
location or use `GOINFRE=0` for a normal local project environment. For example,
on a machine without the 42 goinfre directory:

```sh
make install GOINFRE=0
make run GOINFRE=0
make test GOINFRE=0
```

During a run, **every prompt** displays a token-level demonstration: an encoded-input
summary, the constrained function-name selection when choices exist, the selected
function, generated argument values, token IDs and their decoded text, and the
completed JSON object. Progress and visualization go to standard error; the complete
JSON array is saved to the output file after all prompts finish. The demonstration
uses information already produced during generation and does not request extra
model inference solely for display.

## Example usage

For an input prompt `What is the sum of 2 and 3?`, with `fn_add_numbers` available,
the corresponding output object is:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

The actual output file contains an array of such objects, in input order. Function
definitions specify each function's name, description, parameter names and types,
and return type. Input prompts are objects with a `prompt` string.

## Algorithm and design decisions

1. Pydantic validates both input files. The parser rejects malformed or missing
   files, empty function lists, duplicate function names, and unsupported parameter
   types.
2. The program builds an instruction context from the available functions and
   encodes it together with the user prompt and the beginning of the JSON output.
3. Each available function name, including its closing quote, is encoded as a token
   sequence. At each position, only next tokens belonging to a remaining function
   are eligible. When several different tokens remain, the model's logits choose
   among them. A shared next token, or the rest of a single remaining function,
   is appended without another model call.
4. The program constructs JSON keys and punctuation. For booleans, it compares the
   logits of `true` and `false`. For integers and numbers, it permits tokens whose
   text keeps a valid numeric prefix and stops only at a valid endpoint. For strings,
   a small state machine handles ordinary text, escapes, Unicode escape digits,
   and the closing quote. Generated strings are parsed with `json.loads`.
5. Before storing each call, Pydantic checks its output shape and the decoder checks
   its function name, required parameter names, and parameter types against the
   selected definition. The values are then serialized with `json.dumps` into the
   final JSON array. The program reports input and output errors with a short message.

Token IDs and their text are stored in dictionaries during initialization, so
generation can look up candidate text without decoding the same token repeatedly.
For strings, valid token IDs are cached by parser state. The demonstration reads IDs
and decisions already produced during generation; it adds no model inference calls.

## Performance and reliability

The stored output for the 11 supplied public prompts was checked against the
expected calls and gave 11/11 correct results. Previous runs on the development
laptop took roughly 75–80 seconds; a 42 machine initially took approximately
2–2.5 minutes, and a more recent run was reported at about 1 minute 55 seconds.
These are observed runs, not a controlled benchmark. Hardware, cache location,
model download, and input size can affect timings.

Constrained candidates and final JSON serialization reduce malformed output. These
measurements do not prove perfect accuracy or valid output for every possible
prompt and tokenizer sequence. Generation has finite limits of 20 numeric tokens
and 128 string tokens; reaching a limit reports an error. Decimal `number` values
may be written as `2.0` even when the prompt contains `2`. Exponent notation is not
implemented in the numeric prefix validator.

## Challenges faced

- Negative numbers needed a decision between a space and a space-plus-minus token
  before generating digits.
- JSON strings needed to distinguish a literal escape such as `\\n` from an
  unescaped control character, and to handle escapes split across model tokens.
- Regex arguments needed instructions that preserve literal symbols: for example,
  “asterisks” should produce `"*"`, not the word `"asterisk"`.
- Model tokens can contain several characters, so numeric and string validation
  examines token text in the current generation state rather than assuming that
  one token equals one character.

## Testing strategy

Run the unit-test suite with:

```sh
make test
```

The suite uses **pytest** and **pytest-sugar** and currently contains **45 passing
test cases**, grouped into three files:

| File | Cases | What it tests |
| --- | ---: | --- |
| `tests/test_parser.py` | 11 | Valid and invalid JSON input, missing fields, types, duplicate function names, and file errors |
| `tests/test_constraints.py` | 30 | Numeric prefixes, JSON string state transitions, function-name constraints, boolean selection, and final-output validation |
| `tests/test_recovery.py` | 4 | No retry, successful retry, two failed attempts, and retry feedback |

The 45 cases ran in approximately **3.29 seconds** in one development run. They do
**not** execute Qwen inference. The constraints tests call validation methods
directly and use controlled logits where required. The recovery tests substitute a
lightweight fake model and decoder generation methods while executing the real
application retry loop.

This separation makes the tests fast and deterministic. It does **not** replace
end-to-end runs with the actual model: unit tests validate specific rules and
control flow, whereas real-model runs assess integrated generation quality,
accuracy, and timing. The unit suite does not claim 100% code coverage.

To run only one group:

```sh
uv run python -m pytest tests/test_parser.py -v
uv run python -m pytest tests/test_constraints.py -v
uv run python -m pytest tests/test_recovery.py -v
```

On 42 machines, use `make test` to inherit the goinfre cache and environment
configuration from the Makefile.

## Bonus features

- **Visualization of the generation process:** Run `uv run python -m src` or
  `make run`. **Every prompt** displays its request, selected function, parameter
  values, token-level trace, and completed JSON. Progress goes to standard error;
  the JSON array is written to the output file.
- **Demonstration of how encoding and decoding integrate with constrained decoding:**
  Each prompt shows a short sample of encoded input IDs, constrained function-name
  decisions where applicable, and generated IDs beside their decoded text.
  The trace reuses decisions already made during generation; no additional model
  inference is requested for visualization.
- **Performance optimizations:** Vocabulary token IDs and text are indexed once,
  so generation need not decode each candidate repeatedly. Valid string IDs are
  cached by parser state. Shared function-name tokens and the remainder of a
  single candidate are appended without asking the model for more logits. These
  optimizations do not use batching. Execution time depends on the machine and
  the model cache.
- **Comprehensive test suite:** `make test` runs **45 pytest cases** across the parser,
  constrained-decoding rules, and retry control flow. They run without loading
  Qwen and use simulated model behavior only where necessary. Full generation
  must still be assessed with a separate real-model run.
- **Advanced error recovery mechanisms:** Each prompt allows up to two attempts.
  If generation or output validation raises `ValueError`, the second attempt starts
  with a fresh token context and adds the first error to the model instructions.
  Errors do not carry over to the next prompt. If both attempts fail, the program
  reports the error instead of writing a partial output file. A failed attempt
  prints `Generation failed: ...` before the retry. The four recovery tests verify
  control flow and feedback, not that Qwen can correct every semantic mistake.

## Resources

- [Flexible and Efficient Grammar-Constrained Decoding](https://proceedings.mlr.press/v267/park25l.html): background on restricting model outputs with a grammar and tokenizer-aware token masks.
- [Python `json` documentation](https://docs.python.org/3/library/json.html): JSON parsing and serialization.
- [pytest documentation](https://docs.pytest.org/en/stable/): unit tests, parametrization, fixtures, and mocking/monkeypatching.
- [Pydantic model documentation](https://docs.pydantic.dev/latest/concepts/models/): validation of input data.
- [uv project documentation](https://docs.astral.sh/uv/guides/projects/): environments, dependency locking, and running the project.
- [Hugging Face tokenizer documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer): the relationship between token IDs and decoded text.

AI assistance was used to discuss and review the constrained-decoding logic in
`src/constraints.py`, investigate tokenization and JSON edge cases, design and draft
the pytest tests, configure the `Makefile`, and draft this README. The work was
reviewed and checked through local tests and executions with the supplied Qwen
model.
