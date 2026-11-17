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

`make install` and `make run` provide shortcuts. To pass paths through Make, use
`make run ARGS="--input path/to/prompts.json --functions_definition path/to/functions.json --output path/to/results.json"`.
`make debug` runs the program under `pdb`; `make clean` removes Python caches; and
`make lint` runs flake8 and mypy. The Makefile defaults to `GOINFRE=1` and stores
the virtual environment and caches under `/home/acoromin/sgoinfre/call_me_maybe`.
Set `GOINFRE_ROOT` to use a different location, or `GOINFRE=0` to use the project
environment.

During a run, the first prompt displays a short demonstration of token IDs, the
restricted function-name choice, and decoded values. Later prompts display progress
and their completed JSON without the detailed trace. This display goes to standard
error; the complete array is written to the output file after all prompts finish.

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
expected calls and gave 11/11 correct results. Runs of this set on the development
laptop took roughly 75–80 seconds. A run on a 42 machine took about two and a half
minutes, below the five-minute target. Hardware, model download, and input size
affect the time.

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

Run the eight checks in `tests/test_callme.py` with:

```sh
make test
```

Seven checks use the real model against the supplied public inputs and the additional
integer/boolean inputs in `tests/`. They cover booleans, numbers, integers, strings,
regex behavior, input validation, and output structure. The two model runs are
shared across those checks. One separate check simulates a generation failure and
verifies that the retry receives the error, produces one result, and does not pass
the error to the next prompt. It does not load Qwen. Manual checks also included
malformed JSON, unavailable input files, and output write errors.

The test suite expects `tests/prompts_test.json` and `tests/function_test.json` to
be present. Tests can take around two minutes on the development laptop because
they run real inference.

To demonstrate only the retry check without loading the model, run:

```sh
uv run python -m unittest discover -s tests -p 'test_callme.py' -k retry -v
```

## Bonus features

- **Visualization of the generation process:** Run `uv run python -m src`. Each
  prompt shows its request and completed JSON. The first prompt also shows the
  selected function, generated parameters, and a short token-level trace; later
  prompts omit that detail to keep the display readable. Progress goes to
  standard error, while the complete JSON array is written to the output file.
- **Demonstration of how encoding and decoding integrate with constrained decoding:**
  During the first prompt, the same run shows a sample of encoded input IDs,
  the allowed next IDs for function selection, the selected ID, and generated
  IDs beside their decoded text. This displays decisions already made during
  generation and adds no model inference calls.
- **Performance optimizations:** Vocabulary token IDs and text are indexed once,
  so generation need not decode each candidate repeatedly. Valid string IDs are
  cached by parser state. Shared function-name tokens and the remainder of a
  single candidate are appended without asking the model for more logits. These
  optimizations do not use batching; the 42 run took about two and a half minutes.
- **Comprehensive test suite:** `make test` runs eight checks in one file. Seven
  inspect two shared real-model runs; the retry check uses a simulated failure so
  it is deterministic. Use the retry-only command above to demonstrate that
  branch quickly during an evaluation.
- **Advanced error recovery mechanisms:** Each prompt allows up to two attempts.
  If generation or output validation raises `ValueError`, the second attempt starts
  with a fresh token context and adds the first error to the model instructions.
  Errors do not carry over to the next prompt. If both attempts fail, the program
  reports the error instead of writing a partial output file. A normal run shows
  no retry messages; a failed attempt prints `Generation failed: ...` before the
  prompt is attempted again. The retry test verifies this control flow, not that
  Qwen can correct every mistaken argument value.

## Resources

- [Flexible and Efficient Grammar-Constrained Decoding](https://proceedings.mlr.press/v267/park25l.html): background on restricting model outputs with a grammar and tokenizer-aware token masks.
- [Python `json` documentation](https://docs.python.org/3/library/json.html): JSON parsing and serialization.
- [Python `unittest` documentation](https://docs.python.org/3/library/unittest.html): test cases and shared setup.
- [Pydantic model documentation](https://pydantic.dev/docs/validation/latest/concepts/models/): validation of input data.
- [uv project documentation](https://docs.astral.sh/uv/guides/projects/): environments, dependency locking, and running the project.
- [Hugging Face tokenizer documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer): the relationship between token IDs and decoded text.

AI assistance was used to discuss and review the constrained-decoding logic in
`src/constraints.py`, investigate tokenization and JSON edge cases, and compare
generated outputs with expected calls. It also helped draft and simplify
`tests/test_callme.py`, configure the `Makefile`, and draft this README. The work
was checked through code review and executions with the supplied Qwen model.
