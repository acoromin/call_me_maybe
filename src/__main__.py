from llm_sdk import Small_LLM_Model
from src.json_parser import load_functions, load_prompts
from src.models import FunctionModel
from src.constraints import ConstrainedDecoder
import json
import argparse
import sys
from pathlib import Path


def build_context(functions: list[FunctionModel]) -> str:
    """Describe available functions and output rules in the model prompt.

    Include one JSON example to anchor the required function-call format.
    """

    context = (
        "You are a function calling assistant.\n"
        "Use only the available functions and their defined parameters.\n"
        "Available functions:\n"
    )
    for fn in functions:
        context += f"- {fn.name}: {fn.description} Parameters: "
        for name, param in fn.parameters.items():
            context += f"{name}: {param.type} "
        context += "\n"

    context += (
        "\nCopy literal argument values exactly, preserving leading slashes, "
        "backslashes, spaces, punctuation, and quotes. "
        "Exclude surrounding descriptive words. "
        "For regex replacements, match only the requested characters "
        "and preserve unmatched text.\n"
        "Return only a valid JSON function call.\n"
        "Example:\n"
        'User request: Greet Alice\n'
        '{"prompt": "Greet Alice", "name": "fn_greet", '
        '"parameters": {"name": "Alice"}}\n'
    )

    return context


def main() -> None:
    """Read inputs, generate one call per prompt, and save a JSON array.

    Show the token-level demonstration for the first prompt and shorter
    progress for the others on standard error.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json"
    )
    parser.add_argument(
        "--output", default="data/output/function_calling_results.json"
    )
    args = parser.parse_args()
    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)

    except (OSError, ValueError) as e:
        raise SystemExit(f"Error: {e}")
    model = Small_LLM_Model()
    constrained = ConstrainedDecoder(model, functions)
    context = build_context(functions)
    results = []
    for i, prompt in enumerate(prompts, start=1):
        system_message = "<|im_start|>system\n" + context + "<|im_end|>\n"
        error_message = ""
        for attempt in range(2):
            try:
                retry_context = context

                if attempt == 1:
                    retry_context += (
                        f"\nPrevious generation failed: {error_message}\n"
                        "Retry the function call, avoiding the previous "
                        "failure.\n"
                        "Preserve the exact argument values from the user "
                        "request.\n"
                    )
                    system_message = (
                        "<|im_start|>system\n"
                        + retry_context
                        + "<|im_end|>\n"
                    )
                constrained.demo = i == 1
                progress = f"[{i}/{len(prompts)}] Prompt: {prompt.prompt}\n"
                constrained.show(progress)
                user_message = (
                    "<|im_start|>user\n"
                    + prompt.prompt
                    + "<|im_end|>\n"
                )
                assistant_start = (
                    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                )
                json_prefix = (
                    '{"prompt": ' + json.dumps(prompt.prompt) + ', "name": "'
                )
                input_ids = model.encode(
                    system_message
                    + user_message
                    + assistant_start
                    + json_prefix
                ).tolist()[0]
                constrained.show_demo_input(input_ids)
                name_ids = constrained.generate_function_name(input_ids)
                name = model.decode(name_ids).removesuffix('"')
                constrained.show_demo_tokens("function", name_ids, name)
                for fn in functions:
                    if fn.name == name:
                        selected_function = fn
                        break
                input_ids.extend(
                    model.encode(', "parameters": {').tolist()[0]
                    )
                parameters = constrained.generate_parameters(
                    input_ids, selected_function
                    )
                result = {
                    "prompt": prompt.prompt,
                    "name": name,
                    "parameters": parameters
                }
                constrained.validate_output(result)
                results.append(result)
                print(
                    f"JSON -> {json.dumps(result, ensure_ascii=False)}\n",
                    file=sys.stderr,
                    flush=True,
                )
                break
            except ValueError as e:
                error_message = str(e)
                print(f"Generation failed: {e}")
                if attempt == 1:
                    raise
    output_path = Path(args.output)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        print(
            f"Saved {len(results)} results to {output_path}",
            file=sys.stderr,
            flush=True,
        )
    except (OSError, ValueError) as e:
        raise SystemExit(
            f"Error: Could not write output file '{output_path}': {e}"
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("\nInterrupted by user")
    except ValueError as e:
        raise SystemExit(f"Error generating output: {e}")
