from llm_sdk import Small_LLM_Model
from src.json_parser import load_functions, load_prompts
from src.models import FunctionModel
from src.constraints import ConstrainedDecoder
import json
import argparse
from pathlib import Path

def build_context(functions: list[FunctionModel]) -> str:
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


def main():
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
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show token choices during generation",
    )
    args = parser.parse_args()
    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
    
    except (OSError, ValueError) as e:
        raise SystemExit(f"Error: {e}")
    model = Small_LLM_Model()
    constrained = ConstrainedDecoder(
            model, functions, visualize=args.visualize
            )
    context = build_context(functions)
    system_message = "<|im_start|>system\n" + context + "<|im_end|>\n"
    results = []
    for i, prompt in enumerate(prompts, start=1):
        constrained.show(f"[{i}/{len(prompts)}] Prompt: {prompt.prompt}\n")
        user_message = (
            "<|im_start|>user\n"
            + prompt.prompt
            + "<|im_end|>\n"
        )
        assistant_start = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        json_prefix = '{"prompt": ' + json.dumps(prompt.prompt) + ', "name": "'
        input_ids = model.encode(
            system_message + user_message + assistant_start + json_prefix
        ).tolist()[0]
        name_ids = constrained.generate_function_name(input_ids)
        name = model.decode(name_ids)
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
        results.append(result)
        if args.visualize:
            constrained.show(
                f"JSON -> {json.dumps(result, ensure_ascii=False)}\n\n"
            )
    output_path = Path(args.output)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
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
