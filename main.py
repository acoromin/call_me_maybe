from llm_sdk.llm_sdk import Small_LLM_Model
from src.json_parser import load_functions, load_prompts
from src.models import FunctionModel
from src.constraints import ConstrainedDecoder
import json

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
        "\nReturn only a valid JSON function call.\n"
        "Example:\n"
        'User request: Greet Alice\n'
        '{"prompt": "Greet Alice", "name": "fn_greet", '
        '"parameters": {"name": "Alice"}}\n'
    )

    return context


def main():
    try:
        functions = load_functions("data/input/functions_definition.json")
        prompts = load_prompts("data/input/function_calling_tests.json")
    
    except (OSError, ValueError) as e:
        print(e)
        return
    model = Small_LLM_Model()
    constrained = ConstrainedDecoder(model, functions)
    context = build_context(functions)
    results = []

    for prompt in prompts:
        json_prefix = '{"prompt": ' + json.dumps(prompt.prompt) + ', "name": "'
        input_ids = model.encode(
            context + prompt.prompt + json_prefix
            ).tolist()[0]
        output_ids = model.encode(json_prefix).tolist()[0]
        name_ids = constrained.generate_function_name(input_ids, output_ids)
        name = model.decode(name_ids)
        for fn in functions:
            if fn.name == name:
                selected_function = fn
                break

                
        result = {
            "prompt": prompt.prompt,
            "name": model.decode(name_ids),
            "parameters": {}
        }
        results.append(result)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
