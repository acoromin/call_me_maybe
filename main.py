from llm_sdk import Small_LLM_Model
from src.json_parser import load_functions, load_prompts
import numpy as np
from src.models import FunctionModel
from src.constraints import ConstrainedDecoder

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
    for tokens in constrained.fn_tokens:
        print(tokens)
    '''for prompt in prompts:
        encoded = model.encode(context + prompt.prompt).tolist()[0]
        output_ids = []
        for _ in range(50):
            logits = model.get_logits_from_input_ids(encoded)
            token_id = int(np.argmax(logits))
            encoded.append(token_id)
            output_ids.append(token_id)
        response = model.decode(output_ids)
        print(response)'''


if __name__ == "__main__":
    main()
