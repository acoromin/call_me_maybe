from pathlib import Path
from pydantic import TypeAdapter, ValidationError

from src.models import FunctionModel, InputPrompt


def load_functions(path: str) -> list[FunctionModel]:
    try:
        content = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as e:
        raise OSError(f"Could not read file '{path}': {e}")

    try:
        adapter = TypeAdapter(list[FunctionModel])
        functions = adapter.validate_json(content)
    except ValidationError as e:
        raise ValueError(f"Invalid functions JSON: {e}")
    
    if not functions:
        raise ValueError("Functions list cannot be empty")
    seen_names = set()
    for function in functions:
        if function.name in seen_names:
            raise ValueError(f"Duplicate function name: {function.name}")
        seen_names.add(function.name)
        for name, param in function.parameters.items():
            if param.type not in ("string", "number", "integer", "boolean"):
                raise ValueError(
                    f"Unsupported type '{param.type}' in "
                    f"{function.name}.{name}"
                )

    return functions

def load_prompts(path: str) -> list[InputPrompt]:
    try:
        content = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as e:
        raise OSError(f"Could not read file '{path}': {e}")

    try:
        adapter = TypeAdapter(list[InputPrompt])
        prompts = adapter.validate_json(content)
    except ValidationError as e:
        raise ValueError(f"Invalid prompts JSON: {e}")

    return prompts  
