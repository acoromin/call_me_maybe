from pathlib import Path
from pydantic import TypeAdapter

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
