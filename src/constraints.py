from llm_sdk import Small_LLM_Model
from src.models import FunctionModel
import numpy as np

class ConstrainedDecoder:
    def __init__(
            self,
            model: Small_LLM_Model,
            functions: list[FunctionModel],
            ) -> None:
        self.model = model
        self.functions = functions
        self.function_tokens = []
        self.build_function_tokens()

    def build_function_tokens(self) -> None:
        for fn in self.functions:
            ids = self.model.encode(fn.name).tolist()[0]
            self.function_tokens.append(ids)

    def get_matching_functions(self, prefix: list[int]) -> list[list[int]]:
        matches = []
        for token_ids in self.function_tokens:
            if token_ids[:len(prefix)] == prefix:
                matches.append(token_ids)
        return matches

    def generate_function_name(
            self,
            input_ids: list[int],
            output_ids: list[int],
            ) -> None:
        prefix = []
        while True:
            matches = self.get_matching_functions(prefix)
            if len(matches) == 1:
                break
            next_tokens = [match[len(prefix)] for match in matches]
            logits = np.asarray(
                    self.model.get_logits_from_input_ids(input_ids))



        
