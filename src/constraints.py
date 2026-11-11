from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import FunctionModel
import numpy as np
import json

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
        self.bool_tokens = [
                self.model.encode(value).tolist()[0]
                for value in ("true", "false")
                ]
        self.int_tokens = {}
        self.float_tokens = {}
        self.str_special_tokens = {}
        self.build_token_constraints()

    def build_function_tokens(self) -> None:
        for fn in self.functions:
            ids = self.model.encode(f'{fn.name}"').tolist()[0]
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
            ) -> list[int]:
        prefix = []
        while True:
            matches = self.get_matching_functions(prefix)
            if not matches:
                raise ValueError("No matching functions")
            if len(matches) == 1:
                input_ids.extend(matches[0][len(prefix):])
                prefix.extend(matches[0][len(prefix):])
                break
            next_tokens = [match[len(prefix)] for match in matches]
            if len(set(next_tokens)) == 1:
                select_token = next_tokens[0]
            else:
                logits = np.asarray(
                        self.model.get_logits_from_input_ids(input_ids))
                allowed_scores = logits[next_tokens]
                select_token = next_tokens[int(np.argmax(allowed_scores))]
            input_ids.append(select_token)
            prefix.append(select_token)

        return prefix[:-1]

    def build_token_constraints(self) -> None:
        vocab_path = self.model.get_path_to_vocab_file()
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        for token, token_id in vocab.items():
            if token and all(c in "0123456789-" for c in token):
                self.int_tokens[token_id] = token
            if token and all(c in "0123456789.-eE+" for c in token):
                self.float_tokens[token] = token_id
            if any(c in '"\\' or ord(c) < 32 for c in token):
                self.str_special_tokens[token] = token_id

    def is_valid_integer_prefix(self, prefix: str, token: str) -> bool:
        number = prefix + token

        if number == "-":
            return True
        if number.startswith("-"):
            number = number[1:]
        if not number:
            return False
        if not all(c in "0123456789" for c in number):
            return False
        if len(number) > 1 and number[0] == "0":
            return False

        return True

    def generate_int(self, input_ids: list[int], stop_char: str) -> int:
        prefix = ""
        stop_id = self.model.encode(stop_char).tolist()[0][0]
        for _ in range(20):
            logits = np.asarray(
                    self.model.get_logits_from_input_ids(input_ids))
            valid_tokens = []
            for token_id, token in self.int_tokens.items():
                if self.is_valid_integer_prefix(prefix, token):
                    valid_tokens.append(token_id)
            if prefix and prefix != "-":
                valid_tokens.append(stop_id)
            if not valid_tokens:
                raise ValueError(
                        f"No valid integer tokens for prefix: {prefix!r}"
                        )
            allowed_scores = logits[valid_tokens]
            select_token = valid_tokens[int(np.argmax(allowed_scores))]
            if select_token == stop_id:
                return int(prefix)
            input_ids.append(select_token)
            prefix += self.int_tokens[select_token]
        raise ValueError("Integer generation exceeded token limit")

    def generate_param_value(
            self,
            input_ids: list[int],
            param_type: str,
            stop_char: str,
            ) -> str | int | float | bool:
       
        if param_type == "boolean":
            true_id = self.bool_tokens[0][0]
            false_id = self.bool_tokens[1][0]

            logits = self.model.get_logits_from_input_ids(input_ids)

            if logits[true_id] >= logits[false_id]:
                input_ids.append(true_id)
                return True
            else:
                input_ids.append(false_id)
                return False
        elif param_type == "integer":
            return self.generate_int(input_ids, stop_char)
        elif param_type == "number":
            return "pending"
        elif param_type == "string":
            return "pending"
        else:
            raise ValueError(f"Unsupported parameter type {param_type}")

    def generate_parameters(
            self,
            input_ids: list[int],
            selected_function: FunctionModel,
            ) -> dict[str, str | int | float | bool]:
        result = {}
        for i, (name, param) in enumerate(selected_function.parameters.items()):
            stop = "}" if i == len(selected_function.parameters) - 1 else ","
            key = (", " if i > 0 else "") + json.dumps(name) + ": "
            input_ids.extend(self.model.encode(key).tolist()[0])
            value = self.generate_param_value(input_ids, param.type, stop)
            result[name] = value
        return result
