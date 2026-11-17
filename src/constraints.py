from llm_sdk import Small_LLM_Model
from src.models import FunctionCall, FunctionModel
import numpy as np
import json
import sys
from collections.abc import Mapping
from typing import cast


class ConstrainedDecoder:
    """Build function calls from tokens allowed by the current JSON position.

    Keep token lookups and string-state caches across generated prompts.
    """

    def __init__(
            self,
            model: Small_LLM_Model,
            functions: list[FunctionModel],
            ) -> None:
        """Encode the function choices and index useful vocabulary tokens."""

        self.model = model
        self.functions = functions
        self.demo = False
        self.function_tokens: list[list[int]] = []
        self.build_function_tokens()
        self.bool_tokens = [
                self.model.encode(value).tolist()[0]
                for value in ("true", "false")
                ]
        self.int_tokens: dict[int, str] = {}
        self.float_tokens: dict[int, str] = {}
        self.str_special_tokens: dict[str, int] = {}
        self.str_tokens: dict[int, str] = {}
        self.str_valid_cache: dict[str, list[int]] = {}
        self.build_token_constraints()

    def build_function_tokens(self) -> None:
        """Encode function names with the quote that ends the JSON value."""

        for fn in self.functions:
            ids = self.model.encode(f'{fn.name}"').tolist()[0]
            self.function_tokens.append(ids)

    def get_matching_functions(self, prefix: list[int]) -> list[list[int]]:
        """Return function encodings that still match the generated IDs."""

        matches = []
        for token_ids in self.function_tokens:
            if token_ids[:len(prefix)] == prefix:
                matches.append(token_ids)
        return matches

    def generate_function_name(
            self,
            input_ids: list[int],
            ) -> list[int]:
        """Select a function name one token at a time.

        Ask the model only when multiple allowed next tokens remain;
        append forced tokens directly. Return the name IDs without its quote.
        """

        prefix: list[int] = []
        demo_choice_shown = False
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
                if self.demo and not demo_choice_shown:
                    self.show_demo_choice(next_tokens, select_token)
                    demo_choice_shown = True
            input_ids.append(select_token)
            prefix.append(select_token)
        if self.demo and not demo_choice_shown:
            self.show("  function choice -> forced by constraints\n")
        if self.demo:
            self.show(f"function -> {self.model.decode(prefix[:-1])}\n")

        return prefix[:-1]

    def build_token_constraints(self) -> None:
        """Index tokens usable by the numeric and string generators.

        Numeric candidates use vocabulary text; string candidates use decoded
        text so later validation sees the characters the model will emit.
        """

        vocab_path = self.model.get_path_to_vocab_file()
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        for token, token_id in vocab.items():
            if token and all(c in "0123456789-" for c in token):
                self.int_tokens[token_id] = token
            if token and all(c in "0123456789.-eE+" for c in token):
                self.float_tokens[token_id] = token
            if any(c in '"\\' or ord(c) < 32 for c in token):
                self.str_special_tokens[token] = token_id
            decoded = self.model.decode([token_id])
            if decoded:
                self.str_tokens[token_id] = decoded

    def is_valid_number_prefix(self, prefix: str, token: str) -> bool:
        """Allow prefixes of a JSON number with digits and an optional point.

        An unfinished minus or trailing point can be continued, but cannot
        finish a generated value. Exponent notation is not supported here.
        """

        number = prefix + token

        if number == "-":
            return True
        if number.startswith("-"):
            number = number[1:]
        if not number:
            return False
        if number[0] == ".":
            return False
        if number.count(".") > 1:
            return False
        if not all(c in "0123456789." for c in number):
            return False
        if len(number) > 1 and number[0] == "0" and number[1] != ".":
            return False

        return True

    def generate_number(self, input_ids: list[int], stop_char: str) -> float:
        """Generate a number from valid continuations and return a float.

        Choose the leading space or minus first, then allow the closing
        delimiter only after the accumulated number is complete.
        """

        prefix = ""
        space_id = self.model.encode(" ").tolist()[0][0]
        space_minus_id = self.model.encode(" -").tolist()[0][0]
        logits = np.asarray(self.model.get_logits_from_input_ids(input_ids))
        if logits[space_minus_id] > logits[space_id]:
            input_ids.append(space_minus_id)
            prefix = "-"
        else:
            input_ids.append(space_id)
        stop_id = self.model.encode(stop_char).tolist()[0][0]

        for _ in range(20):
            logits = np.asarray(
                    self.model.get_logits_from_input_ids(input_ids))
            valid_tokens = []
            for token_id, token in self.float_tokens.items():
                if self.is_valid_number_prefix(prefix, token):
                    valid_tokens.append(token_id)
            if prefix and prefix != "-" and not prefix.endswith("."):
                valid_tokens.append(stop_id)
            if not valid_tokens:
                raise ValueError(
                        f"No valid number tokens for prefix: {prefix!r}"
                        )
            allowed_scores = logits[valid_tokens]
            select_token = valid_tokens[int(np.argmax(allowed_scores))]
            if select_token == stop_id:
                return float(prefix)
            input_ids.append(select_token)
            prefix += self.float_tokens[select_token]
        raise ValueError("Number generation exceeded token limit")

    def is_valid_integer_prefix(self, prefix: str, token: str) -> bool:
        """Accept digit or minus tokens that preserve an integer prefix.

        Reject non-digits and leading zeros; a lone minus is only a prefix.
        """

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
        """Generate an integer from valid tokens and return its value.

        Choose the leading space or minus first, then permit the closing
        delimiter only when at least one digit has been generated.
        """

        prefix = ""
        space_id = self.model.encode(" ").tolist()[0][0]
        space_minus_id = self.model.encode(" -").tolist()[0][0]
        logits = np.asarray(self.model.get_logits_from_input_ids(input_ids))
        if logits[space_minus_id] > logits[space_id]:
            input_ids.append(space_minus_id)
            prefix = "-"
        else:
            input_ids.append(space_id)
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

    def check_str_token(self, state: str, token: str) -> tuple[bool, str]:
        """Check whether token text can continue a JSON string.

        Track normal text, escapes, four hexadecimal digits after ``\\u``,
        and a closing quote. Return validity and the resulting state.
        """

        for char in token:
            if state == "CLOSED":
                return False, state
            if state == "ESCAPE":
                if char == "u":
                    state = "U4"
                elif char in ('"', "\\", "/", "b", "f", "n", "r", "t"):
                    state = "NORMAL"
                else:
                    return False, state
            elif state == "U4":
                if char not in "0123456789abcdefABCDEF":
                    return False, state
                state = "U3"
            elif state == "U3":
                if char not in "0123456789abcdefABCDEF":
                    return False, state
                state = "U2"
            elif state == "U2":
                if char not in "0123456789abcdefABCDEF":
                    return False, state
                state = "U1"
            elif state == "U1":
                if char not in "0123456789abcdefABCDEF":
                    return False, state
                state = "NORMAL"
            elif state == "NORMAL" and char == "\\":
                state = "ESCAPE"
            elif state == "NORMAL" and char == '"':
                state = "CLOSED"
            elif state == "NORMAL" and ord(char) < 32:
                return False, state
        return True, state

    def generate_string(self, input_ids: list[int], stop_char: str) -> str:
        """Generate a quoted JSON string and return the parsed Python text.

        Cache valid token IDs by string state. A closing quote, optionally
        joined to the delimiter, ends generation when the state allows it.
        """

        state = "NORMAL"
        quote_id = self.model.encode('"').tolist()[0][0]
        input_ids.append(quote_id)
        string_ids = [quote_id]
        closing_tokens = [
            token_id for token_id, token in self.str_tokens.items()
            if token == '"' + stop_char
        ]

        for _ in range(128):
            logits = np.asarray(
                self.model.get_logits_from_input_ids(input_ids)
            )
            if state not in self.str_valid_cache:
                self.str_valid_cache[state] = [
                    token_id
                    for token_id, token in self.str_tokens.items()
                    if self.check_str_token(state, token)[0]
                ]
            valid_tokens = self.str_valid_cache[state].copy()
            if state == "NORMAL":
                valid_tokens.extend(closing_tokens)
            if not valid_tokens:
                raise ValueError("No valid string tokens")

            scores = logits[valid_tokens]
            selected = valid_tokens[int(np.argmax(scores))]
            if selected in closing_tokens:
                input_ids.append(selected)
                string_ids.append(selected)
                return cast(
                    str, json.loads(self.model.decode(string_ids)[:-1])
                )
            token = self.str_tokens[selected]
            _, state = self.check_str_token(state, token)
            input_ids.append(selected)
            string_ids.append(selected)
            if state == "CLOSED":
                return cast(str, json.loads(self.model.decode(string_ids)))

        raise ValueError("String generation exceeded token limit")

    def generate_param_value(
            self,
            input_ids: list[int],
            param_type: str,
            stop_char: str,
            ) -> str | int | float | bool:
        """Dispatch to the generator for a parameter's declared type.

        Boolean values are selected directly between the encoded true and
        false tokens; other types use their constrained generators.
        """

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
            return self.generate_number(input_ids, stop_char)
        elif param_type == "string":
            return self.generate_string(input_ids, stop_char)
        else:
            raise ValueError(f"Unsupported parameter type {param_type}")

    def generate_parameters(
            self,
            input_ids: list[int],
            selected_function: FunctionModel,
            ) -> dict[str, str | int | float | bool]:
        """Generate parameters in definition order and collect their values.

        Add each JSON key to the model input before generating its value;
        pass the expected delimiter to the value generator.
        """

        result = {}
        label = "parameters -> "
        comma_already_written = False
        for i, (name, param) in enumerate(
            selected_function.parameters.items()
        ):
            stop = "}" if i == len(selected_function.parameters) - 1 else ","
            separator = " " if comma_already_written else ", "
            key = (separator if i > 0 else "") + json.dumps(name)
            key += ":" if param.type in ("integer", "number") else ": "
            input_ids.extend(self.model.encode(key).tolist()[0])
            value_start = len(input_ids)
            value = self.generate_param_value(input_ids, param.type, stop)
            if self.demo:
                indent = label if i == 0 else " " * len(label)
                shown_value = json.dumps(value, ensure_ascii=False)
                self.show(f"{indent}{name} = {shown_value}\n")
                value_ids = input_ids[value_start:]
                self.show_demo_tokens(
                    f"value {name}", value_ids,
                    self.model.decode(value_ids),
                )
            result[name] = value
            comma_already_written = (
                param.type == "string"
                and stop == ","
                and self.str_tokens.get(input_ids[-1]) == '",'
            )
        if self.demo and not selected_function.parameters:
            self.show("parameters -> {}\n")
        return result

    def validate_output(self, result: Mapping[str, object]) -> None:
        """Validate the result shape and the chosen function's parameters."""

        if set(result) != {"prompt", "name", "parameters"}:
            raise ValueError("Output must contain prompt, name and parameters")
        call = FunctionCall.model_validate(result, strict=True)
        selected_function = next(
            (fn for fn in self.functions if fn.name == call.name), None
        )
        if selected_function is None:
            raise ValueError(f"Unknown function in output: {call.name}")
        if set(call.parameters) != set(selected_function.parameters):
            raise ValueError(f"Incorrect parameters for {call.name}")

        allowed_types = {
            "string": (str,),
            "integer": (int,),
            "number": (int, float),
            "boolean": (bool,),
        }
        for name, param in selected_function.parameters.items():
            expected = allowed_types.get(param.type)
            if expected is None or type(call.parameters[name]) not in expected:
                raise ValueError(
                    f"Incorrect type for {call.name}.{name}: "
                    f"expected {param.type}"
                )

    def show(self, text: str) -> None:
        """Write generation progress to standard error."""

        print(text, end="", file=sys.stderr, flush=True)

    def show_demo_input(self, input_ids: list[int]) -> None:
        """Show a short preview of the encoded model input."""

        if self.demo:
            self.show(
                f"  encoded input -> {len(input_ids)} IDs; "
                f"last 8: {input_ids[-8:]}\n"
            )

    def show_demo_choice(
            self, allowed_ids: list[int], selected_id: int
            ) -> None:
        """Show the allowed next IDs and the ID selected by the model."""

        unique_ids = list(dict.fromkeys(allowed_ids))
        self.show(
            f"  allowed next IDs -> {unique_ids[:8]} "
            f"({len(unique_ids)} total)\n"
            f"  selected ID -> {selected_id}\n"
        )

    def show_demo_tokens(
            self, label: str, token_ids: list[int], decoded: str
            ) -> None:
        """Show how generated IDs decode into a function or value."""

        if self.demo:
            self.show(
                f"  {label} IDs -> {token_ids}\n"
                f"  {label} decoded -> {decoded!r}\n"
            )
