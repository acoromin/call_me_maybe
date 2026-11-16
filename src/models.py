from pydantic import BaseModel, Field


class ParamType(BaseModel):
    """A parameter or return value type."""

    type: str = Field(min_length=1)


class FunctionModel(BaseModel):
    """A function available for the model to call."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, ParamType]
    returns: ParamType


class InputPrompt(BaseModel):
    """A natural-language request to convert into a function call."""

    prompt: str = Field(min_length=1)


class FunctionCall(BaseModel):
    """The JSON shape of a generated function call."""

    prompt: str = Field(min_length=1)
    name: str = Field(min_length=1)
    parameters: dict[str, str | int | float | bool]
