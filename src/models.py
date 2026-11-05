from pydantic import BaseModel, Field


class ParamType(BaseModel):
    type: str = Field(min_length=1)


class FunctionModel(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, ParamType]
    returns: ParamType


class InputPrompt(BaseModel):
    prompt: str = Field(min_length=1)


class FunctionCall(BaseModel):
    prompt: str = Field(min_length=1)
    name: str = Field(min_length=1)
    parameters: dict[str, str | int | float | bool]
