from pydantic import BaseModel, Field


class ParamType(BaseModel):
    type: str


class FunctionModel(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParamType]
    returns: ParamType


class InputPrompt(BaseModel):
    prompt: str


class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]



