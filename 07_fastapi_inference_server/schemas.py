"""
Project 07

Pydantic schemas for request validation.
"""

from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        description="User prompt",
    )

class StreamRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        description="User prompt",
    )

class GenerateResponse(BaseModel):
    response: str


"""
Field(...)      # Required field
Field(None)     # Optional field (default = None)
Field("")       # Optional field (default = "")
"""