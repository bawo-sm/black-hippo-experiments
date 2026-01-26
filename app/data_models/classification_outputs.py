"""Pydantic output models for each classification step."""
from pydantic import BaseModel, Field
from typing import Optional


class MainClassificationOutput(BaseModel):
    """Output model for main classification."""
    main: str = Field(
        description="Main category value from main_values.json"
    )
    confidence: Optional[str] = Field(
        default=None,
        description="Optional confidence level or reasoning"
    )


class SubClassificationOutput(BaseModel):
    """Output model for sub classification."""
    sub: str = Field(
        description="Sub category value from sub_values.json"
    )
    confidence: Optional[str] = Field(
        default=None,
        description="Optional confidence level or reasoning"
    )


class DetailClassificationOutput(BaseModel):
    """Output model for detail classification."""
    detail: str = Field(
        description="Detail category value from detail_values.json"
    )
    confidence: Optional[str] = Field(
        default=None,
        description="Optional confidence level or reasoning"
    )


class Level4ClassificationOutput(BaseModel):
    """Output model for level4 classification."""
    level4: str = Field(
        description="Level4 category value from level4_values.json"
    )
    confidence: Optional[str] = Field(
        default=None,
        description="Optional confidence level or reasoning"
    )

