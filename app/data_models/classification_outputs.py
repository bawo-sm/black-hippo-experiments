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


# ── Color Detection Output Models ──────────────────────────────────────────────

class ColorDescriptionOutput(BaseModel):
    """Output model for color-focused image description."""
    description: str = Field(
        description="Compact color-focused description of the product image"
    )
    estimated_color_count: int = Field(
        description="Estimated number of distinct colors visible in the product (1-10+)"
    )


class PrimaryColorOutput(BaseModel):
    """Output model for primary detail color selection."""
    detail_color: str = Field(
        description="Primary detail color name from the allowed color list"
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0 for this color choice"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief reasoning for the color selection"
    )


class MultiGateOutput(BaseModel):
    """Output model for the multi-gate binary classifier."""
    is_multi: bool = Field(
        default=False,
        description="True ONLY if the product has 4+ truly distinct prominent colors (patchwork, rainbow, etc.)"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief reasoning for the multi/not-multi decision"
    )


class SecondaryColorsOutput(BaseModel):
    """Output model for secondary detail colors selection (multi already decided upstream)."""
    detail_color_2: Optional[str] = Field(
        default=None,
        description="Second detail color name from the allowed color list, or null if not applicable"
    )
    confidence_2: Optional[float] = Field(
        default=None,
        description="Confidence score (0.0-1.0) for second color"
    )
    reasoning_2: Optional[str] = Field(
        default=None,
        description="Brief reasoning for color 2"
    )
    detail_color_3: Optional[str] = Field(
        default=None,
        description="Third detail color name from the allowed color list, or null if not applicable"
    )
    confidence_3: Optional[float] = Field(
        default=None,
        description="Confidence score (0.0-1.0) for third color"
    )
    reasoning_3: Optional[str] = Field(
        default=None,
        description="Brief reasoning for color 3"
    )


class NeutralVerifyOutput(BaseModel):
    """Output model for the neutral-family verification step."""
    detail_color: str = Field(
        description="Corrected neutral color name from the narrow neutral list"
    )
    confidence: float = Field(
        description="Confidence score (0.0-1.0) for the corrected choice"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief reasoning for choosing this specific neutral shade"
    )


