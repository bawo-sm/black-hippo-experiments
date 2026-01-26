"""State model for the classification workflow."""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ClassificationState(BaseModel):
    """State model for the classification graph."""
    
    # Image data
    image_data: Optional[str] = Field(
        default=None,
        description="Base64 encoded image or image file path"
    )
    
    # Item metadata (optional, for enhanced classification)
    supplier_name: Optional[str] = Field(
        default=None,
        description="Supplier name for item metadata"
    )
    supplier_reference_description: Optional[str] = Field(
        default=None,
        description="Supplier reference description for item metadata"
    )
    materials: Optional[str] = Field(
        default=None,
        description="Materials information for item metadata"
    )
    
    # Classification results
    main: Optional[str] = Field(
        default=None,
        description="Main category classification result"
    )
    sub: Optional[str] = Field(
        default=None,
        description="Sub category classification result"
    )
    detail: Optional[str] = Field(
        default=None,
        description="Detail category classification result"
    )
    level4: Optional[str] = Field(
        default=None,
        description="Level4 category classification result"
    )
    
    # Metadata
    classification_history: list[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of classification steps"
    )
    
    errors: list[str] = Field(
        default_factory=list,
        description="List of errors encountered during classification"
    )
    
    def add_classification_step(self, step: str, result: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a classification step to history."""
        self.classification_history.append({
            "step": step,
            "result": result,
            "metadata": metadata or {}
        })
    
    def add_error(self, error: str):
        """Add an error to the errors list."""
        self.errors.append(error)
    
    def is_complete(self) -> bool:
        """Check if classification is complete."""
        return all([self.main, self.sub, self.detail, self.level4])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for API response."""
        return {
            "main": self.main,
            "sub": self.sub,
            "detail": self.detail,
            "level4": self.level4,
            "is_complete": self.is_complete(),
            "errors": self.errors
        }

