"""FastAPI routes for color detection."""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import base64

from app.data_models.color_detection_state import ColorDetectionState
from app.langchain_modules.graph.colorDetectionGraph import ColorDetectionGraphBuilder
from app.langchain_modules.llm_definitions.openrouter_client import create_dual_clients

router = APIRouter(prefix="/api", tags=["color-detection"])

# Initialize graph builder (lazy initialization)
_color_graph_builder: Optional[ColorDetectionGraphBuilder] = None


def get_color_graph_builder() -> ColorDetectionGraphBuilder:
    """Get or create the color detection graph builder instance."""
    global _color_graph_builder
    if _color_graph_builder is None:
        llm_vision, llm_fast = create_dual_clients()
        _color_graph_builder = ColorDetectionGraphBuilder(llm_vision, llm_fast)
        _color_graph_builder.build()
    return _color_graph_builder


@router.post("/classify-colors")
async def classify_colors(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    supplier_reference_description: Optional[str] = None,
    materials: Optional[str] = None
):
    """
    Detect colors in a product image.
    
    Accepts image in one of three formats:
    - File upload
    - Image URL
    - Base64 encoded image
    
    Optional metadata for improved accuracy:
    - supplier_reference_description: Product name/description
    - materials: Materials composition string
    
    Returns:
        Color detection results with up to 3 detail colors,
        corresponding main colors, confidence metadata, and multi flag.
    """
    # Determine image data source
    image_data = None
    
    if file:
        contents = await file.read()
        image_base64_encoded = base64.b64encode(contents).decode('utf-8')
        content_type = file.content_type or "image/jpeg"
        image_data = f"data:image/{content_type.split('/')[-1]};base64,{image_base64_encoded}"
    elif image_url:
        image_data = image_url
    elif image_base64:
        if image_base64.startswith("data:image"):
            image_data = image_base64
        else:
            image_data = f"data:image/jpeg;base64,{image_base64}"
    else:
        raise HTTPException(
            status_code=400,
            detail="Must provide either file, image_url, or image_base64"
        )
    
    try:
        graph_builder = get_color_graph_builder()
        result = graph_builder.detect_colors(
            image_data=image_data,
            supplier_reference_description=supplier_reference_description,
            materials=materials,
        )
        return JSONResponse(content=result.to_dict())
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Color detection error: {str(e)}"
        )


@router.get("/color-health")
async def color_health_check():
    """Health check endpoint for color detection service."""
    return {"status": "healthy", "service": "color-detection"}
