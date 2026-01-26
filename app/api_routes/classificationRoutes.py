"""FastAPI routes for classification."""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import base64
from io import BytesIO

from app.data_models.classification_state import ClassificationState
from app.langchain_modules.graph.classificationGraph import ClassificationGraphBuilder
from app.langchain_modules.llm_definitions.openrouter_client import create_openrouter_client

router = APIRouter(prefix="/api", tags=["classification"])

# Initialize graph builder (lazy initialization)
_graph_builder: Optional[ClassificationGraphBuilder] = None


def get_graph_builder() -> ClassificationGraphBuilder:
    """Get or create the graph builder instance."""
    global _graph_builder
    if _graph_builder is None:
        llm = create_openrouter_client()
        _graph_builder = ClassificationGraphBuilder(llm)
        _graph_builder.build()
    return _graph_builder


@router.post("/classify")
async def classify_image(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None
):
    """
    Classify an image into main, sub, detail, and level4 categories.
    
    Accepts image in one of three formats:
    - File upload
    - Image URL
    - Base64 encoded image
    
    Returns:
        Classification results with main, sub, detail, and level4 values
    """
    # Determine image data source
    image_data = None
    
    if file:
        # Read uploaded file and convert to base64
        contents = await file.read()
        image_base64_encoded = base64.b64encode(contents).decode('utf-8')
        image_data = f"data:image/{file.content_type.split('/')[-1]};base64,{image_base64_encoded}"
    elif image_url:
        image_data = image_url
    elif image_base64:
        # Assume base64 is provided (may or may not have data URI prefix)
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
        # Get graph builder and run classification
        graph_builder = get_graph_builder()
        result = graph_builder.classify(image_data)
        
        # Return results
        return JSONResponse(content=result.to_dict())
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Classification error: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "classification"}

