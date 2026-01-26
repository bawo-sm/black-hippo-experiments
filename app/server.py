"""FastAPI application server."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api_routes.classificationRoutes import router

# Create FastAPI app
app = FastAPI(
    title="Image Classification API",
    description="Hierarchical 4-step image classification system for home decor and furniture",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Image Classification API",
        "version": "1.0.0",
        "endpoints": {
            "classify": "/api/classify",
            "health": "/api/health"
        }
    }

