from fastapi import FastAPI, BackgroundTasks
from sentence_transformers import SentenceTransformer
from src.ai.ai_search_connector import AISearchConnector
from src.common.schema import SimSearchClassificationRequest
from src.services import IdentityService
from src.endpoints import *
from src.settings import *


embedder = SentenceTransformer(EMBEDDING_MODEL)
app = FastAPI(
    title="ItemsClassification"
)


@app.get(HOME_URL)
async def endpoint_home():
    return "Hello bro!"


@app.post(SIM_SEARCH_CLASSIFICATION_URL)
async def endpoint_sim_search_classification_items(
    order: SimSearchClassificationRequest,
    background_tasks: BackgroundTasks
):  
    def task():
        SimilaritySearchClassification().run(
            documents=order.documents,
            ai_search_connector=AISearchConnector(IdentityService.get_azure_crdentials(ADMIN_INDEX_SECRET_KEY)), # !!!!!!!!
            embedder=embedder
        )

    background_tasks.add_task(task)

    return {"message": "Similarity Search classification is running"}
