from dotenv import load_dotenv
from uuid import uuid4
from fastapi import FastAPI, BackgroundTasks
from sentence_transformers import SentenceTransformer
from src.ai.ai_search_connector import AISearchConnector
from src.common.schema import SimSearchClassificationRequest, GetStatusRequest, GetStatusResponse
from src.common.db_schema import SQLTaskStatus, TaskEnum, TaskStatusEnum
from src.common.utils import get_env_variable
from src.services import IdentityService, SQLService
from src.endpoints import *
from src.settings import *


load_dotenv(".env")
embedder = SentenceTransformer(EMBEDDING_MODEL)
app = FastAPI(
    title="ItemsClassification"
)


@app.get(HOME_URL)
async def endpoint_home():
    return "Hello bro!"


@app.post(GET_STATUS_URL, response_model=GetStatusResponse)
async def endpoint_get_status(order: GetStatusRequest):
    return GetTaskStatus().run(order.task_id)


@app.post(SIM_SEARCH_CLASSIFICATION_URL)
async def endpoint_sim_search_classification_items(
    order: SimSearchClassificationRequest,
    background_tasks: BackgroundTasks
):  
    task_id = str(uuid4())
    SQLService.set_task_status(
        SQLTaskStatus(
            task_uuid=task_id,
            task=TaskEnum.classification,
            status=TaskStatusEnum.in_progress
        )
    )

    def task():
        try:
            creds = IdentityService.get_azure_credentials(get_env_variable("ADMIN_INDEX_SECRET_KEY"))
            SimilaritySearchClassification().run(
                item_ids=order.item_ids,
                ai_search_connector=AISearchConnector(creds),
                embedder=embedder
            )
        except Exception as e:
            SQLService.update_task_status(
                status=TaskStatusEnum.error,
                info=str(e),
                task_uuid=task_id
            )
            raise e
        SQLService.update_task_status(
            task_uuid=task_id,
            status=TaskStatusEnum.success,
            info=f"Classifeid {len(order.item_ids)} items."
        )

    background_tasks.add_task(task)

    return {
        "message": f"Similarity Search classification is running. See task {task_id}"
    }
