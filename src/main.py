from dotenv import load_dotenv
from uuid import uuid4
from fastapi import FastAPI, BackgroundTasks
from sentence_transformers import SentenceTransformer
from logging import getLogger, basicConfig, INFO
from src.ai.vector_db_connector import VectorDBConnector
from src.common.schema import (
    SimSearchClassificationRequest, 
    SimSearchClassificationResponse, 
    GetStatusRequest, 
    GetStatusResponse,
    CheckItemsResponse,
    CheckedItem, 
    CheckItemsRequest,
    Item
)
from src.common.db_schema import SQLTaskStatus, TaskEnum, TaskStatusEnum
from src.common.utils import get_env_variable
from src.common.check_items import check_items
from src.services.sql_service import SQLService
from src.services.blob_service import BlobService
from src.endpoints import *
from src.settings import *


load_dotenv(".env")

basicConfig(level=INFO)
logger = getLogger("Main")

logger.info("Loading embedder")
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


@app.post(GET_ITEMS_URL)
async def endpoint_get_items(items_ids: list[int]) -> list[Item]:
    records = SQLService.load_items_by_origin_id(items_ids)
    return [Item(**x.to_dict()) for x in records]


@app.post(CHECK_ITEMS_URL, response_model=CheckItemsResponse)
async def endpoint_check_items(items_ids: list[int]):
    items = []
    for x in items_ids:
        items.append(
            CheckedItem(
                item_id=x,
                in_sql_db=SQLService.check_item_exists(x),
                in_blob_storage=BlobService().check_file_exists(
                    container_name=IMAGES_CONTAINER,
                    file_name=str(x)+".jpg"
                )
            )
        )
    return CheckItemsResponse(items=items)


@app.post(SIM_SEARCH_CLASSIFICATION_URL, response_model=SimSearchClassificationResponse)
async def endpoint_sim_search_classification_items(
    order: SimSearchClassificationRequest,
    background_tasks: BackgroundTasks
):  
    # set status
    task_id = str(uuid4())
    SQLService.set_task_status(
        SQLTaskStatus(
            task_uuid=task_id,
            task=TaskEnum.classification,
            status=TaskStatusEnum.in_progress
        )
    )


    check_items(task_id, order.item_ids)


    # define & run task
    def task():
        try:
            SimilaritySearchClassification().run(
                item_ids=order.item_ids,
                vector_db_conn=VectorDBConnector(),
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


    # return immediate response
    return SimSearchClassificationResponse(
        message=f"Similarity Search classification is running",
        task_id=task_id
    )
