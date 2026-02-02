from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from sentence_transformers import SentenceTransformer
from logging import getLogger, basicConfig, INFO
from src.common.schema import (
    SimSearchClassificationRequest, 
    SimSearchClassificationResponse, 
    GetStatusRequest, 
    GetStatusResponse,
    CheckItemsResponse,
    Item,
    CreateReferenceDataRequest,
    CreateReferenceDataResponse
)
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
    return GetItems().run(items_ids)


@app.post(CHECK_ITEMS_URL, response_model=CheckItemsResponse)
async def endpoint_check_items(items_ids: list[int]):
    return CheckItems().run(items_ids)


@app.post(SIM_SEARCH_CLASSIFICATION_URL, response_model=SimSearchClassificationResponse)
async def endpoint_sim_search_classification_items(
    order: SimSearchClassificationRequest,
    background_tasks: BackgroundTasks
):  
    return SimilaritySearchClassification.endpoint(
        embedder=embedder,
        order=order,
        background_tasks=background_tasks
    )


@app.put(CREATE_REFERENCE_DATA_URL, response_model=CreateReferenceDataResponse)
async def enpoint_create_reference_data(
    order: CreateReferenceDataRequest,
    background_tasks: BackgroundTasks
):
    return ReferenceData.endpoint_create(
        embedder=embedder,
        order=order,
        background_tasks=background_tasks
    )


@app.delete(DELETE_REFERENCE_DATA_URL)
async def enpoint_create_reference_data() -> str:
    return ReferenceData.endpoint_delete()