from fastapi import FastAPI
from src.endpoints import ClassifyItemsEndpoint
from .settings import *


app = FastAPI(
    title="ItemsClassification"
)


@app.get(HOME_ENDPOINT)
async def endpoint_home():
    return "Hello bro!"


@app.post(CLASSIFY_ITEMS_ENDPOINT)
async def endpoint_classify_items():
    pass
