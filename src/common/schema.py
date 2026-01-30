from pydantic import BaseModel
from datetime import datetime
from src.common.enums import TaskStatusEnum, TaskEnum


class Item(BaseModel):
    id: int
    origin_id: int
    season: str
    supplier_name: str
    supplier_reference_description: str
    materials: str | None
    main: str | None
    sub: str | None
    detail: str | None
    level4: str | None
    colors: str | None
    hs_code: str | None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class Classification(BaseModel):
    main: str | None
    sub: str | None
    detail: str | None
    level4: str | None


class SimSearchDocument(BaseModel):
    doc_id: int
    season: str | None = None
    supplier_name: str | None = None
    supplier_reference_description: str | None = None
    materials: str | None = None
    image_description: str | None = None
    embedding: list[float] | None = None
    predicted_class: Classification | None = None
    
    def product_representation(self):
        return f"""
        Product for season: {self.season}
        Producer: {self.supplier_name}
        Product name: {self.supplier_reference_description}
        Materials: {self.materials}
        Image description: {self.image_description}
        """


class SimSearchClassificationRequest(BaseModel):
    item_ids: list[int]


class SimSearchClassificationResponse(BaseModel):
    message: str
    task_id: str


class GetStatusRequest(BaseModel):
    task_id: str | None = None


class TaskStatus(BaseModel):
    task_uuid: str
    task: TaskEnum
    status: TaskStatusEnum
    info: str | None = None
    updated_at: datetime


class GetStatusResponse(BaseModel):
    tasks: list[TaskStatus]


class QdrantRecord(BaseModel):
    metadata: dict[str, str | int | float | bool]
    embedding: list[float]


class CheckItemsRequest(BaseModel):
    items_ids: list[int]


class CheckedItem(BaseModel):
    item_id: int
    in_sql_db: bool
    in_blob_storage: bool


class CheckItemsResponse(BaseModel):
    items: list[CheckedItem]
