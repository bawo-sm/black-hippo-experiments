from pydantic import BaseModel
from datetime import datetime
from src.common.enums import TaskStatusEnum, TaskEnum


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