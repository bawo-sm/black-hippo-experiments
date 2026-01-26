from pydantic import BaseModel


class ExampleModel(BaseModel):
    record_id: int
    image_path: str
    