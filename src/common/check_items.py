from src.services.sql_service import SQLService
from src.services.blob_service import BlobService
from src.common.db_schema import SQLItem, SQLTaskStatus, TaskEnum, TaskStatusEnum
from src.settings import IMAGES_CONTAINER


def check_items(task_id: str, items_ids: list[int]):
    for x in items_ids:
        if not SQLService.check_item_exists(x):
            info = f"Cannot find item {x} in the SQL db"
            SQLService.set_task_status(
                SQLTaskStatus(
                    task_uuid=task_id,
                    task=TaskEnum.classification,
                    status=TaskStatusEnum.error,
                    info=info
                )
            )
            raise AssertionError(info) 

        if not BlobService().check_file_exists(
            container_name=IMAGES_CONTAINER,
            file_name=str(x)+".jpg"
        ):
            info = f"Cannot find iamge for item {x} in the Blob storage"
            SQLService.set_task_status(
                SQLTaskStatus(
                    task_uuid=task_id,
                    task=TaskEnum.classification,
                    status=TaskStatusEnum.error,
                    info=info
                )
            )
            raise AssertionError(info) 