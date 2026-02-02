from src.services.sql_service import SQLService
from src.services.blob_service import BlobService
from src.common.schema import CheckItemsResponse, CheckedItem
from src.settings import IMAGES_CONTAINER



class CheckItems:

    def run(self, items_ids: list[int]) -> CheckItemsResponse:
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