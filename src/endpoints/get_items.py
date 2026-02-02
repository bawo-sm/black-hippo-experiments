from src.services.sql_service import SQLService
from src.common.schema import Item


class GetItems:

    def run(self, items_ids: list[int]) -> list[Item]:
        records = SQLService.load_items_by_origin_id(items_ids)
        return [Item(**x.to_dict()) for x in records]