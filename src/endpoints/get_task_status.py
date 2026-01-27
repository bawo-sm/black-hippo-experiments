from src.common.schema import GetStatusRequest, GetStatusResponse, TaskStatus
from src.services.sql_service import SQLService


class GetTaskStatus:
    
    def run(self, task_uuid: str | None):
        db_rows = SQLService.load_task_statuses(task_uuid)
        return GetStatusResponse(
            tasks=[
                TaskStatus(
                    task_uuid=x[0].task_uuid,
                    task=x[0].task,
                    status=x[0].status,
                    info=x[0].info,
                    updated_at=x[0].updated_at
                )
                for x in db_rows
            ]
        )
    