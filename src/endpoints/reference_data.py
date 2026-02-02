from uuid import uuid4
from fastapi import BackgroundTasks
from sentence_transformers import SentenceTransformer
from src.ai.vector_db_connector import VectorDBConnector
from src.common.schema import (
    CreateReferenceDataRequest, 
    CreateReferenceDataResponse,
    QdrantRecord
)
from src.services.sql_service import SQLService
from src.common.db_schema import SQLTaskStatus, TaskEnum, TaskStatusEnum
from src.settings import SIM_SEARCH_INDEX_NAME


class ReferenceData:

    @staticmethod
    def endpoint_create(
        embedder: SentenceTransformer, 
        order: CreateReferenceDataRequest,
        background_tasks: BackgroundTasks
    ) -> CreateReferenceDataResponse:
        # set status
        task_id = str(uuid4())
        SQLService.set_task_status(
            SQLTaskStatus(
                task_uuid=task_id,
                task=TaskEnum.create_reference_data,
                status=TaskStatusEnum.in_progress
            )
        )

        # define & run task
        def task():
            try:
                embeddings = embedder.encode([x.text() for x in order.items])
                records = [
                    QdrantRecord(
                        embedding=[float(x) for x in embeddings[i]], 
                        metadata=order.items[i].metadata()
                    )
                    for i in range(len(order.items))
                ]
                VectorDBConnector().upsert_records(records)
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
                info=f"Saved {len(order.items)} as reference data"
            )

        background_tasks.add_task(task)


        # return immediate response
        return CreateReferenceDataResponse(
            message=f"Creating reference data is running",
            task_id=task_id
        )

    @staticmethod
    def endpoint_delete():
        VectorDBConnector().delete_collection(collection_name=SIM_SEARCH_INDEX_NAME)
        return f"Collection with reference data is successfully deleted"
