from typing import Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter, 
    FieldCondition, 
    MatchAny
)
from uuid import uuid4
from logging import getLogger
from src.common.schema import QdrantRecord
from src.common.utils import get_env_variable
from src.settings import SIM_SEARCH_INDEX_NAME, EMBEDDING_SIZE


logger = getLogger("VectorDBConnector")


class VectorDBConnector:
    """
    This class handles connection and actions on Qdrant Vector DB
    """

    embedding_size = 768
    embedding_field = "embedding"

    def __init__(self):
        self.__client = QdrantClient(
            host=get_env_variable('VECTOR_DB_HOST'),
            port=get_env_variable('VECTOR_DB_PORT'),
        )
        try:
            self.__client.get_collections()
        except Exception as e:
            logger.error(f"Error in initializing Qdrant client: {e}")
            raise e


    def create_colection(self, collection_name: str = SIM_SEARCH_INDEX_NAME, distance: str = Distance.COSINE):
        self.__client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_SIZE,
                distance=distance
            )
        )

    def recreate_colection(self, collection_name: str = SIM_SEARCH_INDEX_NAME, distance: str = Distance.COSINE):
        self.__client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_SIZE,
                distance=distance
            )
        )

    def list_collections(self):
        return self.__client.get_collections()
    
    def upsert_records(self, records: list[QdrantRecord], collection_name: str = SIM_SEARCH_INDEX_NAME):
        points = [
            PointStruct(
                id=str(uuid4()),
                vector=x.embedding,
                payload=x.metadata
            )
            for x in records
        ]
        logger.info(f"Upserting {len(points)} records")
        self.__client.upsert(
            collection_name=collection_name,
            points=points
        )

    def similarity_search(
        self, 
        embedding: list[float],
        limit: int,
        items_ids: list[int] | None = None,
        collection_name: str = SIM_SEARCH_INDEX_NAME
    ):
        if items_ids:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="item_id",
                        match=MatchAny(any=items_ids)
                    )
                ]
            )
        else:
            query_filter = None

        return (
            self.__client
            .query_points(
                collection_name=collection_name,
                query=embedding,
                limit=limit,
                query_filter=query_filter
            )
            .points
        )
