import requests
import json
from qdrant_client.models import Distance
from uuid import uuid4
from logging import getLogger
from src.common.schema import QdrantRecord
from src.common.utils import get_env_variable
from src.settings import SIM_SEARCH_INDEX_NAME, EMBEDDING_SIZE


logger = getLogger("VectorDBConnector")


class VectorDBConnector:
    """
    This class handles connection and actions on Qdrant Vector DB.
    Uses native requests package.
    """

    def create_collection(self, collection_name: str = SIM_SEARCH_INDEX_NAME, distance: str = Distance.COSINE):
        url = f"{get_env_variable('QDRANT_URL')}/collections/{collection_name}"
        headers = {
            "api-key": get_env_variable('QDRANT_API_KEY'),
            "Content-Type": "application/json"
        }
        data=json.dumps(
            {
                "vectors": {
                    "size": EMBEDDING_SIZE,
                    "distance": distance
                }
            }
        )
        response = requests.put(
            url=url,
            headers=headers,
            data=data
        )

        if response.status_code == 200:
            logger.info(f"Collection {collection_name} has been created.")
        else:
            logger.error(f"Error in creating collection {collection_name} ({response.status_code}): {response.text}")
            assert 0
    
    def delete_collection(self, collection_name: str):
        url = f"{get_env_variable('QDRANT_URL')}/collections/{collection_name}"
        headers = {
            "api-key": get_env_variable('QDRANT_API_KEY'),
            "Content-Type": "application/json"
        }
        response = requests.delete(
            url=url,
            headers=headers,
        )

        if response.status_code == 200:
            logger.info(f"Collection {collection_name} has been deleted.")
        else:
            logger.error(f"Error in deleting collection {collection_name} ({response.status_code}): {response.text}")

    def list_collections(self):
        url = f"{get_env_variable('QDRANT_URL')}/collections"
        headers = {
            "api-key": get_env_variable('QDRANT_API_KEY'),
            "Content-Type": "application/json"
        }
        response = requests.get(
            url=url,
            headers=headers,
        )

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error in listing collections ({response.status_code}): {response.text}")
    
    def upsert_records(self, records: list[QdrantRecord], collection_name: str = SIM_SEARCH_INDEX_NAME):
        url = f"{get_env_variable('QDRANT_URL')}/collections/{collection_name}/points"
        headers = {
            "api-key": get_env_variable('QDRANT_API_KEY'),
            "Content-Type": "application/json"
        }
        start, stop = 0, 100
        while True:
            data = json.dumps({
                "points": [
                    {
                        "id": str(uuid4()),
                        "vector": x.embedding,
                        "payload": x.metadata
                    }
                    for x in records[start:min([stop, len(records)])]
                ]
            })
            response = requests.put(
                url=url,
                headers=headers,
                data=data
            )
            assert response.status_code == 200, response.text

            start += 100
            stop += 100
            if start >= len(records):
                break

    def similarity_search(
        self, 
        embedding: list[float],
        limit: int,
        items_ids: list[int] | None = None,
        collection_name: str = SIM_SEARCH_INDEX_NAME
    ):
        url = f"{get_env_variable('QDRANT_URL')}/collections/{collection_name}/points/search"
        headers = {
            "api-key": get_env_variable('QDRANT_API_KEY'),
            "Content-Type": "application/json"
        }
        data_dict = {
            "vector": embedding,
            "limit": limit,
            "with_payload": True
        }

        if items_ids:
            query_filter = {
                "must": [
                    {
                        "key": "item_id",
                        "match": {
                            "any": items_ids
                        }
                    }
                ]
            }
            data_dict["filter"] = query_filter

        response = requests.post(
            url=url,
            headers=headers,
            data=json.dumps(data_dict)
        )
        assert response.status_code == 200, response.text

        return response.json()["result"]
