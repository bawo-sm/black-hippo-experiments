from typing import Any
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    HnswParameters
)


class AISearchConnector:
    """
    This class handles connection and actions on Azure AI Search
    """

    embedding_size = 768
    endpoint = "https://<your-search-service>.search.windows.net"
    embedding_field = "embedding"

    def __init__(self, credentials: AzureKeyCredential):
        self.__credentials = credentials
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=credentials
        )

    def create_index(self, index_name: str):
        index_schema = self.__get_index_schema(index_name)
        self.index_client.create_index(index_schema)

    def upload_documents(self, index_name: str, documents: list[dict[str, Any]]):
        search_client = self.__get_search_client(index_name)
        search_client.upload_documents(documents)

    # def text_search(self, index_name: str):
    #     pass

    def similarity_search(
            self, 
            index_name: str, 
            query_embedding: list[float],
            top_k: int = 1,
            search_text: str | None = None
    ):
        search_client = self.__get_search_client(index_name)
        return search_client.search(
            search_text=search_text,
            vector=query_embedding,
            top_k=top_k,
            vector_fields=self.embedding_field
        )

    def delete_index(self, index_name: str):
        self.index_client.delete_index(index_name)

    def __get_search_client(self, index_name: str) -> SearchClient:
        return SearchClient(
            endpoint=self.endpoint,
            index_name=index_name,
            credential=self.__credentials
        )

    def __get_index_schema(self, index_name: str):
        return SearchIndex(
            name=index_name,
            fields=[
                SimpleField(
                    name="id",
                    type="Edm.String",
                    key=True
                ),
                SimpleField(
                    name="season",
                    type="Edm.String"
                ),
                SearchableField(
                    name="supplier_name",
                    type="Edm.String"
                ),
                SearchableField(
                    name="supplier_reference_description",
                    type="Edm.String"
                ),
                SearchableField(
                    name="materials",
                    type="Edm.String"
                ),
                SearchableField(
                    name="image_description",
                    type="Edm.String"
                ),
                SearchableField(
                    name="image_link",
                    type="Edm.String"
                ),
                SearchableField(
                    name="main",
                    type="Edm.String"
                ),
                SearchableField(
                    name="sub",
                    type="Edm.String"
                ),
                SearchableField(
                    name="detail",
                    type="Edm.String"
                ),
                SearchableField(
                    name="level4",
                    type="Edm.String"
                ),
                SearchableField(
                    name=self.embedding_field,
                    vector_search_dimensions=self.embedding_size,
                    vector_search_configuration="vector-config"
                )
            ],
            vector_search=VectorSearch(
                algorithm_configurations=[
                    VectorSearchAlgorithmConfiguration(
                        name="vector-config",
                        kind="hnsw",
                        hnsw_parameters=HnswParameters(
                            metric="cosine",
                            m=4,
                            ef_construction=400,
                            ef_search=500
                        )
                    )
                ]
            )
        )