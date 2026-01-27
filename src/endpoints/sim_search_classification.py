import pprint
from sentence_transformers import SentenceTransformer
from src.ai import AISearchConnector
from src.common.schema import SimSearchClassificationRequest, SimSearchDocument, Classification
from src.settings import SIM_SEARCH_INDEX_NAME, RESULTS_INDEX_NAME
from src.services.sql_service import SQLService


class SimilaritySearchClassification:
    
    def run(
            self, 
            item_ids: list[int], 
            embedder: SentenceTransformer,
            ai_search_connector: AISearchConnector
    ):
        documents = self._prepare_documents(item_ids)
        documents = self._generate_embeddings(documents, embedder)
        documents = self._classification(documents, ai_search_connector)
        self._export_results(documents, ai_search_connector)

    def _prepare_documents(self, item_ids: list[str]) -> list[SimSearchDocument]:
        sql_items = SQLService.load_items_by_origin_id(item_ids)
        return [
            SimSearchDocument(
                doc_id=x.id,
                season=x.season,
                supplier_name=x.supplier_name,
                supplier_reference_description=x.supplier_reference_description,
                materials=x.materials
            )
            for x in sql_items
        ]

    def _generate_embeddings(self, documents: list[SimSearchDocument], embedder: SentenceTransformer):
        for i in range(len(documents)):
            documents[i].embedding = embedder.encode(documents[i].product_representation())
        return documents

    def _classification(self, documents: list[SimSearchDocument], ai_search_connector: AISearchConnector):
        for i in range(len(documents)):
            record = ai_search_connector.similarity_search(
                index_name=SIM_SEARCH_INDEX_NAME,
                query_embedding=documents[i].embedding,
                top_k=1
            )
            pprint.pp(record)
            documents[i].predicted_class = Classification(
                main=record["main"],
                sub=record["sub"],
                detail=record["detail"],
                level4=record["level4"],
            )
        return documents
    
    def _export_results(self, documents: list[SimSearchDocument], ai_search_connector: AISearchConnector):
        ai_search_connector.upload_documents(
            index_name=RESULTS_INDEX_NAME,
            documents=[x.model_dump() for x in documents]
        )