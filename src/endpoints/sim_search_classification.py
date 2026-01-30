from sentence_transformers import SentenceTransformer
from src.ai.vector_db_connector import VectorDBConnector
from src.common.schema import SimSearchDocument, Classification
from src.services.sql_service import SQLService


class SimilaritySearchClassification:
    
    def run(
            self, 
            item_ids: list[int], 
            embedder: SentenceTransformer,
            vector_db_conn: VectorDBConnector
    ):
        documents = self._prepare_documents(item_ids)
        documents = self._generate_embeddings(documents, embedder)
        documents = self._classification(documents, vector_db_conn)
        self._export_results(documents)

    def _prepare_documents(self, item_ids: list[str]) -> list[SimSearchDocument]:
        sql_items = SQLService.load_items_by_origin_id(item_ids)
        return [
            SimSearchDocument(
                doc_id=x.origin_id,
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

    def _classification(self, documents: list[SimSearchDocument], vector_db_conn: VectorDBConnector):
        for i in range(len(documents)):
            records = vector_db_conn.similarity_search(
                embedding=documents[i].embedding,
                limit=1
            )
            documents[i].predicted_class = Classification(
                main=records[0].payload["main"],
                sub=records[0].payload["sub"],
                detail=records[0].payload["detail"],
                level4=records[0].payload["level4"],
            )
        return documents
    
    def _export_results(self, documents: list[SimSearchDocument]):
        for x in documents:
            SQLService.update_item(
                origin_id=x.doc_id,
                kwargs=dict(
                    main=x.predicted_class.main,
                    sub=x.predicted_class.sub,
                    detail=x.predicted_class.detail,
                    level4=x.predicted_class.level4
                )
            )
