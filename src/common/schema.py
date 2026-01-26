from pydantic import BaseModel


class Classification(BaseModel):
    main: str | None
    sub: str | None
    detail: str | None
    level4: str | None


class SimSearchDocument(BaseModel):
    doc_id: int
    image_link: str
    season: str | None = None
    supplier_name: str | None = None
    supplier_reference_description: str | None = None
    materials: str | None = None
    image_description: str | None = None
    embedding: list[float] | None
    predicted_class: Classification | None = None
    
    def product_representation(self):
        return f"""
        Product for season: {self.season}
        Producer: {self.supplier_name}
        Product name: {self.supplier_reference_description}
        Materials: {self.materials}
        Image description: {self.image_description}
        """


class SimSearchClassificationRequest(BaseModel):
    documents: list[SimSearchDocument]
