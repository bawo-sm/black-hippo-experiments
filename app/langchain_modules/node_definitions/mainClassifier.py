"""Main classifier node - first step in classification hierarchy."""
from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser

from app.langchain_modules.base.abstractGraphNodes import AbstractToolLessAgent
from app.data_models.classification_state import ClassificationState
from app.data_models.classification_outputs import MainClassificationOutput
from app.langchain_modules.utils.hierarchy_loader import load_json_values


class MainClassifier(AbstractToolLessAgent):
    """Classifier for main category classification."""
    
    def __init__(self, llm: BaseChatModel):
        output_parser = PydanticOutputParser(pydantic_object=MainClassificationOutput)
        super().__init__("MainClassifier", llm, output_parser)
        self.main_values = load_json_values("main_values.json")
    
    def _build_system_prompt(self, state: ClassificationState) -> str:
        """Build system prompt for main classification."""
        main_values_str = ", ".join(self.main_values)
        
        return f"""You are an assistant that helps with product categorization. Products are typically related to house, home, and garden items.

TASK:
1. Analyze the given product (described in the user message, which may include an image and/or product details).
2. Find the best main category from the available categories below.

HIGHER-CLASS CONTEXT:
This is the highest general level classification. You are selecting the main category for this product.

AVAILABLE MAIN CATEGORIES:
{main_values_str}

OUTPUT FORMAT:
Return ONLY valid JSON, no other text. Use this format:
{{"main": "category_name"}}

You MUST select from the categories listed above. Choose the most specific category that matches the product."""
    
    def _build_user_prompt(self, state: ClassificationState) -> str:
        """Build user prompt for main classification."""
        # Build structured product information
        product_parts = []
        if state.supplier_name:
            product_parts.append(f"Supplier name = {state.supplier_name}")
        if state.supplier_reference_description:
            product_parts.append(f"Product name = {state.supplier_reference_description}")
        if state.materials:
            product_parts.append(f"Materials = {state.materials}")
        
        product_info = "\n".join(product_parts) if product_parts else "Product information not available"
        
        # Build prompt based on what's available
        if state.image_data:
            if product_parts:
                return f"""PRODUCT INFORMATION:
{product_info}

Analyze the product (image provided above) and classify it into the main category. Return only JSON: {{"main": "category_name"}}"""
            else:
                return """Analyze the product image (provided above) and classify it into the main category. Return only JSON: {"main": "category_name"}"""
        else:
            return f"""PRODUCT INFORMATION:
{product_info}

Analyze the product information above and classify it into the main category. Return only JSON: {{"main": "category_name"}}"""
    
    def _process_output(self, output: Any, state: ClassificationState) -> ClassificationState:
        """Process main classification output and update state."""
        # Handle both Pydantic output and raw string
        if isinstance(output, MainClassificationOutput):
            main_value = output.main
        elif isinstance(output, str):
            # Try to extract JSON from string response
            import json
            import re
            # Look for JSON in the response
            json_match = re.search(r'\{[^}]*"main"[^}]*\}', output)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    main_value = parsed.get("main", None)
                except:
                    main_value = None
            else:
                main_value = None
        else:
            main_value = getattr(output, 'main', None)
        
        # Validate that the output is in the allowed values
        if not main_value or main_value not in self.main_values:
            if main_value:
                state.add_error(f"Invalid main value: {main_value}. Must be one of {self.main_values}")
            # Use first valid value as fallback
            if self.main_values:
                main_value = self.main_values[0]
            else:
                main_value = None
        
        state.main = main_value
        state.add_classification_step(
            step="main",
            result=main_value,
            metadata={}
        )
        
        return state

