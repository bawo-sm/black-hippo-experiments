"""Level4 classifier node - final step in classification hierarchy."""
from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser

from app.langchain_modules.base.abstractGraphNodes import AbstractToolLessAgent
from app.data_models.classification_state import ClassificationState
from app.data_models.classification_outputs import Level4ClassificationOutput
from app.langchain_modules.utils.hierarchy_loader import load_json_values


class Level4Classifier(AbstractToolLessAgent):
    """Classifier for level4 category classification with hierarchy filtering."""
    
    def __init__(self, llm: BaseChatModel):
        output_parser = PydanticOutputParser(pydantic_object=Level4ClassificationOutput)
        super().__init__("Level4Classifier", llm, output_parser)
        self.all_level4_values = load_json_values("level4_values.json")
    
    def _build_system_prompt(self, state: ClassificationState) -> str:
        """Build system prompt for level4 classification - free form, no hierarchy constraints."""
        if not state.main or not state.sub or not state.detail:
            return "Error: Main, sub, and detail classifications are required before level4 classification."
        
        # Use all level4 values - no hierarchy filtering
        level4_values_str = ", ".join(self.all_level4_values)
        
        return f"""You are an assistant that helps with product categorization. Products are typically related to house, home, and garden items.

TASK:
1. Analyze the given product (described in the user message, which may include an image and/or product details).
2. Find the best level4 category from the available categories below, given the main, sub, and detail categories that were already selected.

HIGHER-CLASS CONTEXT:
The general classification is: {state.main}/{state.sub}/{state.detail}
You are now selecting the level4 category (most specific level) within this hierarchy.

AVAILABLE LEVEL4 CATEGORIES:
{level4_values_str}

OUTPUT FORMAT:
Return ONLY valid JSON, no other text. Use this format:
{{"level4": "category_name"}}

You MUST select from the categories listed above. Choose the most specific category that matches the product."""
    
    def _build_user_prompt(self, state: ClassificationState) -> str:
        """Build user prompt for level4 classification."""
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

The classification so far is: {state.main}/{state.sub}/{state.detail}
Analyze the product (image provided above) and classify it into the level4 category. Return only JSON: {{"level4": "category_name"}}"""
            else:
                return f"""The classification so far is: {state.main}/{state.sub}/{state.detail}
Analyze the product image (provided above) and classify it into the level4 category. Return only JSON: {{"level4": "category_name"}}"""
        else:
            return f"""PRODUCT INFORMATION:
{product_info}

The classification so far is: {state.main}/{state.sub}/{state.detail}
Analyze the product information above and classify it into the level4 category. Return only JSON: {{"level4": "category_name"}}"""
    
    def _process_output(self, output: Any, state: ClassificationState) -> ClassificationState:
        """Process level4 classification output and update state."""
        if not state.main or not state.sub or not state.detail:
            state.add_error("Cannot classify level4 category without main, sub, and detail classifications")
            return state
        
        # Handle both Pydantic output and raw string
        if isinstance(output, Level4ClassificationOutput):
            level4_value = output.level4
        elif isinstance(output, str):
            import json
            import re
            json_match = re.search(r'\{[^}]*"level4"[^}]*\}', output)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    level4_value = parsed.get("level4", None)
                except:
                    level4_value = None
            else:
                level4_value = None
        else:
            level4_value = getattr(output, 'level4', None)
        
        # Validate that the output is in the allowed values
        if not level4_value or level4_value not in self.all_level4_values:
            if level4_value:
                state.add_error(f"Invalid level4 value: {level4_value}. Must be one of {self.all_level4_values}")
            # Use first valid value as fallback
            if self.all_level4_values:
                level4_value = self.all_level4_values[0]
            else:
                level4_value = None
        
        state.level4 = level4_value
        state.add_classification_step(
            step="level4",
            result=level4_value,
            metadata={"main": state.main, "sub": state.sub, "detail": state.detail}
        )
        
        return state

