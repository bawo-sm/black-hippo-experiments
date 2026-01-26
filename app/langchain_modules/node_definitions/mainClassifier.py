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
        
        # Check if "Unspecified" is in the valid values
        has_unspecified = "Unspecified" in self.main_values
        unspecified_note = ""
        if has_unspecified:
            unspecified_note = "\n\nNote: Only use 'Unspecified' if none of the specific categories above match. Prefer specific categories when possible."
        
        return f"""Classify into ONE main category. Return ONLY valid JSON, no other text.

Valid main categories: {main_values_str}{unspecified_note}

Return JSON format: {{"main": "category_name"}}

You MUST select from the list above. Choose the most specific category that matches."""
    
    def _build_user_prompt(self, state: ClassificationState) -> str:
        """Build user prompt for main classification."""
        # Build item data context if available
        item_data_parts = []
        if state.supplier_name:
            item_data_parts.append(f"Supplier: {state.supplier_name}")
        if state.supplier_reference_description:
            item_data_parts.append(f"Description: {state.supplier_reference_description}")
        if state.materials:
            item_data_parts.append(f"Materials: {state.materials}")
        
        item_data_str = ", ".join(item_data_parts) if item_data_parts else None
        
        # Build prompt based on what's available
        if state.image_data and item_data_str:
            return f"Classify this image. Item details: {item_data_str}. Return only JSON: {{\"main\": \"category\"}}"
        elif state.image_data:
            return "Classify this image. Return only JSON: {\"main\": \"category\"}"
        elif item_data_str:
            return f"Classify based on item details: {item_data_str}. Return only JSON: {{\"main\": \"category\"}}"
        else:
            return "Classify. Return only JSON: {\"main\": \"category\"}"
    
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
                    main_value = parsed.get("main", "Unspecified")
                except:
                    main_value = "Unspecified"
            else:
                main_value = "Unspecified"
        else:
            main_value = getattr(output, 'main', 'Unspecified')
        
        # Validate that the output is in the allowed values
        if main_value not in self.main_values:
            state.add_error(f"Invalid main value: {main_value}. Must be one of {self.main_values}")
            main_value = "Unspecified"
        
        state.main = main_value
        state.add_classification_step(
            step="main",
            result=main_value,
            metadata={}
        )
        
        return state

