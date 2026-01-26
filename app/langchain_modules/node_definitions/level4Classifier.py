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
        
        return f"""Main: {state.main}, Sub: {state.sub}, Detail: {state.detail}. Classify into ONE level4 category. Return ONLY valid JSON, no other text.

Valid level4 categories: {level4_values_str}

Return JSON format: {{"level4": "category_name"}}

Select from the list above. Use "Unspecified" if unclear."""
    
    def _build_user_prompt(self, state: ClassificationState) -> str:
        """Build user prompt for level4 classification."""
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
            return f"Classify level4 category. Item details: {item_data_str}. Return only JSON: {{\"level4\": \"category\"}}"
        elif state.image_data:
            return f"Classify level4 category. Return only JSON: {{\"level4\": \"category\"}}"
        elif item_data_str:
            return f"Classify level4 category based on item details: {item_data_str}. Return only JSON: {{\"level4\": \"category\"}}"
        else:
            return f"Classify level4 category. Return only JSON: {{\"level4\": \"category\"}}"
    
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
                    level4_value = parsed.get("level4", "Unspecified")
                except:
                    level4_value = "Unspecified"
            else:
                level4_value = "Unspecified"
        else:
            level4_value = getattr(output, 'level4', 'Unspecified')
        
        # No validation - accept whatever the model returns for level4
        # If empty or None, default to "Unspecified"
        if not level4_value or not level4_value.strip():
            level4_value = "Unspecified"
        
        state.level4 = level4_value
        state.add_classification_step(
            step="level4",
            result=level4_value,
            metadata={"main": state.main, "sub": state.sub, "detail": state.detail}
        )
        
        return state

