"""Sub classifier node - second step in classification hierarchy."""
from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser

from app.langchain_modules.base.abstractGraphNodes import AbstractToolLessAgent
from app.data_models.classification_state import ClassificationState
from app.data_models.classification_outputs import SubClassificationOutput
from app.langchain_modules.utils.hierarchy_loader import (
    load_json_values,
    get_valid_sub_values,
    filter_values_by_hierarchy
)


class SubClassifier(AbstractToolLessAgent):
    """Classifier for sub category classification with hierarchy filtering."""
    
    def __init__(self, llm: BaseChatModel):
        output_parser = PydanticOutputParser(pydantic_object=SubClassificationOutput)
        super().__init__("SubClassifier", llm, output_parser)
        self.all_sub_values = load_json_values("sub_values.json")
    
    def _build_system_prompt(self, state: ClassificationState) -> str:
        """Build system prompt for sub classification with hierarchy filtering."""
        if not state.main:
            return "Error: Main classification is required before sub classification."
        
        # Get valid sub values based on main classification
        valid_sub_values = get_valid_sub_values(state.main)
        
        # If no valid values found (e.g., Unspecified), use all values
        if not valid_sub_values:
            sub_values_to_use = self.all_sub_values
        else:
            # Filter to only show valid sub values
            sub_values_to_use = filter_values_by_hierarchy(self.all_sub_values, valid_sub_values)
        
        sub_values_str = ", ".join(sub_values_to_use)
        
        # Check if "Unspecified" is in the valid values
        has_unspecified = "Unspecified" in sub_values_to_use
        unspecified_note = ""
        if not has_unspecified and len(sub_values_to_use) > 0:
            unspecified_note = "\n\nIMPORTANT: You must select one of the specific categories listed above. 'Unspecified' is NOT available for this main category - you must choose a specific sub category."
        elif has_unspecified:
            unspecified_note = "\n\nNote: Only use 'Unspecified' if none of the specific categories above match. Prefer specific categories when possible."
        
        return f"""Main category: {state.main}. Classify into ONE sub category. Return ONLY valid JSON, no other text.

Valid sub categories for "{state.main}": {sub_values_str}{unspecified_note}

Return JSON format: {{"sub": "category_name"}}

You MUST select from the list above. Choose the most specific category that matches."""
    
    def _build_user_prompt(self, state: ClassificationState) -> str:
        """Build user prompt for sub classification."""
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
            return f"Main: {state.main}. Classify sub category. Item details: {item_data_str}. Return only JSON: {{\"sub\": \"category\"}}"
        elif state.image_data:
            return f"Main: {state.main}. Classify sub category. Return only JSON: {{\"sub\": \"category\"}}"
        elif item_data_str:
            return f"Main: {state.main}. Classify sub category based on item details: {item_data_str}. Return only JSON: {{\"sub\": \"category\"}}"
        else:
            return f"Main: {state.main}. Classify sub category. Return only JSON: {{\"sub\": \"category\"}}"
    
    def _process_output(self, output: Any, state: ClassificationState) -> ClassificationState:
        """Process sub classification output and update state."""
        if not state.main:
            state.add_error("Cannot classify sub category without main classification")
            return state
        
        # Handle both Pydantic output and raw string
        if isinstance(output, SubClassificationOutput):
            sub_value = output.sub
        elif isinstance(output, str):
            import json
            import re
            json_match = re.search(r'\{[^}]*"sub"[^}]*\}', output)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    sub_value = parsed.get("sub", "Unspecified")
                except:
                    sub_value = "Unspecified"
            else:
                sub_value = "Unspecified"
        else:
            sub_value = getattr(output, 'sub', 'Unspecified')
        
        # Get valid sub values for the main category
        valid_sub_values = get_valid_sub_values(state.main)
        
        # If main is Unspecified, allow any sub value
        if state.main == "Unspecified":
            allowed_values = self.all_sub_values
        else:
            # Filter to only valid values
            allowed_values = filter_values_by_hierarchy(self.all_sub_values, valid_sub_values)
            if not allowed_values:
                allowed_values = self.all_sub_values  # Fallback to all if no mapping found
        
        # Validate that the output is in the allowed values
        # Special case: "Unspecified" is always allowed as a fallback, even if not in hierarchy mapping
        if sub_value == "Unspecified":
            # Always allow "Unspecified" - it's a valid fallback category
            pass
        elif sub_value not in allowed_values:
            # For other invalid values, log error and fall back to first valid value or "Unspecified"
            state.add_error(
                f"Invalid sub value: {sub_value} for main: {state.main}. "
                f"Must be one of {allowed_values}. Using fallback."
            )
            # Try to use first valid value, otherwise use "Unspecified"
            if allowed_values:
                sub_value = allowed_values[0]
            else:
                sub_value = "Unspecified"
        
        state.sub = sub_value
        state.add_classification_step(
            step="sub",
            result=sub_value,
            metadata={"main": state.main}
        )
        
        return state
    
    def __call__(self, state) -> dict:
        """Execute the agent node with retry logic for 'Unspecified'."""
        # Handle both dict and ClassificationState inputs
        if isinstance(state, dict):
            state_obj = ClassificationState(**state)
        elif isinstance(state, ClassificationState):
            state_obj = state
        else:
            # Convert other dict-like objects
            state_obj = ClassificationState(**dict(state))
        
        try:
            system_prompt = self._build_system_prompt(state_obj)
            user_prompt = self._build_user_prompt(state_obj)
            
            # Get valid sub values to check if we should retry on "Unspecified"
            valid_sub_values = get_valid_sub_values(state_obj.main) if state_obj.main else []
            if state_obj.main and state_obj.main != "Unspecified":
                allowed_values = filter_values_by_hierarchy(self.all_sub_values, valid_sub_values) if valid_sub_values else self.all_sub_values
                has_specific_options = len([v for v in allowed_values if v != "Unspecified"]) > 0
            else:
                has_specific_options = False
            
            max_retries = 2  # One initial attempt + one retry
            for attempt in range(max_retries):
                output = self._invoke_llm(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_data=state_obj.image_data
                )
                
                # Extract the value to check if it's "Unspecified"
                if isinstance(output, SubClassificationOutput):
                    sub_value = output.sub
                elif isinstance(output, str):
                    import json
                    import re
                    json_match = re.search(r'\{[^}]*"sub"[^}]*\}', output)
                    if json_match:
                        try:
                            parsed = json.loads(json_match.group())
                            sub_value = parsed.get("sub", "Unspecified")
                        except:
                            sub_value = "Unspecified"
                    else:
                        sub_value = "Unspecified"
                else:
                    sub_value = getattr(output, 'sub', 'Unspecified')
                
                # If we got "Unspecified" but there are specific options, retry once
                if sub_value == "Unspecified" and has_specific_options and attempt < max_retries - 1:
                    # Add a note to the prompt to be more specific
                    system_prompt = system_prompt.replace(
                        "Choose the most specific category that matches.",
                        "Choose the most specific category that matches. DO NOT use 'Unspecified' - you must select one of the specific categories listed."
                    )
                    continue
                
                # Process the output
                updated_state = self._process_output(output, state_obj)
                
                # Return as dict for LangGraph compatibility
                return updated_state.model_dump()
        
        except Exception as e:
            error_msg = str(e)
            # For rate limit and system message errors, propagate them up
            if '429' in error_msg or 'rate limit' in error_msg.lower() or 'rate-limited' in error_msg.lower():
                state_obj.add_error(f"{self.name} error: {error_msg}")
                return state_obj.model_dump()
            elif '400' in error_msg or 'Developer instruction' in error_msg:
                state_obj.add_error(f"{self.name} error: {error_msg}")
                return state_obj.model_dump()
            else:
                # Other errors - add to state and continue
                state_obj.add_error(f"{self.name} error: {error_msg}")
                return state_obj.model_dump()

