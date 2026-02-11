"""Main color mapper node - deterministically maps detail colors to main colors.

Also applies post-processing corrections:
- Transparent override for colored glass when text hints indicate a hue.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.data_models.color_detection_state import ColorDetectionState, ColorConfidence
from app.langchain_modules.utils.color_priors import extract_text_color_hints


# Path to mapping file
RESOURCES_DIR = Path(__file__).parent.parent.parent.parent / "resources"

# Colors that indicate the glass/plastic is actually colored, not transparent
CHROMATIC_COLORS = {
    "Red", "Dark red", "Christmas red", "Coral red", "Carmine red", "Burgundy", "Marsala",
    "Light blue", "Blue", "Dark blue", "Grey blue", "Cobalt blue", "Azure blue", "Turquoise", "Petrol",
    "Light yellow", "Yellow", "Warm yellow", "Mustard yellow", "Ocher yellow",
    "Light green", "Green", "Dark green", "Grey-green", "Moss green", "Olive", "Khaki green",
    "Mint green", "Emerald green",
    "Light purple", "Purple", "Dark purple", "Lilac", "Lavender", "Mauve", "Violet", "Eggplant",
    "Light orange", "Orange", "Dark orange", "Peach", "Apricot",
    "Light pink", "Pink", "Dark pink", "Old pink", "Salmon pink", "Fuchsia",
    "Gold", "Bronze", "Copper", "Brass", "Rose gold",
    "Brown", "Dark brown", "Camel", "Caramel", "Terracotta", "Cognac", "Rust brown",
}


def _load_color_mapping() -> Dict[str, Any]:
    """Load color main mapping from JSON."""
    filepath = RESOURCES_DIR / "color_main_mapping.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Color mapping file not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


class MainColorMapper:
    """Deterministically maps detail colors to their corresponding main colors.
    
    Also applies post-processing corrections:
    - If primary is "Transparent" but text hints mention a chromatic color,
      override to that color (colored glass correction).
    """
    
    def __init__(self):
        self.name = "MainColorMapper"
        self.color_mapping = _load_color_mapping()
    
    def _map_detail_to_main(self, detail_color: Optional[str]) -> Optional[str]:
        """Look up the main color for a given detail color."""
        if not detail_color:
            return None
        entry = self.color_mapping.get(detail_color)
        if entry:
            return entry.get("main_color")
        return None
    
    def _apply_transparent_correction(self, state_obj: ColorDetectionState) -> None:
        """
        If primary color is 'Transparent' but text mentions a chromatic color,
        override primary to that color (colored glass/plastic case).
        """
        if state_obj.detail_color_1 != "Transparent":
            return
        
        # Check text hints for chromatic colors
        text_hints = extract_text_color_hints(state_obj.supplier_reference_description)
        chromatic_hints = [h for h in text_hints if h in CHROMATIC_COLORS]
        
        if chromatic_hints:
            # Override: the glass/plastic is colored, not truly transparent
            new_color = chromatic_hints[0]
            old_conf = state_obj.confidence_1
            state_obj.detail_color_1 = new_color
            state_obj.confidence_1 = ColorConfidence(
                detail_color_confidence=old_conf.detail_color_confidence * 0.85 if old_conf and old_conf.detail_color_confidence else 0.7,
                reasoning=f"Corrected Transparent -> {new_color} based on text hint '{state_obj.supplier_reference_description}'"
            )
            state_obj.add_classification_step(
                step="transparent_correction",
                result=f"Transparent -> {new_color}",
                metadata={"text_hints": text_hints, "chromatic_hints": chromatic_hints}
            )
    
    def __call__(self, state) -> dict:
        """Map detail colors to main colors, with post-processing corrections."""
        if isinstance(state, dict):
            state_obj = ColorDetectionState(**state)
        elif isinstance(state, ColorDetectionState):
            state_obj = state
        else:
            state_obj = ColorDetectionState(**dict(state))
        
        try:
            # Apply transparent correction before mapping
            self._apply_transparent_correction(state_obj)
            
            # Map each detail color slot
            for i in range(1, 4):
                detail_color = getattr(state_obj, f"detail_color_{i}")
                main_color = self._map_detail_to_main(detail_color)
                setattr(state_obj, f"main_color_{i}", main_color)
                
                # Update confidence
                conf = getattr(state_obj, f"confidence_{i}")
                if conf and detail_color:
                    if main_color:
                        conf.main_color_confidence = conf.detail_color_confidence
                    else:
                        conf.main_color_confidence = 0.0
                        state_obj.add_error(
                            f"MainColorMapper: No main color mapping for '{detail_color}'"
                        )
            
            state_obj.add_classification_step(
                step="main_color_mapping",
                result=f"main_1={state_obj.main_color_1}, main_2={state_obj.main_color_2}, main_3={state_obj.main_color_3}",
                metadata={
                    "main_color_1": state_obj.main_color_1,
                    "main_color_2": state_obj.main_color_2,
                    "main_color_3": state_obj.main_color_3,
                }
            )
            
        except Exception as e:
            state_obj.add_error(f"MainColorMapper error: {str(e)}")
        
        return state_obj.model_dump()
