"""Primary color classifier node - selects the dominant detail color."""
from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser

from app.langchain_modules.base.abstractColorNode import AbstractColorAgent
from app.data_models.color_detection_state import ColorDetectionState, ColorConfidence
from app.data_models.classification_outputs import PrimaryColorOutput
from app.langchain_modules.utils.hierarchy_loader import load_json_values
from app.langchain_modules.utils.color_priors import (
    normalize_color_name,
    build_prior_hints_block,
)

# Organized color groups for prompt clarity
COLOR_GROUPS = {
    "Neutrals": ["White", "Off-white", "Light grey", "Grey", "Dark grey", "Graphite", "Anthracite", "Black"],
    "Beige/Brown": ["Ecru", "Light beige", "Beige", "Greige", "Taupe", "Camel", "Caramel", "Brown", "Dark brown", "Terracotta", "Cognac", "Rust brown"],
    "Red": ["Red", "Dark red", "Christmas red", "Coral red", "Carmine red", "Burgundy", "Marsala", "Neon Red"],
    "Blue": ["Light blue", "Blue", "Dark blue", "Grey blue", "Cobalt blue", "Azure blue", "Turquoise", "Petrol", "Neon Blue"],
    "Yellow": ["Light yellow", "Yellow", "Warm yellow", "Mustard yellow", "Ocher yellow", "Neon Yellow"],
    "Green": ["Light green", "Green", "Dark green", "Grey-green", "Moss green", "Olive", "Khaki green", "Mint green", "Emerald green", "Neon Green"],
    "Purple": ["Light purple", "Purple", "Dark purple", "Lilac", "Lavender", "Mauve", "Violet", "Eggplant", "Neon Purple"],
    "Orange": ["Light orange", "Orange", "Dark orange", "Peach", "Apricot", "Neon Orange"],
    "Pink": ["Light pink", "Pink", "Dark pink", "Old pink", "Salmon pink", "Fuchsia", "Neon Pink"],
    "Metallic": ["Gold", "Champagne", "Silver", "Bronze", "Copper", "Brass", "Rose gold"],
    "Special": ["Natural", "Transparent"],
}


def _format_color_list() -> str:
    """Format colors in grouped layout for prompt."""
    lines = []
    for group, colors in COLOR_GROUPS.items():
        lines.append(f"  {group}: {', '.join(colors)}")
    return "\n".join(lines)


class PrimaryColorClassifier(AbstractColorAgent):
    """Selects the primary (most dominant) detail color for the product."""
    
    def __init__(self, llm: BaseChatModel):
        output_parser = PydanticOutputParser(pydantic_object=PrimaryColorOutput)
        super().__init__("PrimaryColorClassifier", llm, output_parser)
        self.detail_colors = load_json_values("color_detail_values.json")
        self.choosable_colors = [c for c in self.detail_colors if c != "Multi"]
    
    def _build_system_prompt(self, state: ColorDetectionState) -> str:
        grouped_colors = _format_color_list()
        prior_hints = build_prior_hints_block(
            supplier_reference_description=state.supplier_reference_description,
            materials=state.materials,
        )
        
        return f"""You are a product color classification expert for a home & garden catalog.

TASK: Select the single most dominant color of the product from the allowed list below.

CRITICAL - BACKGROUND IS NOT THE PRODUCT:
- The white or grey area behind/around the product is a STUDIO BACKGROUND — ignore it completely.
- Only classify the color of the PRODUCT ITSELF, not the background, floor, table, or shadows.
- If the product looks white, confirm it is actually the product surface, not just background showing through.

COLOR DESCRIPTION FROM PREVIOUS STEP:
{state.image_description or "No description available."}

ALLOWED DETAIL COLORS (grouped by family):
{grouped_colors}

IMPORTANT RULES:
1. Pick EXACTLY ONE color from the list above. Never invent new color names.
2. Choose the color covering the LARGEST visible area of the product surface.
3. For unpainted wood, wicker, rattan, jute, or seagrass products, use "Natural".
4. For colored glass or colored transparent plastic (e.g. green glass vase, amber bottle), use the GLASS COLOR (Green, Turquoise, Orange, etc.), NOT "Transparent". Only use "Transparent" for truly clear/colorless glass.
5. For metallic finishes: use Gold, Silver, Bronze, Copper, Brass, or Rose gold.
6. NEUTRAL DISAMBIGUATION — distinguish carefully:
   - "Natural" = unpainted wood, wicker, rattan, bamboo, jute, seagrass, cane, cork
   - "Beige" = warm tan fabric, ceramic, sandstone, linen-colored textile
   - "Off-white" = slightly warm white, cream, ivory, bone-colored ceramic
   - "Ecru" = raw/unbleached fabric tone, yellowish off-white
   - "Brown" = painted/stained dark wood, dark leather, chocolate-toned
   - "Dark brown" = very dark stained wood, espresso, walnut-stained
   - "Camel" = light warm brown, caramel-toned leather or suede
   - "Taupe" = grey-brown, cool-toned mid neutral
7. If the product description explicitly mentions a color and it matches the allowed list, prefer it.
{prior_hints}
EXAMPLES:
- Rattan basket -> {{"detail_color": "Natural", "confidence": 0.95, "reasoning": "unpainted rattan weave"}}
- Mango wood side table -> {{"detail_color": "Natural", "confidence": 0.93, "reasoning": "unpainted mango wood"}}
- White ceramic plate with gold rim -> {{"detail_color": "White", "confidence": 0.90, "reasoning": "white ceramic is dominant surface"}}
- Green glass vase -> {{"detail_color": "Green", "confidence": 0.90, "reasoning": "colored green glass"}}
- Beige linen cushion -> {{"detail_color": "Beige", "confidence": 0.85, "reasoning": "beige fabric"}}
- Cream ceramic pot -> {{"detail_color": "Off-white", "confidence": 0.85, "reasoning": "cream/ivory ceramic"}}
- Dark stained wooden shelf -> {{"detail_color": "Dark brown", "confidence": 0.85, "reasoning": "dark stained wood"}}
- Light brown suede pouf -> {{"detail_color": "Camel", "confidence": 0.85, "reasoning": "light warm brown suede"}}
- Jute rug -> {{"detail_color": "Natural", "confidence": 0.93, "reasoning": "natural jute fiber"}}
- Grey-brown stone vase -> {{"detail_color": "Taupe", "confidence": 0.80, "reasoning": "grey-brown stone"}}

Respond with ONLY a single JSON object. No markdown, no code blocks, no explanation.
{{"detail_color": "ColorName", "confidence": 0.85, "reasoning": "brief reason"}}"""
    
    def _build_user_prompt(self, state: ColorDetectionState) -> str:
        if state.image_data:
            return 'Look at the PRODUCT in the image (ignore the white/grey studio background). Select its single most dominant color from the allowed list. Return ONLY JSON.'
        else:
            return 'Select the most likely dominant color based on the description. Return ONLY JSON.'
    
    def _process_output(self, output: Any, state: ColorDetectionState) -> ColorDetectionState:
        if isinstance(output, PrimaryColorOutput):
            color = output.detail_color
            confidence = output.confidence
            reasoning = output.reasoning
        elif isinstance(output, str):
            try:
                parsed = self._extract_json(output)
                color = parsed.get("detail_color")
                confidence = parsed.get("confidence", 0.0)
                reasoning = parsed.get("reasoning")
            except Exception:
                color = None
                confidence = 0.0
                reasoning = None
        else:
            color = getattr(output, 'detail_color', None)
            confidence = getattr(output, 'confidence', 0.0)
            reasoning = getattr(output, 'reasoning', None)
        
        # Normalize using alias mapping before validation
        if color:
            normalized = normalize_color_name(color)
            if normalized:
                color = normalized
            elif color not in self.choosable_colors:
                match = next((c for c in self.choosable_colors if c.lower() == color.lower()), None)
                if match:
                    color = match
                else:
                    state.add_error(f"PrimaryColorClassifier: Invalid color '{color}'.")
                    color = None
                    confidence = 0.0
        
        state.detail_color_1 = color
        state.confidence_1 = ColorConfidence(
            detail_color_confidence=min(max(confidence, 0.0), 1.0) if confidence else None,
            reasoning=reasoning
        )
        
        state.add_classification_step(
            step="primary_color",
            result=color or "None",
            metadata={"confidence": confidence, "reasoning": reasoning}
        )
        
        return state
