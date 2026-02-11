"""Secondary colors classifier node - selects colors 2 and 3 (multi already decided upstream)."""
from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser

from app.langchain_modules.base.abstractColorNode import AbstractColorAgent
from app.data_models.color_detection_state import ColorDetectionState, ColorConfidence
from app.data_models.classification_outputs import SecondaryColorsOutput
from app.langchain_modules.utils.hierarchy_loader import load_json_values
from app.langchain_modules.utils.color_priors import (
    normalize_color_name,
    build_prior_hints_block,
)
from app.langchain_modules.node_definitions.primaryColorClassifier import _format_color_list


class SecondaryColorsClassifier(AbstractColorAgent):
    """Selects secondary detail colors (2 and 3). Multi is already decided upstream by MultiGateClassifier."""
    
    def __init__(self, llm: BaseChatModel):
        output_parser = PydanticOutputParser(pydantic_object=SecondaryColorsOutput)
        super().__init__("SecondaryColorsClassifier", llm, output_parser)
        self.detail_colors = load_json_values("color_detail_values.json")
        self.choosable_colors = [c for c in self.detail_colors if c != "Multi"]
    
    def _build_system_prompt(self, state: ColorDetectionState) -> str:
        grouped_colors = _format_color_list()
        primary = state.detail_color_1 or "unknown"
        prior_hints = build_prior_hints_block(
            supplier_reference_description=state.supplier_reference_description,
            materials=state.materials,
        )
        
        return f"""You are a product color classification expert for a home & garden catalog.

TASK: Determine if the product has additional colors beyond the primary color.
The product is NOT multicolored — that was already checked. Focus only on finding 0, 1, or 2 secondary colors.

CRITICAL - BACKGROUND IS NOT THE PRODUCT:
- The white or grey area around the product is a STUDIO BACKGROUND — do NOT count it as a product color.
- Shadows, reflections on the background, and the surface the product sits on are NOT product colors.

COLOR DESCRIPTION:
{state.image_description or "No description available."}

PRIMARY COLOR ALREADY ASSIGNED: {primary}

ALLOWED DETAIL COLORS (grouped by family):
{grouped_colors}

RULES:
1. Only add a secondary color if it covers a SIGNIFICANT visible area of the product (roughly >15% of product surface).
2. Do NOT count the studio background (white/grey) as a product color.
3. Do NOT count shades of the same hue as separate colors (light brown trim on brown product = still one color).
4. Do NOT repeat the primary color "{primary}".
5. Small accents, logos, labels, textures, and shadows do NOT count as separate colors.
6. For colored glass, the glass color is the product color, not "Transparent".
7. Colors must be from the allowed list.
{prior_hints}
EXAMPLES:
- Single-color brown table -> {{"detail_color_2": null, "confidence_2": null, "reasoning_2": null, "detail_color_3": null, "confidence_3": null, "reasoning_3": null}}
- White vase with gold rim -> {{"detail_color_2": "Gold", "confidence_2": 0.85, "reasoning_2": "gold metallic rim is significant accent", "detail_color_3": null, "confidence_3": null, "reasoning_3": null}}
- Natural wood tray with iron handles -> {{"detail_color_2": "Black", "confidence_2": 0.80, "reasoning_2": "black iron handles", "detail_color_3": null, "confidence_3": null, "reasoning_3": null}}

Respond with ONLY a single JSON object. No markdown, no code blocks, no explanation.
{{"detail_color_2": "ColorName or null", "confidence_2": 0.8, "reasoning_2": "reason", "detail_color_3": "ColorName or null", "confidence_3": 0.7, "reasoning_3": "reason"}}"""
    
    def _build_user_prompt(self, state: ColorDetectionState) -> str:
        if state.image_data:
            return 'Look at the PRODUCT in the image (ignore white/grey studio background). Are there significant secondary colors beyond the primary? Return ONLY JSON.'
        else:
            return 'Determine secondary colors based on the description. Return ONLY JSON.'
    
    def _process_output(self, output: Any, state: ColorDetectionState) -> ColorDetectionState:
        # Parse output
        if isinstance(output, SecondaryColorsOutput):
            color_2 = output.detail_color_2
            conf_2 = output.confidence_2
            reasoning_2 = output.reasoning_2
            color_3 = output.detail_color_3
            conf_3 = output.confidence_3
            reasoning_3 = output.reasoning_3
        elif isinstance(output, str):
            try:
                parsed = self._extract_json(output)
                color_2 = parsed.get("detail_color_2")
                conf_2 = parsed.get("confidence_2")
                reasoning_2 = parsed.get("reasoning_2")
                color_3 = parsed.get("detail_color_3")
                conf_3 = parsed.get("confidence_3")
                reasoning_3 = parsed.get("reasoning_3")
            except Exception:
                color_2 = color_3 = None
                conf_2 = conf_3 = None
                reasoning_2 = reasoning_3 = None
        else:
            color_2 = getattr(output, 'detail_color_2', None)
            conf_2 = getattr(output, 'confidence_2', None)
            reasoning_2 = getattr(output, 'reasoning_2', None)
            color_3 = getattr(output, 'detail_color_3', None)
            conf_3 = getattr(output, 'confidence_3', None)
            reasoning_3 = getattr(output, 'reasoning_3', None)
        
        # ── Normalize and validate color 2 ────────────────────────────────
        color_2 = self._normalize_color(color_2)
        if color_2 and color_2 not in self.choosable_colors:
            state.add_error(f"SecondaryColorsClassifier: Invalid color_2 '{color_2}'.")
            color_2 = None
            conf_2 = None
        if color_2 and color_2 == state.detail_color_1:
            color_2 = None
            conf_2 = None
        
        # ── Normalize and validate color 3 ────────────────────────────────
        color_3 = self._normalize_color(color_3)
        if color_3 and color_3 not in self.choosable_colors:
            state.add_error(f"SecondaryColorsClassifier: Invalid color_3 '{color_3}'.")
            color_3 = None
            conf_3 = None
        if color_3 and (color_3 == state.detail_color_1 or color_3 == color_2):
            color_3 = None
            conf_3 = None
        
        # Shift color_3 up if color_2 is empty
        if not color_2 and color_3:
            color_2 = color_3
            conf_2 = conf_3
            reasoning_2 = reasoning_3
            color_3 = None
            conf_3 = None
            reasoning_3 = None
        
        state.detail_color_2 = color_2
        state.detail_color_3 = color_3
        
        if color_2:
            state.confidence_2 = ColorConfidence(
                detail_color_confidence=min(max(conf_2, 0.0), 1.0) if conf_2 else None,
                reasoning=reasoning_2
            )
        
        if color_3:
            state.confidence_3 = ColorConfidence(
                detail_color_confidence=min(max(conf_3, 0.0), 1.0) if conf_3 else None,
                reasoning=reasoning_3
            )
        
        state.add_classification_step(
            step="secondary_colors",
            result=f"color_2={color_2}, color_3={color_3}",
            metadata={
                "color_2": color_2,
                "color_3": color_3,
                "confidence_2": conf_2,
                "confidence_3": conf_3,
            }
        )
        
        return state
    
    @staticmethod
    def _normalize_color(raw: Any) -> str | None:
        """Normalize a raw color value through alias mapping."""
        if raw is None:
            return None
        if not isinstance(raw, str):
            raw = str(raw)
        if raw.lower() in ("null", "none", ""):
            return None
        return normalize_color_name(raw)
