"""Multi-gate classifier node - binary yes/no multi detection."""
from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser

from app.langchain_modules.base.abstractColorNode import AbstractColorAgent
from app.data_models.color_detection_state import ColorDetectionState, ColorConfidence
from app.data_models.classification_outputs import MultiGateOutput
from app.langchain_modules.utils.color_priors import build_prior_hints_block


class MultiGateClassifier(AbstractColorAgent):
    """Binary classifier: is this product genuinely multicolored (4+ distinct colors)?

    Runs AFTER primary color selection. If multi=true, the pipeline skips
    secondary color selection and sets detail_color_1='Multi' directly.
    """

    def __init__(self, llm: BaseChatModel):
        output_parser = PydanticOutputParser(pydantic_object=MultiGateOutput)
        super().__init__("MultiGateClassifier", llm, output_parser)

    def _build_system_prompt(self, state: ColorDetectionState) -> str:
        est_count = state.estimated_color_count or 0
        primary = state.detail_color_1 or "unknown"
        prior_hints = build_prior_hints_block(
            supplier_reference_description=state.supplier_reference_description,
            materials=state.materials,
        )

        return f"""You are a product color classification expert. Your ONLY job is to decide:
Does this product have 4 or more TRULY DISTINCT, prominent colors?

COLOR DESCRIPTION: {state.image_description or "No description available."}
ESTIMATED COLOR COUNT: {est_count}
PRIMARY COLOR: {primary}

ANSWER is_multi=true ONLY when:
- The product is genuinely multicolored — patchwork, rainbow, multi-stripe, printed pattern with 4+ distinct hues.
- Each color covers a meaningful area of the product (not tiny accents).

ANSWER is_multi=false for ALL of these (common traps):
- 1-3 color products, even with small accents or trims
- Gold/copper/bronze tonal variation on one metallic item (1 metallic color)
- Wood grain, stone veins, or wicker texture variation (1 natural color)
- A product with a different-color lid, handle, or base (2-3 colors, NOT multi)
- Studio background (white/grey) creating an illusion of extra colors
- Estimated color count of 3 or less

IMPORTANT: Most products are NOT multi. When in doubt, answer false.
{prior_hints}
Respond with ONLY a JSON object:
{{"is_multi": false, "reasoning": "brief reason"}}"""

    def _build_user_prompt(self, state: ColorDetectionState) -> str:
        if state.image_data:
            return "Look at the product image. Is this product genuinely multicolored (4+ distinct prominent colors)? Return ONLY JSON."
        else:
            return "Based on the description, is this product multicolored? Return ONLY JSON."

    def _process_output(self, output: Any, state: ColorDetectionState) -> ColorDetectionState:
        if isinstance(output, MultiGateOutput):
            is_multi = output.is_multi
            reasoning = output.reasoning
        elif isinstance(output, str):
            try:
                parsed = self._extract_json(output)
                is_multi = parsed.get("is_multi", False)
                reasoning = parsed.get("reasoning")
            except Exception:
                is_multi = False
                reasoning = None
        else:
            is_multi = getattr(output, "is_multi", False)
            reasoning = getattr(output, "reasoning", None)

        # ── Hard code guards ──────────────────────────────────────────────
        est_count = state.estimated_color_count or 0

        # Block multi if description saw <= 3 colors
        if is_multi and est_count <= 3:
            is_multi = False
            reasoning = f"Overridden: est_count={est_count} <= 3"

        # Block multi for single-tone primary materials
        SINGLE_TONE = {
            "Natural", "Gold", "Silver", "Bronze", "Copper", "Brass",
            "Champagne", "Rose gold", "Black", "White",
        }
        if is_multi and state.detail_color_1 in SINGLE_TONE:
            is_multi = False
            reasoning = f"Overridden: primary '{state.detail_color_1}' is single-tone"

        if is_multi:
            state.is_multi = True
            state.detail_color_1 = "Multi"
            state.detail_color_2 = None
            state.detail_color_3 = None
            state.confidence_1 = ColorConfidence(
                detail_color_confidence=1.0,
                reasoning=reasoning or "Multi-gate: 4+ distinct prominent colors",
            )
            state.confidence_2 = None
            state.confidence_3 = None
        else:
            state.is_multi = False

        state.add_classification_step(
            step="multi_gate",
            result="multi" if is_multi else "not_multi",
            metadata={
                "is_multi": is_multi,
                "estimated_color_count": est_count,
                "reasoning": reasoning,
            },
        )

        return state
