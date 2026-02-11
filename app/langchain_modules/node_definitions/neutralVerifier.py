"""Neutral verifier node - re-examines neutral-family primary predictions."""
from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser

from app.langchain_modules.base.abstractColorNode import AbstractColorAgent
from app.data_models.color_detection_state import ColorDetectionState, ColorConfidence
from app.data_models.classification_outputs import NeutralVerifyOutput
from app.langchain_modules.utils.color_priors import (
    normalize_color_name,
    build_prior_hints_block,
)


# The set of neutral colors that trigger this verification step
NEUTRAL_FAMILY = {
    "Natural", "Beige", "Off-white", "Ecru", "Light beige", "Greige",
    "Taupe", "Camel", "Caramel", "Brown", "Dark brown",
}


class NeutralVerifier(AbstractColorAgent):
    """Conditional node: only fires when primary color is in the neutral family.

    Shows the image again with ONLY the neutral options and asks the model
    to compare and choose, reducing confusion between Natural/Beige/Off-white/etc.
    """

    def __init__(self, llm: BaseChatModel):
        output_parser = PydanticOutputParser(pydantic_object=NeutralVerifyOutput)
        super().__init__("NeutralVerifier", llm, output_parser)

    def _build_system_prompt(self, state: ColorDetectionState) -> str:
        current = state.detail_color_1 or "unknown"
        prior_hints = build_prior_hints_block(
            supplier_reference_description=state.supplier_reference_description,
            materials=state.materials,
        )

        return f"""You are a neutral-color specialist for a home & garden product catalog.

The primary color classifier picked: "{current}"
Color description: {state.image_description or "No description available."}

Your job: VERIFY or CORRECT this choice using ONLY the neutral/brown palette below.
Pick the SINGLE BEST match for the product's dominant surface color.

NEUTRAL OPTIONS (pick exactly one):
- "Natural" — unpainted wood, wicker, rattan, bamboo, jute, seagrass, cane, cork. The raw material is visible with no paint or stain.
- "Off-white" — cream, ivory, warm white. Slightly tinted white surfaces (ceramic, fabric, paint).
- "Ecru" — raw unbleached fabric, yellowish off-white linen/cotton.
- "Light beige" — very pale warm tan, lighter than beige.
- "Beige" — warm tan fabric, ceramic, sandstone, linen-colored textile.
- "Greige" — grey-beige blend, modern warm grey.
- "Taupe" — grey-brown, cool-toned mid neutral, stone-colored.
- "Camel" — light warm brown, caramel-toned leather or suede.
- "Caramel" — warm amber-brown, honey-toned.
- "Brown" — medium brown painted/stained wood, leather, fabric.
- "Dark brown" — very dark stained wood, espresso, walnut, dark leather.

KEY DECISION RULES:
1. If you can see raw wood grain, wicker weave, or natural fibers with NO paint/stain → "Natural"
2. If the surface is painted/glazed/coated even if in a wood-like tone → NOT Natural (use Beige/Brown/etc.)
3. Materials hint: if materials say MANGO WOOD, RATTAN, BAMBOO, JUTE etc. → strongly favor "Natural"
4. Cream/ivory ceramic or fabric → "Off-white"
5. Sandy/tan fabric or ceramic → "Beige"
6. Light warm brown leather/suede → "Camel"
7. Dark stained/painted wood → "Brown" or "Dark brown"
{prior_hints}
Respond with ONLY a JSON object:
{{"detail_color": "ColorName", "confidence": 0.85, "reasoning": "brief reason"}}"""

    def _build_user_prompt(self, state: ColorDetectionState) -> str:
        if state.image_data:
            return f'Look at the product image carefully. The initial pick was "{state.detail_color_1}". Is that correct, or should it be a different neutral shade? Return ONLY JSON.'
        else:
            return f'Verify the neutral color "{state.detail_color_1}" based on the description. Return ONLY JSON.'

    def _process_output(self, output: Any, state: ColorDetectionState) -> ColorDetectionState:
        if isinstance(output, NeutralVerifyOutput):
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
                # Verification failed — keep original
                state.add_classification_step(
                    step="neutral_verify",
                    result=f"kept {state.detail_color_1} (parse error)",
                    metadata={"error": "parse_failed"},
                )
                return state
        else:
            color = getattr(output, "detail_color", None)
            confidence = getattr(output, "confidence", 0.0)
            reasoning = getattr(output, "reasoning", None)

        # Normalize
        if color:
            normalized = normalize_color_name(color)
            if normalized and normalized in NEUTRAL_FAMILY:
                color = normalized
            elif color not in NEUTRAL_FAMILY:
                # Invalid — keep original
                state.add_classification_step(
                    step="neutral_verify",
                    result=f"kept {state.detail_color_1} (invalid: {color})",
                    metadata={"attempted": color},
                )
                return state

        old_color = state.detail_color_1
        if color and color != old_color:
            state.detail_color_1 = color
            state.confidence_1 = ColorConfidence(
                detail_color_confidence=min(max(confidence, 0.0), 1.0) if confidence else None,
                reasoning=reasoning,
            )

        state.add_classification_step(
            step="neutral_verify",
            result=f"{old_color} -> {state.detail_color_1}",
            metadata={
                "original": old_color,
                "verified": state.detail_color_1,
                "changed": old_color != state.detail_color_1,
                "confidence": confidence,
                "reasoning": reasoning,
            },
        )

        return state
