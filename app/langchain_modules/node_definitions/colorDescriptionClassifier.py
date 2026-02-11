"""Color description node - generates a color-focused image description."""
from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser

from app.langchain_modules.base.abstractColorNode import AbstractColorAgent
from app.data_models.color_detection_state import ColorDetectionState
from app.data_models.classification_outputs import ColorDescriptionOutput
from app.langchain_modules.utils.color_priors import build_prior_hints_block


class ColorDescriptionClassifier(AbstractColorAgent):
    """Generates a compact, color-focused description of the product image."""
    
    def __init__(self, llm: BaseChatModel):
        output_parser = PydanticOutputParser(pydantic_object=ColorDescriptionOutput)
        super().__init__("ColorDescriptionClassifier", llm, output_parser)
    
    def _build_system_prompt(self, state: ColorDetectionState) -> str:
        prior_hints = build_prior_hints_block(
            supplier_reference_description=state.supplier_reference_description,
            materials=state.materials,
        )
        
        return f"""You are a color analysis expert for home, furniture, and decorative products.

TASK:
Analyze the provided product image and produce a compact color-focused description.

CRITICAL - IGNORE THE BACKGROUND:
- Product photos typically have a WHITE or GREY studio background. This is NOT part of the product.
- Do NOT include the background color, floor, wall, table surface, or shadows in your analysis.
- Only describe colors that are physically part of the product itself.

INSTRUCTIONS:
1. List the dominant colors visible ON THE PRODUCT ITSELF, in order of visual prominence.
2. Mention material finishes that influence perceived color (glossy, matte, metallic, transparent, wooden).
3. Note any patterns, gradients, or multi-color aspects.
4. Estimate the total number of DISTINCT colors on the product.
5. For colored glass (green glass, amber glass, blue glass), describe the glass COLOR, not "transparent".
6. Keep the description under 80 words.

COLOR-COUNTING RULES (be conservative):
- Shades of the same hue = 1 color (light brown + dark brown = 1; gold + copper tones on one metallic item = 1).
- Wood grain variation, stone veins, and metallic sheen on a single base = 1 color, NOT multiple.
- Natural wood/rattan/wicker/bamboo/jute = 1 color ("natural wood tone").
- Only count 2+ when there are CLEARLY DIFFERENT hues (e.g. blue AND red, or black AND gold).
- A white/grey studio background is NOT a product color — never count it.
{prior_hints}
Respond with ONLY a single JSON object. No markdown, no code blocks, no explanation.
{{"description": "...", "estimated_color_count": N}}"""
    
    def _build_user_prompt(self, state: ColorDetectionState) -> str:
        if state.image_data:
            return 'Analyze ONLY THE PRODUCT in the image (ignore white/grey background). Describe its colors and estimate distinct color count. Return ONLY JSON: {"description": "...", "estimated_color_count": N}'
        else:
            return 'No image provided. Return: {"description": "No image available for color analysis", "estimated_color_count": 0}'
    
    def _process_output(self, output: Any, state: ColorDetectionState) -> ColorDetectionState:
        if isinstance(output, ColorDescriptionOutput):
            description = output.description
            estimated_count = output.estimated_color_count
        elif isinstance(output, str):
            try:
                parsed = self._extract_json(output)
                description = parsed.get("description", "")
                estimated_count = parsed.get("estimated_color_count", 0)
            except Exception:
                description = output
                estimated_count = 0
        else:
            description = getattr(output, 'description', str(output))
            estimated_count = getattr(output, 'estimated_color_count', 0)
        
        state.image_description = description
        state.estimated_color_count = estimated_count
        state.add_classification_step(
            step="color_description",
            result=description,
            metadata={"estimated_color_count": estimated_count}
        )
        
        return state
