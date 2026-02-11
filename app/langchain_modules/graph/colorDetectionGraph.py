"""LangGraph builder for color detection workflow (v4 — dual model, multi gate, neutral verify)."""
from typing import TypedDict, Optional, Dict, Any, List
import time
from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel

from app.data_models.color_detection_state import ColorDetectionState
from app.langchain_modules.node_definitions.colorDescriptionClassifier import ColorDescriptionClassifier
from app.langchain_modules.node_definitions.primaryColorClassifier import PrimaryColorClassifier
from app.langchain_modules.node_definitions.multiGateClassifier import MultiGateClassifier
from app.langchain_modules.node_definitions.neutralVerifier import NeutralVerifier, NEUTRAL_FAMILY
from app.langchain_modules.node_definitions.secondaryColorsClassifier import SecondaryColorsClassifier
from app.langchain_modules.node_definitions.mainColorMapper import MainColorMapper


class ColorDetectionStateDict(TypedDict, total=False):
    """TypedDict version of ColorDetectionState for LangGraph compatibility."""
    image_data: Optional[str]
    supplier_reference_description: Optional[str]
    materials: Optional[str]
    image_description: Optional[str]
    estimated_color_count: Optional[int]
    detail_color_1: Optional[str]
    detail_color_2: Optional[str]
    detail_color_3: Optional[str]
    main_color_1: Optional[str]
    main_color_2: Optional[str]
    main_color_3: Optional[str]
    is_multi: bool
    confidence_1: Optional[Dict[str, Any]]
    confidence_2: Optional[Dict[str, Any]]
    confidence_3: Optional[Dict[str, Any]]
    classification_history: List[Dict[str, Any]]
    errors: List[str]


def delay_node(state: ColorDetectionStateDict) -> ColorDetectionStateDict:
    """Delay node to stagger API requests between LLM nodes."""
    time.sleep(1.0)
    return state


# ── Routing functions for conditional edges ────────────────────────────────────

def route_after_multi_gate(state: ColorDetectionStateDict) -> str:
    """Route after multi-gate: if multi, skip to mapper; else continue to neutral check."""
    is_multi = state.get("is_multi", False)
    if is_multi:
        return "map_main_colors"
    return "check_neutral"


def route_after_neutral_check(state: ColorDetectionStateDict) -> str:
    """Route after primary: if neutral, go to verifier; else skip to secondary."""
    detail_1 = state.get("detail_color_1")
    if detail_1 and detail_1 in NEUTRAL_FAMILY:
        return "neutral_verify"
    return "delay_3"


class ColorDetectionGraphBuilder:
    """Builder for the color detection graph (v4 — dual model architecture).

    Graph flow:
        describe_image (vision) -> delay_1
        -> primary_color (vision) -> delay_2
        -> multi_gate (fast)
            ├─ is_multi=true  -> map_main_colors -> END
            └─ is_multi=false -> check_neutral
                ├─ neutral primary -> neutral_verify (fast) -> delay_3
                └─ non-neutral     -> delay_3
            -> secondary_colors (fast) -> map_main_colors -> END
    """
    
    def __init__(
        self,
        llm_vision: BaseChatModel,
        llm_fast: Optional[BaseChatModel] = None,
    ):
        """
        Initialize the graph builder with dual models.
        
        Args:
            llm_vision: Strong vision model (gpt-4o) for description + primary color.
            llm_fast: Fast model (gpt-4o-mini) for multi-gate, secondary, neutral verify.
                      If None, uses llm_vision for all nodes (single-model fallback).
        """
        self.llm_vision = llm_vision
        self.llm_fast = llm_fast or llm_vision
        self.graph = None
    
    def build(self) -> StateGraph:
        """Build and compile the color detection graph."""
        # Create nodes with appropriate models
        describe_image = ColorDescriptionClassifier(self.llm_vision)
        primary_color = PrimaryColorClassifier(self.llm_vision)
        multi_gate = MultiGateClassifier(self.llm_fast)
        neutral_verify = NeutralVerifier(self.llm_fast)
        secondary_colors = SecondaryColorsClassifier(self.llm_fast)
        map_main_colors = MainColorMapper()
        
        # Create graph
        workflow = StateGraph(ColorDetectionStateDict)
        
        # Add nodes
        workflow.add_node("describe_image", describe_image)
        workflow.add_node("primary_color", primary_color)
        workflow.add_node("multi_gate", multi_gate)
        workflow.add_node("neutral_verify", neutral_verify)
        workflow.add_node("secondary_colors", secondary_colors)
        workflow.add_node("map_main_colors", map_main_colors)
        
        # Delay nodes between LLM calls
        workflow.add_node("delay_1", delay_node)
        workflow.add_node("delay_2", delay_node)
        workflow.add_node("delay_3", delay_node)
        
        # ── Wire edges ────────────────────────────────────────────────────
        # Vision pipeline: describe -> primary
        workflow.set_entry_point("describe_image")
        workflow.add_edge("describe_image", "delay_1")
        workflow.add_edge("delay_1", "primary_color")
        workflow.add_edge("primary_color", "delay_2")
        
        # Multi gate
        workflow.add_edge("delay_2", "multi_gate")
        
        # After multi gate: branch
        workflow.add_conditional_edges(
            "multi_gate",
            route_after_multi_gate,
            {
                "map_main_colors": "map_main_colors",
                "check_neutral": "check_neutral",
            },
        )
        
        # Neutral check: virtual routing node (zero-cost)
        # We use a pass-through node to inspect state and route
        workflow.add_node("check_neutral", lambda state: state)
        workflow.add_conditional_edges(
            "check_neutral",
            route_after_neutral_check,
            {
                "neutral_verify": "neutral_verify",
                "delay_3": "delay_3",
            },
        )
        
        # After neutral verify, continue to delay_3
        workflow.add_edge("neutral_verify", "delay_3")
        
        # Secondary colors -> mapper -> END
        workflow.add_edge("delay_3", "secondary_colors")
        workflow.add_edge("secondary_colors", "map_main_colors")
        workflow.add_edge("map_main_colors", END)
        
        # Compile
        self.graph = workflow.compile()
        return self.graph
    
    def detect_colors(
        self,
        image_data: Optional[str] = None,
        supplier_reference_description: Optional[str] = None,
        materials: Optional[str] = None,
    ) -> ColorDetectionState:
        """
        Run color detection on an image.
        
        Args:
            image_data: Base64 encoded image, image URL, or data URI
            supplier_reference_description: Product name/description (optional hint)
            materials: Materials string (optional hint)
            
        Returns:
            ColorDetectionState with results
        """
        if self.graph is None:
            self.build()
        
        initial_state: ColorDetectionStateDict = {
            "image_data": image_data,
            "supplier_reference_description": supplier_reference_description,
            "materials": materials,
            "image_description": None,
            "estimated_color_count": None,
            "detail_color_1": None,
            "detail_color_2": None,
            "detail_color_3": None,
            "main_color_1": None,
            "main_color_2": None,
            "main_color_3": None,
            "is_multi": False,
            "confidence_1": None,
            "confidence_2": None,
            "confidence_3": None,
            "classification_history": [],
            "errors": []
        }
        
        # Run graph
        result_dict = self.graph.invoke(initial_state)
        
        # Convert to ColorDetectionState
        if isinstance(result_dict, ColorDetectionState):
            result = result_dict
        elif isinstance(result_dict, dict):
            result = ColorDetectionState(**result_dict)
        else:
            result = ColorDetectionState(**dict(result_dict))
        
        # Check for critical errors
        if result.errors:
            error_str = '; '.join(result.errors)
            if '429' in error_str or 'rate limit' in error_str.lower():
                raise Exception(f"Rate limit error: {error_str}")
            if '400' in error_str or 'Developer instruction' in error_str:
                raise Exception(f"System message error: {error_str}")
        
        return result
