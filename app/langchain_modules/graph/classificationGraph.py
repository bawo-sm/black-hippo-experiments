"""LangGraph builder for classification workflow."""
from typing import TypedDict, Optional, Dict, Any, List
import time
from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel

from app.data_models.classification_state import ClassificationState
from app.langchain_modules.node_definitions.mainClassifier import MainClassifier
from app.langchain_modules.node_definitions.subClassifier import SubClassifier
from app.langchain_modules.node_definitions.detailClassifier import DetailClassifier
from app.langchain_modules.node_definitions.level4Classifier import Level4Classifier


class ClassificationStateDict(TypedDict, total=False):
    """TypedDict version of ClassificationState for LangGraph compatibility."""
    image_data: Optional[str]
    supplier_name: Optional[str]
    supplier_reference_description: Optional[str]
    materials: Optional[str]
    main: Optional[str]
    sub: Optional[str]
    detail: Optional[str]
    level4: Optional[str]
    classification_history: List[Dict[str, Any]]
    errors: List[str]


def delay_node(state: ClassificationStateDict) -> ClassificationStateDict:
    """Delay node to stagger API requests between classifiers."""
    time.sleep(1.0)  # 1 second delay between each classifier step
    return state


class ClassificationGraphBuilder:
    """Builder for the classification graph."""
    
    def __init__(self, llm: BaseChatModel):
        """
        Initialize the graph builder.
        
        Args:
            llm: LangChain-compatible LLM client
        """
        self.llm = llm
        self.graph = None
    
    def build(self) -> StateGraph:
        """
        Build the classification graph.
        
        Returns:
            Compiled LangGraph StateGraph
        """
        # Create classifier nodes
        main_classifier = MainClassifier(self.llm)
        sub_classifier = SubClassifier(self.llm)
        detail_classifier = DetailClassifier(self.llm)
        level4_classifier = Level4Classifier(self.llm)
        
        # Create graph with TypedDict for LangGraph compatibility
        workflow = StateGraph(ClassificationStateDict)
        
        # Add classifier nodes
        workflow.add_node("main_classifier", main_classifier)
        workflow.add_node("sub_classifier", sub_classifier)
        workflow.add_node("detail_classifier", detail_classifier)
        workflow.add_node("level4_classifier", level4_classifier)
        
        # Add delay nodes to stagger API requests
        workflow.add_node("delay_1", delay_node)
        workflow.add_node("delay_2", delay_node)
        workflow.add_node("delay_3", delay_node)
        
        # Add edges with delays between classifiers
        workflow.set_entry_point("main_classifier")
        workflow.add_edge("main_classifier", "delay_1")
        workflow.add_edge("delay_1", "sub_classifier")
        workflow.add_edge("sub_classifier", "delay_2")
        workflow.add_edge("delay_2", "detail_classifier")
        workflow.add_edge("detail_classifier", "delay_3")
        workflow.add_edge("delay_3", "level4_classifier")
        workflow.add_edge("level4_classifier", END)
        
        # Compile graph
        self.graph = workflow.compile()
        return self.graph
    
    def classify(
        self,
        image_data: Optional[str] = None,
        supplier_name: Optional[str] = None,
        supplier_reference_description: Optional[str] = None,
        materials: Optional[str] = None
    ) -> ClassificationState:
        """
        Run classification on an image and/or item metadata.
        
        Args:
            image_data: Base64 encoded image or image URL (optional - can be None for metadata-only classification)
            supplier_name: Supplier name (optional)
            supplier_reference_description: Supplier reference description (optional)
            materials: Materials information (optional)
            
        Returns:
            ClassificationState with results
        """
        if self.graph is None:
            self.build()
        
        # Create initial state as dict (LangGraph expects dict-like objects)
        initial_state_dict: ClassificationStateDict = {
            "image_data": image_data,
            "supplier_name": supplier_name,
            "supplier_reference_description": supplier_reference_description,
            "materials": materials,
            "main": None,
            "sub": None,
            "detail": None,
            "level4": None,
            "classification_history": [],
            "errors": []
        }
        
        # Run graph - invoke returns a dict-like object
        result_dict = self.graph.invoke(initial_state_dict)
        
        # Convert dict to ClassificationState
        # Handle both dict and AddableValuesDict types
        if isinstance(result_dict, ClassificationState):
            result = result_dict
        elif isinstance(result_dict, dict):
            result = ClassificationState(**result_dict)
        else:
            # If it's an AddableValuesDict or similar, convert to dict first
            result = ClassificationState(**dict(result_dict))
        
        # Check for critical errors and raise exception if needed
        if result.errors:
            error_str = '; '.join(result.errors)
            # Check if it's a rate limit error - these should be retried
            if '429' in error_str or 'rate limit' in error_str.lower() or 'rate-limited' in error_str.lower():
                raise Exception(f"Rate limit error: {error_str}")
            # Check if it's a system message error - these should be retried
            if '400' in error_str or 'Developer instruction' in error_str:
                raise Exception(f"System message error: {error_str}")
            # If main classification failed, that's critical
            if not result.main and any('MainClassifier' in e for e in result.errors):
                raise Exception(f"Main classification failed: {error_str}")
        
        return result

