"""Abstract base classes for graph nodes."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

from app.data_models.classification_state import ClassificationState


class AbstractNode(ABC):
    """Abstract base class for graph nodes."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def __call__(self, state: ClassificationState) -> ClassificationState:
        """
        Execute the node logic.
        
        Args:
            state: Current classification state
            
        Returns:
            Updated classification state
        """
        pass


class AbstractToolLessAgent(AbstractNode):
    """Abstract base class for tool-less agent nodes using LLM."""
    
    def __init__(
        self,
        name: str,
        llm: BaseChatModel,
        output_parser: Optional[PydanticOutputParser] = None
    ):
        super().__init__(name)
        self.llm = llm
        self.output_parser = output_parser
    
    def _create_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        image_data: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> list[BaseMessage]:
        """
        Create messages for LLM with optional image.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            image_data: Optional base64 encoded image or image URL
            model_name: Optional model name to check if system messages are supported
            
        Returns:
            List of messages for LLM
        """
        messages = []
        
        # Check if model supports system messages
        # Some models like Gemma don't support system messages
        supports_system = True
        if model_name:
            # Models that don't support system messages
            no_system_models = ['gemma', 'gemini']
            if any(no_sys in model_name.lower() for no_sys in no_system_models):
                supports_system = False
        
        # Combine system and user prompts if system messages not supported
        if system_prompt and supports_system:
            from langchain_core.messages import SystemMessage
            messages.append(SystemMessage(content=system_prompt))
            final_user_prompt = user_prompt
        elif system_prompt:
            # Combine system and user prompts into user message
            final_user_prompt = f"{system_prompt}\n\n{user_prompt}"
        else:
            final_user_prompt = user_prompt
        
        # Create user message with optional image
        if image_data:
            # For vision models, include image in content
            # Format depends on the model (OpenRouter uses OpenAI format)
            # Check if image_data is base64 or URL
            if image_data.startswith("http://") or image_data.startswith("https://"):
                # URL format
                image_url = image_data
            elif image_data.startswith("data:image"):
                # Already a data URI
                image_url = image_data
            else:
                # Assume base64, create data URI
                image_url = f"data:image/jpeg;base64,{image_data}"
            
            # Create content with text and image
            content = [
                {"type": "text", "text": final_user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
            ]
            
            messages.append(HumanMessage(content=content))
        else:
            messages.append(HumanMessage(content=final_user_prompt))
        
        return messages
    
    def _invoke_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        image_data: Optional[str] = None
    ) -> Any:
        """
        Invoke LLM with prompts and optional image.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            image_data: Optional base64 encoded image
            
        Returns:
            LLM response
        """
        # Get model name from LLM if available
        model_name = None
        if hasattr(self.llm, 'model_name'):
            model_name = self.llm.model_name
        elif hasattr(self.llm, 'model'):
            model_name = self.llm.model
        
        messages = self._create_messages(system_prompt, user_prompt, image_data, model_name)
        response = self.llm.invoke(messages)
        
        if self.output_parser:
            try:
                return self.output_parser.parse(response.content)
            except Exception as e:
                # If parsing fails, try to extract JSON from response
                import json
                import re
                content = response.content
                # Look for JSON object in response
                json_match = re.search(r'\{[^}]*\}', content, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                        # Try to create output object from parsed JSON
                        return self.output_parser.pydantic_object(**parsed)
                    except:
                        pass
                # If all else fails, return raw content for manual parsing
                return response.content
        
        return response.content
    
    @abstractmethod
    def _build_system_prompt(self, state: ClassificationState) -> str:
        """Build the system prompt for this classifier."""
        pass
    
    @abstractmethod
    def _build_user_prompt(self, state: ClassificationState) -> str:
        """Build the user prompt for this classifier."""
        pass
    
    @abstractmethod
    def _process_output(self, output: Any, state: ClassificationState) -> ClassificationState:
        """Process LLM output and update state."""
        pass
    
    def __call__(self, state) -> dict:
        """Execute the agent node."""
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
            
            output = self._invoke_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_data=state_obj.image_data
            )
            
            updated_state = self._process_output(output, state_obj)
            
            # Return as dict for LangGraph compatibility
            return updated_state.model_dump()
        
        except Exception as e:
            error_msg = str(e)
            # For rate limit and system message errors, propagate them up
            if '429' in error_msg or 'rate limit' in error_msg.lower() or 'rate-limited' in error_msg.lower():
                state_obj.add_error(f"{self.name} error: {error_msg}")
                # Still return state but mark the error
                return state_obj.model_dump()
            elif '400' in error_msg or 'Developer instruction' in error_msg:
                state_obj.add_error(f"{self.name} error: {error_msg}")
                return state_obj.model_dump()
            else:
                # Other errors - add to state and continue
                state_obj.add_error(f"{self.name} error: {error_msg}")
                return state_obj.model_dump()

