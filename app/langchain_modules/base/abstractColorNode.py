"""Abstract base classes for color detection graph nodes."""
from abc import ABC, abstractmethod
from typing import Any, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

from app.data_models.color_detection_state import ColorDetectionState


class AbstractColorNode(ABC):
    """Abstract base class for color detection graph nodes."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def __call__(self, state) -> dict:
        """Execute the node logic."""
        pass


class AbstractColorAgent(AbstractColorNode):
    """Abstract base class for LLM-based color detection nodes."""
    
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
        """Create messages for LLM with optional image."""
        messages = []
        
        # Check if model supports system messages
        supports_system = True
        if model_name:
            no_system_models = ['gemma', 'gemini']
            if any(no_sys in model_name.lower() for no_sys in no_system_models):
                supports_system = False
        
        if system_prompt and supports_system:
            from langchain_core.messages import SystemMessage
            messages.append(SystemMessage(content=system_prompt))
            final_user_prompt = user_prompt
        elif system_prompt:
            final_user_prompt = f"{system_prompt}\n\n{user_prompt}"
        else:
            final_user_prompt = user_prompt
        
        # Create user message with optional image
        if image_data:
            if image_data.startswith("http://") or image_data.startswith("https://"):
                image_url = image_data
            elif image_data.startswith("data:image"):
                image_url = image_data
            else:
                image_url = f"data:image/jpeg;base64,{image_data}"
            
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
    
    @staticmethod
    def _extract_json(content: str) -> dict:
        """
        Robustly extract a JSON object from LLM response text.
        
        Tries in order:
        1. Direct json.loads on full content
        2. Strip markdown code fences and parse
        3. Find outermost { ... } with brace balancing
        """
        import json
        import re
        
        text = content.strip()
        
        # 1. Direct parse
        try:
            return json.loads(text)
        except Exception:
            pass
        
        # 2. Strip markdown fences (```json ... ``` or ``` ... ```)
        fenced = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
        if fenced:
            try:
                return json.loads(fenced.group(1).strip())
            except Exception:
                pass
        
        # 3. Brace-balanced extraction
        start = text.find('{')
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i+1])
                        except Exception:
                            break
        
        raise ValueError(f"Could not extract JSON from: {text[:200]}")
    
    def _invoke_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        image_data: Optional[str] = None
    ) -> Any:
        """Invoke LLM with prompts and optional image."""
        model_name = None
        if hasattr(self.llm, 'model_name'):
            model_name = self.llm.model_name
        elif hasattr(self.llm, 'model'):
            model_name = self.llm.model
        
        messages = self._create_messages(system_prompt, user_prompt, image_data, model_name)
        response = self.llm.invoke(messages)
        
        if self.output_parser:
            # Try structured parse first
            try:
                return self.output_parser.parse(response.content)
            except Exception:
                pass
            
            # Robust JSON extraction + manual construction
            try:
                parsed = self._extract_json(response.content)
                return self.output_parser.pydantic_object(**parsed)
            except Exception:
                pass
            
            # Last resort: return raw content for manual parsing in _process_output
            return response.content
        
        return response.content
    
    @abstractmethod
    def _build_system_prompt(self, state: ColorDetectionState) -> str:
        """Build the system prompt for this node."""
        pass
    
    @abstractmethod
    def _build_user_prompt(self, state: ColorDetectionState) -> str:
        """Build the user prompt for this node."""
        pass
    
    @abstractmethod
    def _process_output(self, output: Any, state: ColorDetectionState) -> ColorDetectionState:
        """Process LLM output and update state."""
        pass
    
    def __call__(self, state) -> dict:
        """Execute the agent node."""
        if isinstance(state, dict):
            state_obj = ColorDetectionState(**state)
        elif isinstance(state, ColorDetectionState):
            state_obj = state
        else:
            state_obj = ColorDetectionState(**dict(state))
        
        try:
            system_prompt = self._build_system_prompt(state_obj)
            user_prompt = self._build_user_prompt(state_obj)
            
            output = self._invoke_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_data=state_obj.image_data
            )
            
            updated_state = self._process_output(output, state_obj)
            return updated_state.model_dump()
        
        except Exception as e:
            error_msg = str(e)
            if '429' in error_msg or 'rate limit' in error_msg.lower() or 'rate-limited' in error_msg.lower():
                state_obj.add_error(f"{self.name} error: {error_msg}")
                return state_obj.model_dump()
            elif '400' in error_msg or 'Developer instruction' in error_msg:
                state_obj.add_error(f"{self.name} error: {error_msg}")
                return state_obj.model_dump()
            else:
                state_obj.add_error(f"{self.name} error: {error_msg}")
                return state_obj.model_dump()
