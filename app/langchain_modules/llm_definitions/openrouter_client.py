"""OpenRouter LLM client with LangChain integration."""
from typing import Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from app.config.settings import settings


def create_openrouter_client(
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_retries: int = 3
) -> BaseChatModel:
    """
    Create an OpenRouter LLM client compatible with LangChain.
    
    Args:
        model: Model name (defaults to settings.openrouter_model)
        temperature: Temperature for generation
        max_retries: Maximum number of retries
        
    Returns:
        LangChain-compatible LLM client
    """
    model_name = model or settings.openrouter_model
    api_key = settings.openrouter_api_key
    
    if not api_key:
        raise ValueError("OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable.")
    
    # Use ChatOpenAI with OpenRouter endpoint
    # OpenRouter is compatible with OpenAI API format
    client = ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=temperature,
        max_retries=max_retries,
        timeout=60.0,
    )
    
    # Store model name for later reference
    client.model_name = model_name
    
    return client


def create_dual_clients(
    temperature: float = 0.0,
    max_retries: int = 3
) -> Tuple[BaseChatModel, BaseChatModel]:
    """
    Create a pair of OpenRouter clients: a strong vision model and a fast model.
    
    Returns:
        Tuple of (vision_llm, fast_llm):
        - vision_llm: gpt-4o for description + primary color (vision-critical)
        - fast_llm: gpt-4o-mini for multi-gate, secondary colors, neutral verify
    """
    vision_llm = create_openrouter_client(
        model=settings.openrouter_vision_model,
        temperature=temperature,
        max_retries=max_retries,
    )
    fast_llm = create_openrouter_client(
        model=settings.openrouter_model,
        temperature=temperature,
        max_retries=max_retries,
    )
    return vision_llm, fast_llm

