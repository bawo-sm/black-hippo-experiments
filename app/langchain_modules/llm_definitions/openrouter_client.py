"""OpenRouter LLM client with LangChain integration."""
from typing import Optional
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

