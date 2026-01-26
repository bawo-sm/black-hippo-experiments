"""Configuration settings for the classification system."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""
    
    # OpenRouter API Configuration
    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-4o-mini"  # Vision-capable model
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    
    # Retry Configuration
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    retry_initial_delay: float = 1.0
    
    # Rate Limiting
    requests_per_minute: int = 60
    requests_per_second: float = 2.0
    
    # Resources Path
    resources_dir: str = "resources"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

