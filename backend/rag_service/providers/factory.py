"""
Provider Factory for creating LLM provider instances.

Handles provider creation based on configuration.
"""

from typing import Optional
from loguru import logger

from .base import BaseLLMProvider, ProviderConfig, ProviderType
from .lmstudio import LMStudioProvider
from .gemini import GeminiProvider
from .openai_compat import OpenAICompatibleProvider
from .huggingface import HuggingFaceProvider


class ProviderFactory:
    """Factory for creating LLM provider instances."""
    
    _providers: dict[str, type[BaseLLMProvider]] = {
        "lmstudio": LMStudioProvider,
        "gemini": GeminiProvider,
        "openai": OpenAICompatibleProvider,
        "huggingface": HuggingFaceProvider,
    }
    
    @classmethod
    def register(cls, name: str, provider_class: type[BaseLLMProvider]):
        """Register a custom provider."""
        cls._providers[name.lower()] = provider_class
        logger.info(f"Registered provider: {name}")
    
    @classmethod
    def create(
        cls,
        provider_name: str,
        config: Optional[ProviderConfig] = None,
        **kwargs
    ) -> BaseLLMProvider:
        """
        Create a provider instance.
        
        Args:
            provider_name: Name of the provider (lmstudio, gemini, openai)
            config: Optional provider configuration
            **kwargs: Additional config parameters
            
        Returns:
            Configured provider instance
        """
        provider_name = provider_name.lower()
        
        if provider_name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unknown provider: {provider_name}. Available: {available}"
            )
        
        provider_class = cls._providers[provider_name]
        
        # Build config if not provided
        if config is None:
            provider_type = ProviderType(provider_name)
            config = ProviderConfig(provider_type=provider_type, **kwargs)
        
        return provider_class(config)
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List available provider names."""
        return list(cls._providers.keys())


def get_provider(
    provider_name: Optional[str] = None,
    config: Optional[ProviderConfig] = None,
    **kwargs
) -> BaseLLMProvider:
    """
    Convenience function to get a provider instance.
    
    Args:
        provider_name: Provider name. If None, uses default from settings.
        config: Optional provider configuration
        **kwargs: Additional config parameters
        
    Returns:
        Configured provider instance
    """
    from ..config.settings import settings
    
    if provider_name is None:
        provider_name = settings.default_provider
    
    # If config not provided, build from settings
    if config is None:
        provider_name_lower = provider_name.lower()
        
        if provider_name_lower == "lmstudio":
            s = settings.lmstudio
            config = ProviderConfig(
                provider_type=ProviderType.LMSTUDIO,
                base_url=s.base_url,
                model=s.model,
                max_tokens=s.max_tokens,
                temperature=s.temperature,
                timeout=s.timeout,
            )
        elif provider_name_lower == "gemini":
            s = settings.gemini
            config = ProviderConfig(
                provider_type=ProviderType.GEMINI,
                api_key=s.api_key,
                model=s.model,
                max_tokens=s.max_tokens,
                temperature=s.temperature,
            )
        elif provider_name_lower == "openai":
            s = settings.openai
            config = ProviderConfig(
                provider_type=ProviderType.OPENAI,
                api_key=s.api_key,
                base_url=s.base_url,
                model=s.model,
                max_tokens=s.max_tokens,
                temperature=s.temperature,
                timeout=s.timeout,
            )
        elif provider_name_lower == "huggingface":
            s = settings.huggingface
            config = ProviderConfig(
                provider_type=ProviderType.HUGGINGFACE,
                api_key=s.api_key,
                base_url=s.base_url,
                model=s.model,
                max_tokens=s.max_tokens,
                temperature=s.temperature,
                timeout=s.timeout,
            )
    
    return ProviderFactory.create(provider_name, config, **kwargs)
