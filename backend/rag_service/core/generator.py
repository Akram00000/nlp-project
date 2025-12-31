"""
Generator Module - LLM response generation.

Handles the generation of responses using the configured LLM provider,
with support for streaming, safety checks, and parameter management.
"""

from dataclasses import dataclass, field
from typing import Optional, AsyncIterator
from loguru import logger

from ..providers.base import BaseLLMProvider, LLMResponse, Message
from ..providers.factory import get_provider


@dataclass
class GenerationConfig:
    """Configuration for generation."""
    
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop_sequences: list[str] = field(default_factory=list)


class Generator:
    """
    Response generator using LLM providers.
    
    Features:
    - Provider-agnostic generation
    - Streaming support
    - Safety checks
    - Automatic provider switching
    """
    
    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        provider_name: Optional[str] = None,
    ):
        """
        Initialize generator.
        
        Args:
            provider: Pre-configured provider instance
            provider_name: Name of provider to create (if provider not given)
        """
        self._provider = provider
        self._provider_name = provider_name
        self._fallback_providers: list[str] = []
    
    @property
    def provider(self) -> BaseLLMProvider:
        """Get or create the LLM provider."""
        if self._provider is None:
            self._provider = get_provider(self._provider_name)
        return self._provider
    
    def set_provider(self, provider: BaseLLMProvider | str):
        """
        Switch to a different provider.
        
        Args:
            provider: Provider instance or provider name
        """
        if isinstance(provider, str):
            self._provider = get_provider(provider)
            self._provider_name = provider
        else:
            self._provider = provider
            self._provider_name = provider.name
        
        logger.info(f"Switched to provider: {self._provider_name}")
    
    def set_fallback_providers(self, providers: list[str]):
        """Set fallback providers for when primary fails."""
        self._fallback_providers = providers
    
    def generate(
        self,
        messages: list[Message] | list[dict],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from messages.
        
        Args:
            messages: List of Message objects or dicts with role/content
            config: Generation configuration
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse with generated content
        """
        # Convert dicts to Message objects if needed
        messages = self._normalize_messages(messages)
        
        # Extract config parameters
        gen_params = self._extract_params(config, kwargs)
        
        # Try primary provider, then fallbacks
        providers_to_try = [self.provider.name] + self._fallback_providers
        last_error = None
        
        for i, provider_name in enumerate(providers_to_try):
            try:
                if i > 0:
                    logger.info(f"Trying fallback provider: {provider_name}")
                    self.set_provider(provider_name)
                
                response = self.provider.generate(messages, **gen_params)
                return response
                
            except Exception as e:
                logger.warning(f"Generation failed with {provider_name}: {e}")
                last_error = e
        
        raise RuntimeError(f"All providers failed. Last error: {last_error}")
    
    async def generate_async(
        self,
        messages: list[Message] | list[dict],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Asynchronously generate a response.
        
        Args:
            messages: List of messages
            config: Generation configuration
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with generated content
        """
        messages = self._normalize_messages(messages)
        gen_params = self._extract_params(config, kwargs)
        
        providers_to_try = [self.provider.name] + self._fallback_providers
        last_error = None
        
        for i, provider_name in enumerate(providers_to_try):
            try:
                if i > 0:
                    logger.info(f"Trying fallback provider: {provider_name}")
                    self.set_provider(provider_name)
                
                response = await self.provider.generate_async(messages, **gen_params)
                return response
                
            except Exception as e:
                logger.warning(f"Async generation failed with {provider_name}: {e}")
                last_error = e
        
        raise RuntimeError(f"All providers failed. Last error: {last_error}")
    
    async def stream(
        self,
        messages: list[Message] | list[dict],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream response tokens.
        
        Args:
            messages: List of messages
            config: Generation configuration
            **kwargs: Additional parameters
            
        Yields:
            Generated text tokens
        """
        messages = self._normalize_messages(messages)
        gen_params = self._extract_params(config, kwargs)
        
        try:
            async for token in self.provider.stream(messages, **gen_params):
                yield token
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise
    
    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response with provided context (RAG pattern).
        
        Args:
            query: User query
            context: Retrieved context to augment the query
            system_prompt: Optional system prompt
            config: Generation configuration
            
        Returns:
            LLMResponse with generated content
        """
        from ..utils.prompts import build_rag_prompt
        
        # Build messages
        messages = []
        
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        
        # Combine context and query into user message
        augmented_query = build_rag_prompt(query, context)
        messages.append(Message(role="user", content=augmented_query))
        
        return self.generate(messages, config, **kwargs)
    
    def health_check(self) -> bool:
        """Check if the provider is healthy."""
        try:
            return self.provider.health_check()
        except Exception as e:
            logger.warning(f"Generator health check failed: {e}")
            return False
    
    def _normalize_messages(self, messages: list) -> list[Message]:
        """Convert dict messages to Message objects."""
        normalized = []
        for msg in messages:
            if isinstance(msg, Message):
                normalized.append(msg)
            elif isinstance(msg, dict):
                normalized.append(Message(
                    role=msg.get("role", "user"),
                    content=msg.get("content", "")
                ))
            else:
                raise ValueError(f"Invalid message format: {type(msg)}")
        return normalized
    
    def _extract_params(
        self,
        config: Optional[GenerationConfig],
        kwargs: dict
    ) -> dict:
        """Extract generation parameters from config and kwargs."""
        params = {}
        
        if config:
            if config.max_tokens:
                params["max_tokens"] = config.max_tokens
            if config.temperature is not None:
                params["temperature"] = config.temperature
            if config.top_p is not None:
                params["top_p"] = config.top_p
            if config.stop_sequences:
                params["stop"] = config.stop_sequences
        
        # kwargs override config
        params.update(kwargs)
        
        return params
