"""
OpenAI-Compatible Provider Implementation.

Works with OpenAI API and any OpenAI-compatible endpoint.
Best for: Flexibility, works with various backends that expose OpenAI-style API.
"""

from typing import AsyncIterator, Optional
from loguru import logger

from .base import BaseLLMProvider, ProviderConfig, LLMResponse, Message, ProviderType


class OpenAICompatibleProvider(BaseLLMProvider):
    """
    OpenAI-compatible API provider.
    
    Works with:
    - OpenAI API
    - Azure OpenAI
    - vLLM
    - LocalAI
    - Any OpenAI-compatible endpoint
    """
    
    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.OPENAI,
                base_url="https://api.openai.com/v1",
                model="gpt-4",
            )
        super().__init__(config)
        
        self._client = None
        self._async_client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI, AsyncOpenAI
            
            api_key = self.config.api_key
            if not api_key:
                import os
                api_key = os.getenv("OPENAI_API_KEY")
            
            base_url = self.config.base_url or "https://api.openai.com/v1"
            
            self._client = OpenAI(
                api_key=api_key or "not-needed",  # Some local endpoints don't need key
                base_url=base_url,
                timeout=self.config.timeout
            )
            
            self._async_client = AsyncOpenAI(
                api_key=api_key or "not-needed",
                base_url=base_url,
                timeout=self.config.timeout
            )
            
            self._initialized = True
            
        except ImportError:
            logger.error("openai package not installed")
            raise ImportError("Please install openai: pip install openai")
    
    def generate(
        self,
        messages: list[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI-compatible API."""
        
        if not self._initialized:
            raise RuntimeError("OpenAI provider not initialized.")
        
        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=self._format_messages(messages),
                max_tokens=self._get_param("max_tokens", max_tokens),
                temperature=self._get_param("temperature", temperature),
                **kwargs
            )
            
            return self._parse_response(response)
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise RuntimeError(f"OpenAI generation failed: {e}") from e
    
    async def generate_async(
        self,
        messages: list[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Asynchronously generate response using OpenAI-compatible API."""
        
        if not self._initialized:
            raise RuntimeError("OpenAI provider not initialized.")
        
        try:
            response = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=self._format_messages(messages),
                max_tokens=self._get_param("max_tokens", max_tokens),
                temperature=self._get_param("temperature", temperature),
                **kwargs
            )
            
            return self._parse_response(response)
            
        except Exception as e:
            logger.error(f"OpenAI async generation failed: {e}")
            raise RuntimeError(f"OpenAI generation failed: {e}") from e
    
    async def stream(
        self,
        messages: list[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream response tokens from OpenAI-compatible API."""
        
        if not self._initialized:
            raise RuntimeError("OpenAI provider not initialized.")
        
        try:
            stream = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=self._format_messages(messages),
                max_tokens=self._get_param("max_tokens", max_tokens),
                temperature=self._get_param("temperature", temperature),
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI stream failed: {e}")
            raise RuntimeError(f"OpenAI streaming failed: {e}") from e
    
    def health_check(self) -> bool:
        """Check if OpenAI API is accessible."""
        if not self._initialized:
            return False
        
        try:
            # Try listing models as a health check
            models = self._client.models.list()
            return True
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False
    
    def _parse_response(self, response) -> LLMResponse:
        """Parse OpenAI API response."""
        
        choice = response.choices[0] if response.choices else None
        
        return LLMResponse(
            content=choice.message.content if choice else "",
            model=response.model,
            provider=self.name,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            finish_reason=choice.finish_reason if choice else None,
            raw_response=response
        )
