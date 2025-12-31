"""
Abstract base class for LLM providers.

All provider implementations must inherit from BaseLLMProvider and implement
the required methods for generation and embedding.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, Any
from enum import Enum


class ProviderType(str, Enum):
    """Supported LLM provider types."""
    LMSTUDIO = "lmstudio"
    GEMINI = "gemini"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    
    provider_type: ProviderType
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: str = "default"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: float = 120.0
    extra_params: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Standardized response from LLM providers."""
    
    content: str
    model: str
    provider: str
    usage: dict = field(default_factory=dict)  # tokens used
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None  # Original provider response
    
    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)
    
    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)
    
    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", self.prompt_tokens + self.completion_tokens)


@dataclass
class Message:
    """Chat message structure."""
    
    role: str  # "system", "user", "assistant"
    content: str
    
    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All provider implementations must implement:
    - generate(): Synchronous text generation
    - generate_async(): Asynchronous text generation  
    - stream(): Streaming text generation
    - health_check(): Verify provider availability
    
    Optionally implement:
    - embed(): Text embedding (if provider supports it)
    """
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self._initialized = False
    
    @property
    def name(self) -> str:
        """Provider name for identification."""
        return self.config.provider_type.value
    
    @property
    def model(self) -> str:
        """Current model being used."""
        return self.config.model
    
    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of chat messages
            max_tokens: Override default max tokens
            temperature: Override default temperature
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse with generated content
        """
        pass
    
    @abstractmethod
    async def generate_async(
        self,
        messages: list[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Asynchronously generate a response from the LLM.
        
        Args:
            messages: List of chat messages
            max_tokens: Override default max tokens
            temperature: Override default temperature
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse with generated content
        """
        pass
    
    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream response tokens from the LLM.
        
        Args:
            messages: List of chat messages
            max_tokens: Override default max tokens
            temperature: Override default temperature
            **kwargs: Additional provider-specific parameters
            
        Yields:
            Generated text tokens as they become available
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the provider is available and responding.
        
        Returns:
            True if provider is healthy, False otherwise
        """
        pass
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for texts (optional).
        
        Default implementation raises NotImplementedError.
        Override in providers that support embedding.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        raise NotImplementedError(
            f"Provider {self.name} does not support embedding. "
            "Use a dedicated embedding model instead."
        )
    
    def _get_param(self, param_name: str, override: Optional[Any]) -> Any:
        """Get parameter value with override support."""
        if override is not None:
            return override
        return getattr(self.config, param_name, None)
    
    def _format_messages(self, messages: list[Message]) -> list[dict]:
        """Convert Message objects to provider-expected format."""
        return [msg.to_dict() for msg in messages]
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
