"""
Hugging Face Inference API Provider.

Uses the OpenAI-compatible endpoint provided by Hugging Face.
"""

from typing import Optional
from .openai_compat import OpenAICompatibleProvider
from .base import ProviderConfig, ProviderType


class HuggingFaceProvider(OpenAICompatibleProvider):
    """
    Provider for Hugging Face Inference API.
    
    See: https://huggingface.co/docs/api-inference/detailed_parameters#openai-compatibility
    """
    
    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.HUGGINGFACE,
                base_url="https://api-inference.huggingface.co/v1/",
                model="meta-llama/Llama-3.2-3B-Instruct",
            )
        super().__init__(config)
        self.name = "huggingface"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.HUGGINGFACE
