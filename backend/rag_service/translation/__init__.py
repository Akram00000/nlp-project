"""Translation module for Arabic-English translation integration."""

from .translation_interface import (
    TranslationProvider,
    TranslationResult,
    TranslationConfig,
    Language,
    MockTranslationProvider,
)
from .integration import TranslationIntegration
from .providers import (
    LLMTranslationProvider,
    TranslationProviderFactory,
    TranslationProviderConfig,
    SmartResponseTranslator,
    SmartTranslationResult,
    get_translator,
    translate_to_english,
    translate_to_english_async,
)

__all__ = [
    # Interface
    "TranslationProvider",
    "TranslationResult",
    "TranslationConfig",
    "Language",
    "MockTranslationProvider",
    # Integration
    "TranslationIntegration",
    # Providers
    "LLMTranslationProvider",
    "TranslationProviderFactory",
    "TranslationProviderConfig",
    "SmartResponseTranslator",
    "SmartTranslationResult",
    # Convenience functions
    "get_translator",
    "translate_to_english",
    "translate_to_english_async",
]
