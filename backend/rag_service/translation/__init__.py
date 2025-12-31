"""Translation module for Arabic-English translation integration."""

from .translation_interface import (
    TranslationProvider,
    TranslationResult,
    TranslationConfig,
    MockTranslationProvider,
)
from .integration import TranslationIntegration

__all__ = [
    "TranslationProvider",
    "TranslationResult",
    "TranslationConfig",
    "MockTranslationProvider",
    "TranslationIntegration",
]
