"""
Translation Service Interface - Abstract interface for translation providers.

Defines the contract that Akram's NLLB-200 implementation will fulfill.
Includes a mock implementation for testing RAG pipeline without translation service.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Language(str, Enum):
    """Supported languages."""
    ARABIC = "ar"
    ENGLISH = "en"


@dataclass
class TranslationConfig:
    """Configuration for translation service."""
    
    # Model settings
    model_name: str = "facebook/nllb-200-distilled-600M"
    device: str = "cpu"  # cpu, cuda, mps
    
    # Translation settings
    max_length: int = 512
    batch_size: int = 8
    
    # Quality settings
    num_beams: int = 5
    temperature: float = 1.0
    
    # Timeout
    timeout: float = 30.0


@dataclass
class TranslationResult:
    """Result of a translation operation."""
    
    source_text: str
    translated_text: str
    source_language: Language
    target_language: Language
    confidence: Optional[float] = None
    metadata: dict = field(default_factory=dict)
    
    @property
    def is_arabic_to_english(self) -> bool:
        return self.source_language == Language.ARABIC and self.target_language == Language.ENGLISH
    
    @property
    def is_english_to_arabic(self) -> bool:
        return self.source_language == Language.ENGLISH and self.target_language == Language.ARABIC


class TranslationProvider(ABC):
    """
    Abstract base class for translation providers.
    
    Akram should implement this interface for the NLLB-200 translation service.
    
    Required methods:
    - translate(): Translate single text
    - translate_batch(): Translate multiple texts
    - detect_language(): Detect language of text
    - health_check(): Verify service availability
    """
    
    @abstractmethod
    def translate(
        self,
        text: str,
        source_language: Language,
        target_language: Language,
        **kwargs
    ) -> TranslationResult:
        """
        Translate text from source to target language.
        
        Args:
            text: Text to translate
            source_language: Source language (ar/en)
            target_language: Target language (ar/en)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            TranslationResult with translated text
        """
        pass
    
    @abstractmethod
    async def translate_async(
        self,
        text: str,
        source_language: Language,
        target_language: Language,
        **kwargs
    ) -> TranslationResult:
        """Async version of translate."""
        pass
    
    @abstractmethod
    def translate_batch(
        self,
        texts: list[str],
        source_language: Language,
        target_language: Language,
        **kwargs
    ) -> list[TranslationResult]:
        """
        Translate multiple texts efficiently.
        
        Args:
            texts: List of texts to translate
            source_language: Source language
            target_language: Target language
            **kwargs: Additional parameters
            
        Returns:
            List of TranslationResult objects
        """
        pass
    
    @abstractmethod
    def detect_language(self, text: str) -> Language:
        """
        Detect the language of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected language
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if translation service is available."""
        pass
    
    # Convenience methods
    def translate_ar_to_en(self, text: str, **kwargs) -> TranslationResult:
        """Translate Arabic to English."""
        return self.translate(text, Language.ARABIC, Language.ENGLISH, **kwargs)
    
    def translate_en_to_ar(self, text: str, **kwargs) -> TranslationResult:
        """Translate English to Arabic."""
        return self.translate(text, Language.ENGLISH, Language.ARABIC, **kwargs)
    
    async def translate_ar_to_en_async(self, text: str, **kwargs) -> TranslationResult:
        """Async Arabic to English translation."""
        return await self.translate_async(text, Language.ARABIC, Language.ENGLISH, **kwargs)
    
    async def translate_en_to_ar_async(self, text: str, **kwargs) -> TranslationResult:
        """Async English to Arabic translation."""
        return await self.translate_async(text, Language.ENGLISH, Language.ARABIC, **kwargs)


class MockTranslationProvider(TranslationProvider):
    """
    Mock translation provider for testing.
    
    Returns placeholder translations to allow RAG pipeline testing
    without the actual translation service.
    
    Replace with Akram's NLLB-200 implementation in production.
    """
    
    def __init__(self, config: Optional[TranslationConfig] = None):
        self.config = config or TranslationConfig()
    
    def translate(
        self,
        text: str,
        source_language: Language,
        target_language: Language,
        **kwargs
    ) -> TranslationResult:
        """Return mock translation."""
        
        # Simple mock: just indicate it's a translation
        if source_language == Language.ARABIC and target_language == Language.ENGLISH:
            translated = f"[EN Translation of: {text[:50]}...]" if len(text) > 50 else f"[EN Translation of: {text}]"
        elif source_language == Language.ENGLISH and target_language == Language.ARABIC:
            translated = f"[ترجمة عربية: {text[:50]}...]" if len(text) > 50 else f"[ترجمة عربية: {text}]"
        else:
            translated = text
        
        return TranslationResult(
            source_text=text,
            translated_text=translated,
            source_language=source_language,
            target_language=target_language,
            confidence=0.5,  # Mock confidence
            metadata={"mock": True}
        )
    
    async def translate_async(
        self,
        text: str,
        source_language: Language,
        target_language: Language,
        **kwargs
    ) -> TranslationResult:
        """Async mock translation."""
        return self.translate(text, source_language, target_language, **kwargs)
    
    def translate_batch(
        self,
        texts: list[str],
        source_language: Language,
        target_language: Language,
        **kwargs
    ) -> list[TranslationResult]:
        """Batch mock translation."""
        return [
            self.translate(text, source_language, target_language, **kwargs)
            for text in texts
        ]
    
    def detect_language(self, text: str) -> Language:
        """Simple language detection based on character analysis."""
        if not text:
            return Language.ENGLISH
        
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        arabic_ratio = arabic_chars / len(text)
        
        return Language.ARABIC if arabic_ratio > 0.3 else Language.ENGLISH
    
    def health_check(self) -> bool:
        """Mock always returns True."""
        return True


# ============================================================================
# IMPLEMENTATION GUIDE FOR AKRAM
# ============================================================================
"""
IMPLEMENTATION GUIDE FOR AKRAM - NLLB-200 Translation Service
==============================================================

To implement the actual translation service:

1. Create a new file: `backend/rag_service/translation/nllb_provider.py`

2. Implement the TranslationProvider interface:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from .translation_interface import TranslationProvider, TranslationResult, TranslationConfig, Language

class NLLBTranslationProvider(TranslationProvider):
    
    # NLLB language codes
    LANG_CODES = {
        Language.ARABIC: "ara_Arab",
        Language.ENGLISH: "eng_Latn",
    }
    
    def __init__(self, config: TranslationConfig = None):
        self.config = config or TranslationConfig()
        
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)
        self.model.to(self.config.device)
    
    def translate(
        self,
        text: str,
        source_language: Language,
        target_language: Language,
        **kwargs
    ) -> TranslationResult:
        
        # Set source language
        self.tokenizer.src_lang = self.LANG_CODES[source_language]
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.config.max_length, truncation=True)
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        # Generate translation
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.LANG_CODES[target_language])
        outputs = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=self.config.max_length,
            num_beams=self.config.num_beams,
        )
        
        # Decode
        translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return TranslationResult(
            source_text=text,
            translated_text=translated,
            source_language=source_language,
            target_language=target_language,
        )
    
    # Implement other methods...
```

3. Register in __init__.py when ready:
```python
from .nllb_provider import NLLBTranslationProvider
```

4. Update integration.py to use NLLBTranslationProvider as default

5. Consider:
   - GPU acceleration (device="cuda")
   - Batch processing for efficiency
   - Caching frequent translations (Redis)
   - Error handling and retries
   - Quality metrics (BLEU scores)
"""
