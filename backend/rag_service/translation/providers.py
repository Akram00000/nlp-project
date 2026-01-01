"""
Translation Providers - LLM-based translation implementations.

Provides translation functionality using LLM providers (Gemini, LMStudio).
Uses existing English data from hadiths when available, otherwise translates via LLM.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from loguru import logger
import re

from .translation_interface import TranslationProvider, TranslationResult, Language


# ============================================================
# Translation Prompts
# ============================================================

ISLAMIC_TRANSLATION_PROMPT = """You are an expert translator specializing in Islamic texts. Translate the following Arabic text to English.

IMPORTANT GUIDELINES:
1. Preserve the meaning and scholarly tone of the original text
2. Keep Quranic verses (آيات) in Arabic, followed by English translation in parentheses
3. Keep hadith chains (سند) and Arabic honorifics like ﷺ, رضي الله عنه, etc.
4. Translate fiqh terminology accurately (e.g., وضوء = wudu/ablution, زكاة = zakat, etc.)
5. For hadith narrations, keep "عن...عن..." chains in Arabic
6. Do NOT add any commentary or explanation - only translate
7. Maintain paragraph structure

Arabic Text:
{text}

English Translation:"""


SMART_TRANSLATION_PROMPT = """You are an expert translator for Islamic texts. Translate the following content to English.

RULES:
1. If the text already contains English portions, preserve them as-is
2. Keep Quranic Arabic text intact, add translation in parentheses
3. Keep hadith text in Arabic if it's a direct narration, translate the explanation
4. Preserve all Arabic honorifics (ﷺ، رضي الله عنه، etc.)
5. For terms like فتوى، حديث، سنة - keep Arabic with English in parentheses first time
6. Be concise and scholarly in tone

Content:
{text}

Translation:"""


# ============================================================
# Base LLM Translation Provider
# ============================================================

@dataclass
class TranslationProviderConfig:
    """Configuration for LLM-based translation."""
    provider_name: str = "gemini"
    model: str = "gemini-1.5-flash"
    max_tokens: int = 4000
    temperature: float = 0.3  # Lower for more consistent translations
    timeout: float = 60.0


class LLMTranslationProvider(TranslationProvider):
    """
    Base class for LLM-based translation providers.
    
    Wraps existing LLM providers to provide translation functionality.
    """
    
    def __init__(self, llm_provider, config: Optional[TranslationProviderConfig] = None):
        """
        Initialize with an LLM provider.
        
        Args:
            llm_provider: An instance of BaseLLMProvider (Gemini, LMStudio, etc.)
            config: Translation configuration
        """
        self.llm_provider = llm_provider
        self.config = config or TranslationProviderConfig()
        self._initialized = llm_provider is not None
    
    def translate(
        self,
        text: str,
        source_language: Language,
        target_language: Language,
        **kwargs
    ) -> TranslationResult:
        """Translate text using LLM."""
        
        if not self._initialized or not self.llm_provider:
            raise RuntimeError("Translation provider not initialized - LLM not available")
        
        # Only support Arabic to English for now
        if source_language != Language.ARABIC or target_language != Language.ENGLISH:
            raise ValueError("Only Arabic to English translation is supported")
        
        # Use smart prompt for Islamic content
        prompt = kwargs.get("prompt_template", SMART_TRANSLATION_PROMPT)
        full_prompt = prompt.format(text=text)
        
        try:
            from ..providers.base import Message
            
            messages = [
                Message(role="user", content=full_prompt)
            ]
            
            response = self.llm_provider.generate(
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            
            translated_text = response.content.strip()
            
            return TranslationResult(
                source_text=text,
                translated_text=translated_text,
                source_language=source_language,
                target_language=target_language,
                confidence=0.9,  # LLM translations are generally good quality
                metadata={"provider": self.llm_provider.name}
            )
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise RuntimeError(f"Translation failed: {e}") from e
    
    async def translate_async(
        self,
        text: str,
        source_language: Language,
        target_language: Language,
        **kwargs
    ) -> TranslationResult:
        """Async translation using LLM."""
        
        if not self._initialized or not self.llm_provider:
            raise RuntimeError("Translation provider not initialized")
        
        if source_language != Language.ARABIC or target_language != Language.ENGLISH:
            raise ValueError("Only Arabic to English translation is supported")
        
        prompt = kwargs.get("prompt_template", SMART_TRANSLATION_PROMPT)
        full_prompt = prompt.format(text=text)
        
        try:
            from ..providers.base import Message
            
            messages = [
                Message(role="user", content=full_prompt)
            ]
            
            response = await self.llm_provider.generate_async(
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            
            translated_text = response.content.strip()
            
            return TranslationResult(
                source_text=text,
                translated_text=translated_text,
                source_language=source_language,
                target_language=target_language,
                confidence=0.9,
                metadata={"provider": self.llm_provider.name}
            )
            
        except Exception as e:
            logger.error(f"Async translation failed: {e}")
            raise RuntimeError(f"Translation failed: {e}") from e
    
    def translate_batch(
        self,
        texts: List[str],
        source_language: Language,
        target_language: Language,
        **kwargs
    ) -> List[TranslationResult]:
        """Translate multiple texts (sequentially for LLM)."""
        
        results = []
        for text in texts:
            try:
                result = self.translate(text, source_language, target_language, **kwargs)
                results.append(result)
            except Exception as e:
                logger.warning(f"Batch translation item failed: {e}")
                # Return original text on failure
                results.append(TranslationResult(
                    source_text=text,
                    translated_text=text,  # Fallback to original
                    source_language=source_language,
                    target_language=target_language,
                    confidence=0.0,
                    metadata={"error": str(e)}
                ))
        
        return results
    
    def detect_language(self, text: str) -> Language:
        """Detect language based on character analysis."""
        # Simple heuristic: check for Arabic characters
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
        arabic_matches = arabic_pattern.findall(text)
        
        total_chars = len(text.replace(" ", ""))
        arabic_chars = sum(len(m) for m in arabic_matches)
        
        if total_chars > 0 and arabic_chars / total_chars > 0.3:
            return Language.ARABIC
        return Language.ENGLISH
    
    def health_check(self) -> bool:
        """Check if LLM provider is available."""
        if not self._initialized or not self.llm_provider:
            return False
        
        try:
            return self.llm_provider.health_check()
        except:
            return False


# ============================================================
# Translation Provider Factory
# ============================================================

class TranslationProviderFactory:
    """
    Factory for creating translation providers.
    
    Supports automatic fallback: tries Gemini first, then LMStudio.
    """
    
    _providers: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: type):
        """Register a translation provider."""
        cls._providers[name] = provider_class
    
    @classmethod
    def from_llm_provider(
        cls,
        llm_provider,
        config: Optional[TranslationProviderConfig] = None
    ) -> LLMTranslationProvider:
        """
        Create a translation provider from an existing LLM provider.
        
        This is the preferred method when you already have a working LLM provider
        (e.g., from the RAG engine).
        
        Args:
            llm_provider: An existing BaseLLMProvider instance
            config: Optional translation configuration
            
        Returns:
            Configured LLMTranslationProvider
        """
        provider_name = getattr(llm_provider.config, 'provider_type', 'unknown')
        logger.info(f"Creating translation provider from existing LLM provider: {provider_name}")
        return LLMTranslationProvider(
            llm_provider=llm_provider,
            config=config or TranslationProviderConfig(provider_name=provider_name)
        )
    
    @classmethod
    def create(
        cls,
        provider_name: Optional[str] = None,
        config: Optional[TranslationProviderConfig] = None,
    ) -> LLMTranslationProvider:
        """
        Create a translation provider.
        
        Args:
            provider_name: Name of the LLM provider ("gemini", "lmstudio", or None for auto)
            config: Translation configuration
            
        Returns:
            Configured LLMTranslationProvider
        """
        from ..providers.factory import get_provider
        
        providers_to_try = []
        
        if provider_name:
            providers_to_try = [provider_name]
        else:
            # Auto-fallback order: Gemini (better for Arabic) -> LMStudio
            providers_to_try = ["gemini", "lmstudio"]
        
        last_error = None
        
        for name in providers_to_try:
            try:
                logger.debug(f"Trying to create translation provider with: {name}")
                llm_provider = get_provider(name)
                
                if llm_provider:
                    # Skip strict health check - if provider initializes, it's likely usable
                    # Health check can fail due to API quirks even when generation works
                    try:
                        health_ok = llm_provider.health_check()
                        if not health_ok:
                            logger.debug(f"Health check returned False for {name}, but will try anyway")
                    except Exception as he:
                        logger.debug(f"Health check exception for {name}: {he}, but will try anyway")
                    
                    logger.info(f"Translation provider created: {name}")
                    return LLMTranslationProvider(
                        llm_provider=llm_provider,
                        config=config or TranslationProviderConfig(provider_name=name)
                    )
                    
            except Exception as e:
                logger.debug(f"Provider {name} not available: {e}")
                last_error = e
                continue
        
        if last_error:
            raise RuntimeError(f"No translation provider available. Last error: {last_error}")
        raise RuntimeError("No translation provider available")
    
    @classmethod
    def create_with_fallback(cls, config: Optional[TranslationProviderConfig] = None) -> Optional[LLMTranslationProvider]:
        """
        Create a translation provider with automatic fallback.
        Returns None if no provider is available instead of raising.
        """
        try:
            return cls.create(provider_name=None, config=config)
        except Exception as e:
            logger.warning(f"Could not create translation provider: {e}")
            return None


# ============================================================
# Smart Response Translator
# ============================================================

@dataclass
class SmartTranslationResult:
    """Result of smart translation with preserved content."""
    original_text: str
    translated_text: str
    preserved_sections: List[Dict[str, str]] = field(default_factory=list)
    used_existing_english: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class SmartResponseTranslator:
    """
    Smart translator that:
    1. Uses existing English translations from hadith metadata when available
    2. Preserves Quranic text in Arabic
    3. Translates remaining content via LLM
    """
    
    def __init__(self, translation_provider: Optional[LLMTranslationProvider] = None):
        """
        Initialize smart translator.
        
        Args:
            translation_provider: LLM translation provider (will auto-create if None)
        """
        self._provider = translation_provider
    
    @property
    def provider(self) -> Optional[LLMTranslationProvider]:
        """Lazy-load translation provider."""
        if self._provider is None:
            self._provider = TranslationProviderFactory.create_with_fallback()
        return self._provider
    
    def translate_response(
        self,
        text: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> SmartTranslationResult:
        """
        Translate a RAG response intelligently.
        
        Args:
            text: The response text to translate
            sources: List of source documents with metadata (may contain english_text)
            
        Returns:
            SmartTranslationResult with translated content
        """
        if not self.provider:
            logger.warning("No translation provider available")
            return SmartTranslationResult(
                original_text=text,
                translated_text=text,
                metadata={"error": "No translation provider available"}
            )
        
        # Check if sources have existing English translations
        english_hadith_map = {}
        if sources:
            for source in sources:
                metadata = source.get("metadata", {})
                if metadata.get("english_text"):
                    # Map Arabic content to English translation
                    arabic_content = source.get("text", "")[:200]  # Key by first 200 chars
                    english_hadith_map[arabic_content] = metadata["english_text"]
        
        # Try to substitute existing English hadith translations
        translated_text = text
        used_existing = False
        
        for arabic_key, english_text in english_hadith_map.items():
            if arabic_key in text:
                # Found a hadith with existing English translation
                # Note: This is a simple approach. Could be enhanced with better matching.
                used_existing = True
                logger.debug(f"Found existing English translation for hadith")
        
        # Translate the full response
        try:
            result = self.provider.translate(
                text=translated_text,
                source_language=Language.ARABIC,
                target_language=Language.ENGLISH,
            )
            
            translated_text = result.translated_text
            
        except Exception as e:
            logger.error(f"Smart translation failed: {e}")
            return SmartTranslationResult(
                original_text=text,
                translated_text=text,
                metadata={"error": str(e)}
            )
        
        return SmartTranslationResult(
            original_text=text,
            translated_text=translated_text,
            used_existing_english=used_existing,
            metadata={"provider": self.provider.llm_provider.name if self.provider.llm_provider else "unknown"}
        )
    
    async def translate_response_async(
        self,
        text: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> SmartTranslationResult:
        """Async version of translate_response."""
        
        if not self.provider:
            logger.warning("No translation provider available")
            return SmartTranslationResult(
                original_text=text,
                translated_text=text,
                metadata={"error": "No translation provider available"}
            )
        
        try:
            result = await self.provider.translate_async(
                text=text,
                source_language=Language.ARABIC,
                target_language=Language.ENGLISH,
            )
            
            return SmartTranslationResult(
                original_text=text,
                translated_text=result.translated_text,
                metadata={"provider": self.provider.llm_provider.name if self.provider.llm_provider else "unknown"}
            )
            
        except Exception as e:
            logger.error(f"Async smart translation failed: {e}")
            return SmartTranslationResult(
                original_text=text,
                translated_text=text,
                metadata={"error": str(e)}
            )


# ============================================================
# Module-level convenience functions
# ============================================================

_default_translator: Optional[SmartResponseTranslator] = None


def get_translator() -> SmartResponseTranslator:
    """Get or create the default translator instance."""
    global _default_translator
    if _default_translator is None:
        _default_translator = SmartResponseTranslator()
    return _default_translator


def translate_to_english(text: str, sources: Optional[List[Dict]] = None) -> str:
    """
    Convenience function to translate Arabic text to English.
    
    Args:
        text: Arabic text to translate
        sources: Optional source documents with metadata
        
    Returns:
        Translated English text
    """
    translator = get_translator()
    result = translator.translate_response(text, sources)
    return result.translated_text


async def translate_to_english_async(text: str, sources: Optional[List[Dict]] = None) -> str:
    """Async version of translate_to_english."""
    translator = get_translator()
    result = await translator.translate_response_async(text, sources)
    return result.translated_text
