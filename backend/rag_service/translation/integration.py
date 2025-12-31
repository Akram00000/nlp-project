"""
Translation Integration - Hooks for integrating translation into RAG pipeline.

Provides middleware-style hooks for:
- Translating queries before retrieval
- Translating responses after generation
- Coordinating with WSD service (Iyed's component)
"""

from typing import Optional, Callable
from dataclasses import dataclass
from loguru import logger

from .translation_interface import (
    TranslationProvider,
    TranslationResult,
    Language,
    MockTranslationProvider,
)


@dataclass
class TranslationPipelineConfig:
    """Configuration for translation in RAG pipeline."""
    
    # Whether to auto-translate queries to Arabic for retrieval
    translate_query_to_arabic: bool = True
    
    # Whether to translate response based on query language
    translate_response: bool = True
    
    # Force output language (None = match query language)
    force_output_language: Optional[Language] = None
    
    # Apply WSD before translation (coordinate with Iyed)
    apply_wsd_before_translation: bool = False


class TranslationIntegration:
    """
    Integration layer for translation in RAG pipeline.
    
    Hooks:
    1. pre_retrieval: Translate query to Arabic for better retrieval
    2. post_generation: Translate response to user's language
    3. wsd_hook: Placeholder for WSD integration (Iyed's service)
    
    Usage:
    ```python
    integration = TranslationIntegration()
    
    # In RAG pipeline:
    processed_query = integration.pre_retrieval_hook(user_query)
    # ... retrieval and generation ...
    final_response = integration.post_generation_hook(response, original_query)
    ```
    """
    
    def __init__(
        self,
        provider: Optional[TranslationProvider] = None,
        config: Optional[TranslationPipelineConfig] = None,
    ):
        """
        Initialize translation integration.
        
        Args:
            provider: Translation provider (uses mock if not provided)
            config: Pipeline configuration
        """
        self.provider = provider or MockTranslationProvider()
        self.config = config or TranslationPipelineConfig()
        
        # WSD hook placeholder - Iyed will provide implementation
        self._wsd_hook: Optional[Callable[[str], str]] = None
    
    def set_provider(self, provider: TranslationProvider):
        """Set translation provider."""
        self.provider = provider
        logger.info(f"Translation provider set: {type(provider).__name__}")
    
    def set_wsd_hook(self, hook: Callable[[str], str]):
        """
        Set WSD (Word Sense Disambiguation) hook.
        
        Iyed should call this to integrate WSD service:
        ```python
        def wsd_process(text: str) -> str:
            # Apply WSD to disambiguate classical Arabic terms
            return disambiguated_text
        
        integration.set_wsd_hook(wsd_process)
        ```
        """
        self._wsd_hook = hook
        logger.info("WSD hook registered")
    
    def detect_language(self, text: str) -> Language:
        """Detect language of text."""
        return self.provider.detect_language(text)
    
    def pre_retrieval_hook(
        self,
        query: str,
        target_language: Language = Language.ARABIC,
    ) -> tuple[str, Language]:
        """
        Process query before retrieval.
        
        Steps:
        1. Detect query language
        2. Apply WSD if configured (for Arabic queries)
        3. Translate to Arabic if needed (for better retrieval)
        
        Args:
            query: User's query
            target_language: Language for retrieval (usually Arabic)
            
        Returns:
            Tuple of (processed_query, original_language)
        """
        # Detect original language
        original_language = self.detect_language(query)
        processed_query = query
        
        logger.debug(f"Query language detected: {original_language.value}")
        
        # Apply WSD for Arabic text
        if original_language == Language.ARABIC and self._wsd_hook and self.config.apply_wsd_before_translation:
            try:
                processed_query = self._wsd_hook(processed_query)
                logger.debug("WSD applied to query")
            except Exception as e:
                logger.warning(f"WSD hook failed: {e}")
        
        # Translate to Arabic if configured and query is in English
        if (
            self.config.translate_query_to_arabic 
            and original_language == Language.ENGLISH
            and target_language == Language.ARABIC
        ):
            try:
                result = self.provider.translate_en_to_ar(processed_query)
                processed_query = result.translated_text
                logger.debug(f"Query translated to Arabic: {processed_query[:50]}...")
            except Exception as e:
                logger.warning(f"Query translation failed: {e}. Using original query.")
        
        return processed_query, original_language
    
    async def pre_retrieval_hook_async(
        self,
        query: str,
        target_language: Language = Language.ARABIC,
    ) -> tuple[str, Language]:
        """Async version of pre_retrieval_hook."""
        original_language = self.detect_language(query)
        processed_query = query
        
        if original_language == Language.ARABIC and self._wsd_hook and self.config.apply_wsd_before_translation:
            try:
                processed_query = self._wsd_hook(processed_query)
            except Exception as e:
                logger.warning(f"WSD hook failed: {e}")
        
        if (
            self.config.translate_query_to_arabic 
            and original_language == Language.ENGLISH
            and target_language == Language.ARABIC
        ):
            try:
                result = await self.provider.translate_en_to_ar_async(processed_query)
                processed_query = result.translated_text
            except Exception as e:
                logger.warning(f"Query translation failed: {e}")
        
        return processed_query, original_language
    
    def post_generation_hook(
        self,
        response: str,
        original_query_language: Language,
        force_language: Optional[Language] = None,
    ) -> str:
        """
        Process response after generation.
        
        Steps:
        1. Determine target language (original query language or forced)
        2. Translate response if needed
        
        Args:
            response: Generated response (usually in Arabic)
            original_query_language: Language of original query
            force_language: Force specific output language
            
        Returns:
            Processed response in target language
        """
        if not self.config.translate_response:
            return response
        
        # Determine target language
        target_language = (
            force_language 
            or self.config.force_output_language 
            or original_query_language
        )
        
        # Detect response language
        response_language = self.detect_language(response)
        
        # No translation needed if already in target language
        if response_language == target_language:
            return response
        
        # Translate
        try:
            result = self.provider.translate(
                response,
                source_language=response_language,
                target_language=target_language,
            )
            logger.debug(f"Response translated to {target_language.value}")
            return result.translated_text
        except Exception as e:
            logger.warning(f"Response translation failed: {e}. Returning original.")
            return response
    
    async def post_generation_hook_async(
        self,
        response: str,
        original_query_language: Language,
        force_language: Optional[Language] = None,
    ) -> str:
        """Async version of post_generation_hook."""
        if not self.config.translate_response:
            return response
        
        target_language = (
            force_language 
            or self.config.force_output_language 
            or original_query_language
        )
        
        response_language = self.detect_language(response)
        
        if response_language == target_language:
            return response
        
        try:
            result = await self.provider.translate_async(
                response,
                source_language=response_language,
                target_language=target_language,
            )
            return result.translated_text
        except Exception as e:
            logger.warning(f"Response translation failed: {e}")
            return response
    
    def translate_citations(
        self,
        citations: list[dict],
        target_language: Language,
    ) -> list[dict]:
        """
        Translate citation metadata to target language.
        
        Args:
            citations: List of citation dictionaries
            target_language: Target language for translation
            
        Returns:
            Citations with translated fields
        """
        if target_language == Language.ARABIC:
            # Citations are likely already in Arabic
            return citations
        
        translated_citations = []
        for citation in citations:
            translated = citation.copy()
            
            # Translate key fields
            fields_to_translate = ["formatted", "scholar", "source"]
            for field in fields_to_translate:
                if field in translated and translated[field]:
                    try:
                        result = self.provider.translate_ar_to_en(str(translated[field]))
                        translated[f"{field}_translated"] = result.translated_text
                    except Exception:
                        pass
            
            translated_citations.append(translated)
        
        return translated_citations
    
    def health_check(self) -> dict:
        """Check health of translation components."""
        return {
            "translation_provider": self.provider.health_check(),
            "wsd_hook_registered": self._wsd_hook is not None,
            "provider_type": type(self.provider).__name__,
        }


# ============================================================================
# INTEGRATION GUIDE FOR IYED (WSD)
# ============================================================================
"""
INTEGRATION GUIDE FOR IYED - Word Sense Disambiguation
=======================================================

To integrate WSD with the translation pipeline:

1. Implement WSD service that processes Arabic text:

```python
def disambiguate_text(text: str) -> str:
    '''
    Apply Word Sense Disambiguation to Arabic text.
    
    Focus on:
    - Classical Arabic terms (فقه, عقيدة, etc.)
    - Ambiguous words with multiple meanings
    - Context-dependent terminology
    
    Returns:
        Text with disambiguated terms (possibly with annotations)
    '''
    # Your WSD implementation
    return processed_text
```

2. Register with TranslationIntegration:

```python
from rag_service.translation import TranslationIntegration

integration = TranslationIntegration()
integration.set_wsd_hook(disambiguate_text)
```

3. Configure to use WSD:

```python
from rag_service.translation import TranslationPipelineConfig

config = TranslationPipelineConfig(
    apply_wsd_before_translation=True
)
integration = TranslationIntegration(config=config)
```

The WSD hook will be called:
- Before query translation (Arabic queries)
- Can be extended for response processing

Coordination Points:
- Arabic preprocessing (share with Redhouane's preprocessing module)
- Classical Arabic term dictionary
- Context window for disambiguation
"""
