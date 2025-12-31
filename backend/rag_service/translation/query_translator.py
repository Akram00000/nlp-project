"""
Query Translation Utility

Translates English queries to Arabic for better retrieval
when using Arabic-optimized embedding models.
"""

from typing import Optional
from loguru import logger

from rag_service.providers.base import Message


def translate_query_to_arabic(
    query: str,
    llm_provider,
    detected_lang: str = "en"
) -> str:
    """
    Translate an English query to Arabic for better retrieval.
    
    Args:
        query: The user's query
        llm_provider: LLM provider instance for translation
        detected_lang: Detected language of the query
        
    Returns:
        Arabic translation if input is English, otherwise original query
    """
    # Only translate if the query is in English
    if detected_lang != "en":
        return query
    
    if llm_provider is None:
        logger.warning("No LLM provider available for translation")
        return query
    
    try:
        translation_prompt = f"""Translate the following English question about Islamic jurisprudence (fiqh) to Arabic.
Only output the Arabic translation, nothing else.

English: {query}
Arabic:"""

        messages = [
            Message(role="user", content=translation_prompt)
        ]
        
        response = llm_provider.generate(messages, max_tokens=200, temperature=0.1)
        
        translated = response.content.strip()
        
        # Basic validation - should contain Arabic characters
        if any('\u0600' <= c <= '\u06FF' for c in translated):
            logger.info(f"Translated query: '{query}' -> '{translated}'")
            return translated
        else:
            logger.warning(f"Translation did not produce Arabic text: {translated}")
            return query
            
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return query
