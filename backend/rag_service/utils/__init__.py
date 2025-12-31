"""Utility modules for RAG service."""

from .preprocessing import (
    preprocess_arabic_text,
    normalize_arabic,
    remove_diacritics,
    detect_language,
    preprocess_ocr_text,
    fix_ocr_errors,
    fix_repeated_characters,
)
from .prompts import (
    get_system_prompt,
    build_rag_prompt,
    PromptTemplates,
)
from .validators import (
    validate_query,
    validate_response,
    ValidationResult,
)

__all__ = [
    # Preprocessing
    "preprocess_arabic_text",
    "normalize_arabic", 
    "remove_diacritics",
    "detect_language",
    "preprocess_ocr_text",
    "fix_ocr_errors",
    "fix_repeated_characters",
    # Prompts
    "get_system_prompt",
    "build_rag_prompt",
    "PromptTemplates",
    # Validators
    "validate_query",
    "validate_response",
    "ValidationResult",
]
