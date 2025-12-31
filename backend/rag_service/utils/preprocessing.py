"""
Arabic Text Preprocessing - Normalization and cleaning utilities.

Handles:
- Arabic text normalization
- Diacritics (tashkeel) removal/handling  
- Character standardization
- Language detection
"""

import re
from typing import Optional


# Arabic Unicode ranges
ARABIC_RANGE = '\u0600-\u06FF'
ARABIC_SUPPLEMENT = '\u0750-\u077F'
ARABIC_EXTENDED_A = '\u08A0-\u08FF'
ARABIC_PRESENTATION_A = '\uFB50-\uFDFF'
ARABIC_PRESENTATION_B = '\uFE70-\uFEFF'

# Diacritics (Tashkeel)
DIACRITICS = '\u064B-\u0652\u0670'  # Fatha, Damma, Kasra, Shadda, Sukun, etc.

# Common character mappings for normalization
ALEF_VARIANTS = {
    '\u0622': '\u0627',  # Alef with Madda -> Alef
    '\u0623': '\u0627',  # Alef with Hamza above -> Alef
    '\u0625': '\u0627',  # Alef with Hamza below -> Alef
    '\u0671': '\u0627',  # Alef Wasla -> Alef
}

TEH_MARBUTA_MAP = {
    '\u0629': '\u0647',  # Teh Marbuta -> Heh (optional)
}

YEH_VARIANTS = {
    '\u064A': '\u064A',  # Yeh (keep)
    '\u0649': '\u064A',  # Alef Maksura -> Yeh (for searching)
    '\u06CC': '\u064A',  # Farsi Yeh -> Arabic Yeh
}


def remove_diacritics(text: str) -> str:
    """
    Remove Arabic diacritics (tashkeel) from text.
    
    Removes: Fatha, Damma, Kasra, Sukun, Shadda, Tanween, etc.
    
    Args:
        text: Arabic text with diacritics
        
    Returns:
        Text without diacritics
    """
    return re.sub(f'[{DIACRITICS}]', '', text)


def normalize_alef(text: str) -> str:
    """Normalize Alef variants to plain Alef."""
    for variant, normalized in ALEF_VARIANTS.items():
        text = text.replace(variant, normalized)
    return text


def normalize_yeh(text: str) -> str:
    """Normalize Yeh variants."""
    for variant, normalized in YEH_VARIANTS.items():
        text = text.replace(variant, normalized)
    return text


def normalize_teh_marbuta(text: str, convert_to_heh: bool = False) -> str:
    """
    Handle Teh Marbuta.
    
    Args:
        text: Arabic text
        convert_to_heh: If True, converts to Heh (useful for some search scenarios)
    """
    if convert_to_heh:
        return text.replace('\u0629', '\u0647')
    return text


def remove_tatweel(text: str) -> str:
    """Remove Tatweel (kashida) character used for text stretching."""
    return text.replace('\u0640', '')


def normalize_arabic(
    text: str,
    remove_tashkeel: bool = True,
    normalize_alef_variants: bool = True,
    normalize_yeh_variants: bool = True,
    remove_tatweel_char: bool = True,
) -> str:
    """
    Apply comprehensive Arabic text normalization.
    
    Args:
        text: Arabic text to normalize
        remove_tashkeel: Remove diacritics
        normalize_alef_variants: Unify Alef forms
        normalize_yeh_variants: Unify Yeh forms
        remove_tatweel_char: Remove kashida stretching
        
    Returns:
        Normalized Arabic text
    """
    if not text:
        return text
    
    result = text
    
    if remove_tashkeel:
        result = remove_diacritics(result)
    
    if normalize_alef_variants:
        result = normalize_alef(result)
    
    if normalize_yeh_variants:
        result = normalize_yeh(result)
    
    if remove_tatweel_char:
        result = remove_tatweel(result)
    
    return result


def clean_arabic_text(text: str) -> str:
    """
    Clean Arabic text for processing.
    
    - Removes extra whitespace
    - Normalizes newlines
    - Removes control characters
    """
    if not text:
        return text
    
    # Remove control characters except newline and tab
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n+', '\n', text)    # Multiple newlines to single
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def preprocess_arabic_text(
    text: str,
    normalize: bool = True,
    clean: bool = True,
    for_embedding: bool = False,
) -> str:
    """
    Full preprocessing pipeline for Arabic text.
    
    Args:
        text: Arabic text to process
        normalize: Apply normalization
        clean: Apply cleaning
        for_embedding: Optimize for embedding (more aggressive normalization)
        
    Returns:
        Preprocessed text
    """
    if not text:
        return text
    
    result = text
    
    if clean:
        result = clean_arabic_text(result)
    
    if normalize:
        result = normalize_arabic(
            result,
            remove_tashkeel=True,
            normalize_alef_variants=True,
            normalize_yeh_variants=for_embedding,  # More aggressive for embeddings
            remove_tatweel_char=True,
        )
    
    return result


def detect_language(text: str) -> str:
    """
    Detect if text is primarily Arabic or English.
    
    Args:
        text: Text to analyze
        
    Returns:
        'ar' for Arabic, 'en' for English
    """
    if not text:
        return 'en'
    
    # Count Arabic characters
    arabic_pattern = f'[{ARABIC_RANGE}{ARABIC_SUPPLEMENT}{ARABIC_PRESENTATION_A}{ARABIC_PRESENTATION_B}]'
    arabic_chars = len(re.findall(arabic_pattern, text))
    
    # Count Latin characters
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    
    total_alpha = arabic_chars + latin_chars
    if total_alpha == 0:
        return 'en'  # Default to English if no alphabetic chars
    
    arabic_ratio = arabic_chars / total_alpha
    
    return 'ar' if arabic_ratio > 0.5 else 'en'


def is_arabic(text: str) -> bool:
    """Check if text is primarily Arabic."""
    return detect_language(text) == 'ar'


def extract_arabic_words(text: str) -> list[str]:
    """Extract Arabic words from text."""
    arabic_pattern = f'[{ARABIC_RANGE}]+'
    return re.findall(arabic_pattern, text)


def get_arabic_stopwords() -> set[str]:
    """
    Get common Arabic stopwords.
    
    Useful for filtering in search/retrieval scenarios.
    """
    return {
        # Common prepositions and particles
        'من', 'إلى', 'على', 'في', 'عن', 'مع', 'بين', 'حتى',
        # Articles
        'ال', 'الذي', 'التي', 'الذين', 'اللواتي',
        # Pronouns
        'هو', 'هي', 'هم', 'هن', 'أنا', 'نحن', 'أنت', 'أنتم',
        # Question words
        'ما', 'ماذا', 'من', 'متى', 'أين', 'كيف', 'لماذا', 'هل',
        # Conjunctions
        'و', 'أو', 'ثم', 'لكن', 'بل', 'إن', 'أن', 'لأن',
        # Common verbs
        'كان', 'يكون', 'كانت', 'كانوا', 'ليس', 'ليست',
        # Demonstratives
        'هذا', 'هذه', 'ذلك', 'تلك', 'هؤلاء', 'أولئك',
        # Other common words
        'كل', 'بعض', 'غير', 'قد', 'لا', 'نعم', 'إذا', 'لو',
    }


def remove_stopwords(text: str, stopwords: Optional[set[str]] = None) -> str:
    """
    Remove stopwords from Arabic text.
    
    Args:
        text: Arabic text
        stopwords: Custom stopwords set (uses default if None)
        
    Returns:
        Text with stopwords removed
    """
    if stopwords is None:
        stopwords = get_arabic_stopwords()
    
    words = text.split()
    filtered = [w for w in words if w not in stopwords]
    
    return ' '.join(filtered)


# ============================================================================
# OCR Preprocessing Functions (for book ingestion)
# ============================================================================

# Common OCR errors in Arabic text
OCR_CORRECTIONS = {
    # ه / ة confusion (very common)
    'ه ': 'ة ',  # Isolated heh at end of word often should be teh marbuta
    # ي / ى confusion
    'ي ': 'ى ',  # Ya at word end often should be alef maksura (context-dependent)
    # Common ligature issues
    'لأ': 'لا',  # Lam-alef variants
    'ﻻ': 'لا',
    'ﻷ': 'لأ',
    'ﻹ': 'لإ',
    # Number/character confusion
    '٠': '۰',  # Arabic-Indic digits normalization
}

# Repeated character patterns (OCR artifacts)
REPEATED_CHAR_PATTERN = re.compile(r'(.)\1{3,}')  # 4+ repeated chars


def fix_ocr_errors(text: str) -> str:
    """
    Fix common OCR errors in Arabic text.
    
    Args:
        text: OCR'd Arabic text
        
    Returns:
        Text with common OCR errors corrected
    """
    if not text:
        return text
    
    result = text
    
    # Apply known corrections
    for wrong, correct in OCR_CORRECTIONS.items():
        result = result.replace(wrong, correct)
    
    return result


def fix_repeated_characters(text: str) -> str:
    """
    Fix repeated characters (common OCR artifact).
    
    E.g., "ااااا" -> "ا", "ـــــ" -> "ـ"
    
    Args:
        text: Text with potential repeated characters
        
    Returns:
        Text with repeated chars reduced
    """
    if not text:
        return text
    
    # Reduce 4+ repeated chars to 1
    return REPEATED_CHAR_PATTERN.sub(r'\1', text)


def normalize_whitespace_ocr(text: str) -> str:
    """
    Normalize whitespace in OCR text.
    
    - Removes excessive blank lines
    - Normalizes spaces around punctuation
    - Fixes common OCR spacing issues
    """
    if not text:
        return text
    
    # Multiple spaces to single space (but preserve intentional newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Multiple blank lines to single
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Fix spaces before punctuation
    text = re.sub(r'\s+([،؛:.!؟])', r'\1', text)
    
    # Add space after punctuation if missing
    text = re.sub(r'([،؛:.!؟])([^\s\n])', r'\1 \2', text)
    
    return text.strip()


def remove_page_artifacts(text: str) -> str:
    """
    Remove common OCR page artifacts.
    
    - Page numbers
    - Headers/footers
    - Margin notes markers
    """
    if not text:
        return text
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Skip standalone page numbers
        if re.match(r'^[\d٠١٢٣٤٥٦٧٨٩]+$', stripped):
            continue
        
        # Skip very short lines that are likely artifacts
        if len(stripped) < 3 and not stripped.isalpha():
            continue
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def preprocess_ocr_text(
    text: str,
    fix_errors: bool = True,
    fix_repeated: bool = True,
    normalize_whitespace: bool = True,
    remove_artifacts: bool = True,
    normalize_chars: bool = True,
) -> str:
    """
    Full OCR preprocessing pipeline for Arabic book text.
    
    Applies a series of cleanup operations to improve OCR quality
    for better embedding and retrieval.
    
    Args:
        text: Raw OCR text
        fix_errors: Fix common OCR errors (ه/ة confusion, etc.)
        fix_repeated: Fix repeated character artifacts
        normalize_whitespace: Normalize spacing and newlines
        remove_artifacts: Remove page numbers and artifacts
        normalize_chars: Apply standard Arabic normalization
        
    Returns:
        Cleaned and preprocessed text
    """
    if not text:
        return text
    
    result = text
    
    # Order matters - apply in logical sequence
    if remove_artifacts:
        result = remove_page_artifacts(result)
    
    if fix_repeated:
        result = fix_repeated_characters(result)
    
    if fix_errors:
        result = fix_ocr_errors(result)
    
    if normalize_whitespace:
        result = normalize_whitespace_ocr(result)
    
    if normalize_chars:
        # Light normalization - preserve some diacritics for fiqh text
        result = normalize_arabic(
            result,
            remove_tashkeel=False,  # Keep diacritics - important for fiqh
            normalize_alef_variants=True,
            normalize_yeh_variants=False,  # Preserve ي/ى distinction
            remove_tatweel_char=True,
        )
    
    return result
