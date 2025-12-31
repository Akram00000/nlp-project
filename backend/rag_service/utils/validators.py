"""
Validators - Input/output validation for RAG service.

Provides validation for:
- Query input validation
- Response quality checks
- Citation validation
- Safety checks
"""

from dataclasses import dataclass, field
from typing import Optional
import re
from loguru import logger


@dataclass
class ValidationResult:
    """Result of validation check."""
    
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sanitized_value: Optional[str] = None
    
    def __bool__(self) -> bool:
        return self.valid


@dataclass  
class QueryValidationConfig:
    """Configuration for query validation."""
    
    min_length: int = 3
    max_length: int = 2000
    block_patterns: list[str] = field(default_factory=list)
    require_question_marks: bool = False
    allow_empty: bool = False


@dataclass
class ResponseValidationConfig:
    """Configuration for response validation."""
    
    min_length: int = 10
    max_length: int = 10000
    require_citations: bool = True
    min_citations: int = 1
    check_hallucination_markers: bool = True


# =============================================================================
# QUERY VALIDATION
# =============================================================================

def validate_query(
    query: str,
    config: Optional[QueryValidationConfig] = None,
) -> ValidationResult:
    """
    Validate user query input.
    
    Checks:
    - Length limits
    - Blocked patterns (optional)
    - Content safety
    
    Args:
        query: User query to validate
        config: Validation configuration
        
    Returns:
        ValidationResult with status and any errors
    """
    config = config or QueryValidationConfig()
    errors = []
    warnings = []
    sanitized = query
    
    # Check empty
    if not query or not query.strip():
        if config.allow_empty:
            return ValidationResult(valid=True, sanitized_value="")
        else:
            errors.append("Query cannot be empty")
            return ValidationResult(valid=False, errors=errors)
    
    # Strip and sanitize
    sanitized = query.strip()
    
    # Check length
    if len(sanitized) < config.min_length:
        errors.append(f"Query too short (min {config.min_length} characters)")
    
    if len(sanitized) > config.max_length:
        errors.append(f"Query too long (max {config.max_length} characters)")
        sanitized = sanitized[:config.max_length]
        warnings.append("Query was truncated to max length")
    
    # Check blocked patterns
    for pattern in config.block_patterns:
        if re.search(pattern, sanitized, re.IGNORECASE):
            errors.append(f"Query contains blocked content")
            break
    
    # Remove potentially harmful characters
    sanitized = _sanitize_text(sanitized)
    
    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        sanitized_value=sanitized,
    )


def _sanitize_text(text: str) -> str:
    """Remove potentially harmful characters from text."""
    # Remove control characters except newline and space
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Normalize whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    return sanitized.strip()


# =============================================================================
# RESPONSE VALIDATION
# =============================================================================

def validate_response(
    response: str,
    sources: Optional[list] = None,
    config: Optional[ResponseValidationConfig] = None,
) -> ValidationResult:
    """
    Validate generated response quality.
    
    Checks:
    - Length limits
    - Citation presence
    - Hallucination markers
    - Quality indicators
    
    Args:
        response: Generated response to validate
        sources: Source documents used (for citation checking)
        config: Validation configuration
        
    Returns:
        ValidationResult with status and any errors/warnings
    """
    config = config or ResponseValidationConfig()
    errors = []
    warnings = []
    
    if not response or not response.strip():
        errors.append("Response is empty")
        return ValidationResult(valid=False, errors=errors)
    
    response = response.strip()
    
    # Check length
    if len(response) < config.min_length:
        warnings.append("Response seems too short")
    
    if len(response) > config.max_length:
        errors.append("Response exceeds maximum length")
    
    # Check citations
    if config.require_citations:
        citation_count = _count_citations(response)
        if citation_count < config.min_citations:
            warnings.append(f"Response has {citation_count} citations (expected at least {config.min_citations})")
    
    # Check for hallucination markers
    if config.check_hallucination_markers:
        hallucination_warnings = _check_hallucination_markers(response)
        warnings.extend(hallucination_warnings)
    
    # Check response quality
    quality_warnings = _check_response_quality(response)
    warnings.extend(quality_warnings)
    
    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        sanitized_value=response,
    )


def _count_citations(text: str) -> int:
    """Count citation markers [1], [2], etc. in text."""
    pattern = r'\[(\d+)\]'
    matches = re.findall(pattern, text)
    return len(set(matches))  # Count unique citations


def _check_hallucination_markers(text: str) -> list[str]:
    """Check for phrases that might indicate hallucination."""
    warnings = []
    
    # Phrases that suggest making things up
    hallucination_markers = [
        r"I think|I believe|in my opinion",  # AI shouldn't opinionate
        r"I don't have.*information",  # Should be handled differently
        r"I cannot.*access",  # Shouldn't mention capabilities
        r"As an AI",  # Meta-references
    ]
    
    for marker in hallucination_markers:
        if re.search(marker, text, re.IGNORECASE):
            warnings.append(f"Response contains potentially problematic phrase")
            break
    
    return warnings


def _check_response_quality(text: str) -> list[str]:
    """Check general response quality indicators."""
    warnings = []
    
    # Check for very short sentences (might be incomplete)
    sentences = re.split(r'[.!?؟]', text)
    short_sentences = [s for s in sentences if len(s.strip()) < 10 and s.strip()]
    if len(short_sentences) > len(sentences) / 2:
        warnings.append("Response contains many short sentences")
    
    # Check for repetition
    words = text.lower().split()
    if len(words) > 20:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.4:
            warnings.append("Response may contain repetitive content")
    
    return warnings


# =============================================================================
# CITATION VALIDATION
# =============================================================================

def validate_citations(
    citations: list[dict],
    response_text: str,
) -> ValidationResult:
    """
    Validate citations against response.
    
    Checks:
    - All citations in response are in citation list
    - Citations have required fields
    - No orphan citations
    
    Args:
        citations: List of citation dictionaries
        response_text: The response text containing citations
        
    Returns:
        ValidationResult
    """
    errors = []
    warnings = []
    
    if not citations:
        warnings.append("No citations provided")
        return ValidationResult(valid=True, warnings=warnings)
    
    # Get citations referenced in text
    text_citations = set(re.findall(r'\[(\d+)\]', response_text))
    
    # Get available citation indices
    available_citations = {str(c.get('index', i+1)) for i, c in enumerate(citations)}
    
    # Check for citations in text that don't exist
    invalid_refs = text_citations - available_citations
    if invalid_refs:
        errors.append(f"Response references non-existent citations: {invalid_refs}")
    
    # Check for unused citations (warning only)
    unused_citations = available_citations - text_citations
    if unused_citations:
        warnings.append(f"Some citations are not referenced in response: {unused_citations}")
    
    # Validate citation structure
    for i, citation in enumerate(citations):
        if not citation.get('formatted') and not citation.get('source'):
            warnings.append(f"Citation {i+1} is missing formatted text or source")
    
    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# =============================================================================
# SAFETY VALIDATION
# =============================================================================

def check_content_safety(text: str) -> ValidationResult:
    """
    Check text for potentially unsafe content.
    
    For Islamic content, this includes:
    - Extreme/takfiri content
    - Content encouraging harm
    - Inappropriate content
    
    Args:
        text: Text to check
        
    Returns:
        ValidationResult
    """
    errors = []
    warnings = []
    
    # This is a basic implementation
    # In production, consider using a proper content moderation API
    
    # Check for potentially dangerous topics that need scholar consultation
    sensitive_topics = [
        (r'تكفير|كافر|مرتد', 'Content discusses takfir - ensure proper scholarly context'),
        (r'جهاد.*قتال|قتل.*كفار', 'Content discusses sensitive jihad topics'),
        (r'طلاق.*بائن|لعان', 'Content discusses final divorce - recommend scholar consultation'),
    ]
    
    for pattern, warning in sensitive_topics:
        if re.search(pattern, text, re.IGNORECASE):
            warnings.append(warning)
    
    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# =============================================================================
# COMBINED VALIDATION
# =============================================================================

def validate_rag_output(
    query: str,
    response: str,
    citations: list[dict],
    sources: list = None,
) -> dict:
    """
    Perform comprehensive validation of RAG output.
    
    Args:
        query: Original user query
        response: Generated response
        citations: List of citations
        sources: Source documents used
        
    Returns:
        Dictionary with validation results for all components
    """
    results = {
        'query': validate_query(query),
        'response': validate_response(response, sources),
        'citations': validate_citations(citations, response),
        'safety': check_content_safety(response),
    }
    
    # Overall validity
    results['overall_valid'] = all(r.valid for r in results.values())
    
    # Collect all warnings
    results['all_warnings'] = []
    for key, result in results.items():
        if hasattr(result, 'warnings'):
            results['all_warnings'].extend(result.warnings)
    
    return results
