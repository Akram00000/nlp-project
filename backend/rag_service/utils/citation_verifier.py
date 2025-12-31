"""
Citation Verifier Module - Detect and fix hallucinated citations.

Scans generated answers for fabricated fatwa numbers, hadith references,
and other citations that don't exist in the retrieved sources.
"""

import re
from typing import List, Dict, Tuple
from loguru import logger


def verify_answer_citations(
    generated_answer: str,
    retrieved_sources: List[Dict],
) -> Tuple[str, List[str]]:
    """
    Scan the generated answer for fabricated citations and replace them.
    """
    if not generated_answer or not retrieved_sources:
        return generated_answer, []
    
    # Combine all source text for checking
    all_source_text = " ".join([
        s.get("text", "") + " " + str(s.get("metadata", {}))
        for s in retrieved_sources
    ])
    
    hallucinations = []
    corrected_answer = generated_answer
    
    # Pattern 1: Arabic fatwa number references (more robust)
    fatwa_patterns = [
        (r'فتوى رقم[:\s]*(\d+)', 'فتوى من المصادر'),
        (r'الفتوى رقم[:\s]*(\d+)', 'الفتوى من المصادر'),
        (r'رقم القرض[:\s]*(\d+)', 'القرض المذكور'),
        (r'رقم الفتوى[:\s]*(\d+)', 'المصدر'),
        (r'رقم[:\s]*(\d+)', 'رقم من المصادر'),
    ]
    
    for pattern, replacement in fatwa_patterns:
        matches = re.finditer(pattern, corrected_answer)
        # Iterate backwards to avoid index shifts if needed, 
        # but re.sub handles it if we are careful.
        # Here we use re.sub with a custom function for verification.
        
        def replace_if_fake(match):
            ref_num = match.group(1)
            if ref_num not in all_source_text:
                hallucinations.append(f"رقم {ref_num}")
                # Replace with the generic part of the match, keeping non-digit parts if possible
                full_match = match.group(0)
                return full_match.replace(ref_num, "المذكور في المصادر")
            return match.group(0)

        corrected_answer = re.sub(pattern, replace_if_fake, corrected_answer)
    
    # Pattern 2: Hadith number references
    hadith_patterns = [
        (r'حديث رقم[:\s]*(\d+)', 'الحديث المذكور'),
        (r'رقم الحديث[:\s]*(\d+)', 'الحديث المذكور'),
    ]
    
    for pattern, replacement in hadith_patterns:
        def replace_if_fake_hadith(match):
            ref_num = match.group(1)
            if ref_num not in all_source_text:
                hallucinations.append(f"حديث رقم {ref_num}")
                return replacement
            return match.group(0)
            
        corrected_answer = re.sub(pattern, replace_if_fake_hadith, corrected_answer)

    # ANTI-HALLUCINATION: Logical Contradictions
    contradictions = [
        (r'قرض ربوي بدون فائدة', 'قرض بدون فائدة'),
        (r'قرض ربوي لا فائدة فيه', 'قرض لا فائدة فيه'),
        (r'ربا بدون فائدة', 'معاملة بدون فائدة'),
        (r'قرض ربوي حلال', 'قرض مباح'),
    ]
    
    for pattern, replacement in contradictions:
        if re.search(pattern, corrected_answer):
            logger.warning(f"Logical contradiction detected: {pattern}")
            corrected_answer = re.sub(pattern, replacement, corrected_answer)
            hallucinations.append(f"Contradiction: {pattern}")
    
    if hallucinations:
        logger.info(f"Hallucinations/Contradictions corrected: {len(hallucinations)}")
    
    return corrected_answer, hallucinations


def extract_valid_source_ids(
    generated_answer: str,
    valid_chunk_ids: List[str],
) -> List[str]:
    """
    Extract source IDs mentioned in the answer and verify they are valid.
    
    Args:
        generated_answer: The LLM-generated answer
        valid_chunk_ids: List of valid chunk_ids from retrieved sources
        
    Returns:
        List of valid source IDs found in the answer
    """
    # Pattern: [source:CHUNK_ID]text[/source]
    source_pattern = r'\[source:([^\]]+)\]'
    mentioned_ids = re.findall(source_pattern, generated_answer)
    
    valid_mentioned = []
    for source_id in mentioned_ids:
        if source_id in valid_chunk_ids:
            valid_mentioned.append(source_id)
        else:
            logger.warning(f"Invalid source ID in answer: {source_id}")
    
    return valid_mentioned


def convert_source_links_to_html(answer: str, sources: List[Dict]) -> str:
    """
    Convert [source:ID]text[/source] markers to clickable HTML links.
    
    Args:
        answer: Answer text with source markers
        sources: List of source dicts for metadata lookup
        
    Returns:
        Answer with HTML anchor tags
    """
    # Build source lookup
    source_lookup = {s.get('chunk_id', ''): s for s in sources}
    
    def replace_source_tag(match):
        source_id = match.group(1)
        display_text = match.group(2)
        
        # Get source metadata for tooltip
        source = source_lookup.get(source_id, {})
        source_type = source.get('source_type', source.get('metadata', {}).get('type', 'unknown'))
        
        # Create clickable link with data attributes
        return f'<a href="#" class="source-link" data-source-id="{source_id}" data-source-type="{source_type}" onclick="showSourceModal(\'{source_id}\'); return false;">{display_text}</a>'
    
    # Pattern: [source:ID]text[/source]
    pattern = r'\[source:([^\]]+)\]([^\[]+)\[/source\]'
    converted = re.sub(pattern, replace_source_tag, answer)
    
    return converted


def build_source_context_for_prompt(sources: List[Dict]) -> str:
    """
    Build a structured source context string for the LLM prompt.
    
    Each source includes chunk_id and full metadata for proper citation.
    """
    context_parts = []
    
    for i, source in enumerate(sources):
        chunk_id = source.get('chunk_id', f'source_{i}')
        source_type = source.get('source_type', source.get('metadata', {}).get('type', 'unknown'))
        text = source.get('text', source.get('content', ''))[:1500]
        metadata = source.get('metadata', {})
        
        # Extract key metadata fields
        title = metadata.get('title', metadata.get('question', ''))[:100]
        source_name = metadata.get('source', metadata.get('website', metadata.get('source_name', '')))
        author = metadata.get('author', metadata.get('scholar', ''))
        collection = metadata.get('collection', '')
        
        context_parts.append(f"""
[Source {i+1}]
chunk_id: {chunk_id}
type: {source_type}
title: {title}
source_name: {source_name}
author: {author}
collection: {collection}
---
{text}
---
""")
    
    return "\n".join(context_parts)
