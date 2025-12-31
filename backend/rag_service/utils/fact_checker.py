"""
Fact Checker

Validates generated responses against retrieved sources to detect hallucinations
and ensuring provenance compliance.
"""

from typing import List, Dict, Optional
import re
from loguru import logger
from ..core.retriever import RetrievalResult

# Helper for prefix matching
def citation_matches_source(citation: str, source_id: str) -> bool:
    """
    Check if a citation matches a source ID.
    Accepts exact match or prefix match.
    
    Examples:
        citation_matches_source("fatwa_fatwas_0", "fatwa_fatwas_0_72b686") → True
        citation_matches_source("fatwa_fatwas_0_72b686", "fatwa_fatwas_0_72b686") → True
        citation_matches_source("fatwa_fatwas_1", "fatwa_fatwas_0_72b686") → False
    """
    if not citation or not source_id:
        return False
    
    # Exact match
    if citation == source_id:
        return True
    
    # Prefix match: citation is a prefix of source_id
    # This handles cases like "fatwa_fatwas_0" matching "fatwa_fatwas_0_72b686"
    if source_id.startswith(citation + "_") or source_id.startswith(citation):
        return True
    
    return False

class FactChecker:
    """Check generated content against retrieved context."""
    
    def __init__(self):
        # Extract persistent IDs: source_type_collection_index_hash
        self.citation_pattern = re.compile(r'\((\w+_\w+_\d+_[a-f0-9]+)\)', re.IGNORECASE)
        self.quran_pattern = re.compile(r'(?:سورة\s+)?(\w+)\s*[:\-]\s*(\d+)', re.UNICODE)
        
    def check_response(
        self, 
        response: str, 
        documents: List[RetrievalResult]
    ) -> Dict:
        """
        Validate response against documents with RELAXED rules (Fix #1 + Prefix Fix).
        Allows fuzzy matching and content verification.
        """
        issues = []
        cited_ids = set()
        
        # 1. Build valid ID set & Content Map from sources
        valid_ids = set()
        source_contents = []
        source_ids_list = [] # For logging/lookup
        
        for i, doc in enumerate(documents):
            # ID variants
            ids_for_doc = set()
            ids_for_doc.add(doc.id)
            if doc.enriched_metadata and doc.enriched_metadata.citation_id:
                ids_for_doc.add(doc.enriched_metadata.citation_id)
            
            valid_ids.update(ids_for_doc)
            source_ids_list.extend(list(ids_for_doc))
            
            # Positional variants
            valid_ids.add(f"Source {i+1}")
            valid_ids.add(f"المصدر {i+1}")
            
            source_contents.append(doc.content)

        # 2. Extract citations (Enhanced regex Fix #8.1)
        citation_pattern = r'\[((?:fatwa|book|hadith|tafsir|quran)_[^\]]+|Source \d+|المصدر \d+)\]'
        raw_citations = re.findall(citation_pattern, response)
        
        logger.debug(f"Retrieved Source IDs: {source_ids_list}")
        logger.debug(f"Citations extracted from response: {raw_citations}")
        
        if not raw_citations:
            logger.info("No citations found in answer")
        
        for citation in raw_citations:
            cited_ids.add(citation)
            matched = False
            
            # Direct match check (Fast path)
            if citation in valid_ids:
                logger.debug(f"✓ Citation [{citation}] matched EXACTLY (Found in valid_ids)")
                matched = True
                continue
            
            # Positional match fallback
            source_num_match = re.match(r'(?:Source|المصدر)\s*(\d+)', citation)
            if source_num_match:
                idx = int(source_num_match.group(1)) - 1
                if 0 <= idx < len(documents):
                    logger.debug(f"✓ Citation [{citation}] matched POSITIONAL (Index {idx})")
                    matched = True
                    continue
                else:
                    logger.warning(f"⚠ Citation [{citation}] POSITIONAL check failed: Index {idx} out of range (0-{len(documents)-1})")

            # Prefix Matching Check (The Fix)
            for source_id in source_ids_list:
                if citation_matches_source(citation, source_id):
                    logger.info(f"✓ Citation [{citation}] matched source [{source_id}] (PREFIX match)")
                    matched = True
                    break
            
            if matched:
                continue
            
            # NEW FALLBACK: Content verification check (Fix #1 Requirement)
            # Find the sentence containing this citation
            # Look for ~100 chars before the [citation] to find the claim
            pattern = re.escape(f"[{citation}]")
            match_pos = re.search(pattern, response)
            if match_pos:
                start_idx = max(0, match_pos.start() - 150)
                # Take everything from start_idx to the citation
                claim_text = response[start_idx:match_pos.start()].strip()
                # Clean up claim_text: remove markdown and multiple spaces
                claim_text = re.sub(r'[*#_>-]', '', claim_text)
                # Split into sentences or take the last sentence
                sentences = re.split(r'[.!?;]\s*|\n', claim_text)
                last_sentence = sentences[-1].strip() if sentences else ""
                
                if len(last_sentence) > 15: # Only check significant claims
                    for doc in documents:
                        # Check for fuzzy overlap or direct inclusion
                        # We use a simple high-recall check: does the sentence appear or share many keywords?
                        # For now, sub-string check if the sentence is long enough
                        if last_sentence in doc.content or doc.content in last_sentence:
                            logger.info(f"✓ Citation [{citation}] verified by CONTENT match in Doc {doc.id}")
                            matched = True
                            break
                        
                        # Keyword overlap fallback
                        keywords = [h for h in last_sentence.split() if len(h) > 3]
                        if keywords:
                            match_count = sum(1 for k in keywords if k in doc.content)
                            overlap = match_count / len(keywords)
                            if overlap > 0.7:
                                logger.info(f"✓ Citation [{citation}] verified by KEYWORD overlap ({overlap:.1%}) in Doc {doc.id}")
                                matched = True
                                break
                else:
                    logger.debug(f"Citation [{citation}] claim too short for content verification: '{last_sentence}'")

            if matched:
                continue
            
            # Final failure
            issues.append(f"Citation {citation} not found in retrieval (checked exact, prefix, positional, and content).")

        # 3. Quran Verification (Stricter Regex Fix #10.1)
        # Whitelist of 114 Surahs
        SURAHS = [
            "الفاتحة", "البقرة", "آل عمران", "النساء", "المائدة", "الأنعام", "الأعراف", "الأنفال", "التوبة", "يونس",
            "هود", "يوسف", "الرعد", "إبراهيم", "الحجر", "النحل", "الإسراء", "الكهف", "مريم", "طه",
            "الأنبياء", "الحج", "المؤمنون", "النور", "الفرقان", "الشعراء", "النمل", "القصص", "العنكبوت", "الروم",
            "لقمان", "السجدة", "الأحزاب", "سبأ", "فاطر", "يس", "الصافات", "ص", "الزمر", "غافر",
            "فصلت", "الشورى", "الزخرف", "الدخان", "الجاثية", "الأحقاف", "محمد", "الفتح", "الحجرات", "ق",
            "الذاريات", "الطور", "النجم", "القمر", "الرحمن", "الواقعة", "الحديد", "المجادلة", "الحشر", "الممتحنة",
            "الصف", "الجمعة", "المنافقون", "التغابن", "الطلاق", "التحريم", "الملك", "القلم", "الحاقة", "المعارج",
            "نوح", "الجن", "المزمل", "المدثر", "القيامة", "الإنسان", "المرسلات", "النبأ", "النازعات", "عبس",
            "التكوير", "الانفطار", "المطففين", "الانشقاق", "البروج", "الطارق", "الأعلى", "الغاشية", "الفجر", "البلد",
            "الشمس", "الليل", "الضحى", "الشرح", "التين", "العلق", "القدر", "البينة", "الزلزلة", "العاديات",
            "القارعة", "التكاثر", "العصر", "الهمزة", "الفيل", "قريش", "الماعون", "الكوثر", "الكافرون", "النصر",
            "المسد", "الإخلاص", "الفلق", "الناس"
        ]
        surah_whitelist = "|".join(SURAHS)
        # Regex that ONLY matches [Surah:Ayah] or [سورة Surah:Ayah]
        quran_strict_pattern = re.compile(rf'(?:سورة\s+)?({surah_whitelist})\s*[:\-]\s*(\d+)', re.UNICODE)
        
        verses_in_response = quran_strict_pattern.findall(response)
        logger.debug(f"Quran citations detected for validation: {verses_in_response}")
        
        for surah, ayah in verses_in_response:
             found = False
             # Check metadata
             for doc in documents:
                 if doc.enriched_metadata and doc.enriched_metadata.verse_ref:
                     if f"{surah}" in doc.enriched_metadata.verse_ref and f"{ayah}" in doc.enriched_metadata.verse_ref:
                         found = True
                         break
             
             # Fallback: Check content text
             if not found:
                 for content in source_contents:
                     if surah in content and ayah in content:
                         found = True
                         break
            
             if not found:
                 issues.append(f"Answer cites Quran verse {surah}:{ayah} not found in sources.")

        valid = len(issues) == 0
        hallucination_score = 1.0 if not valid else 0.0
        
        if valid:
            logger.info(f"✅ All citations validated successfully")
        else:
            logger.error(f"❌ Validation failed: {issues}")
        
        return {
            "valid": valid,
            "issues": issues,
            "cited_ids": list(cited_ids),
            "hallucination_score": hallucination_score
        }
