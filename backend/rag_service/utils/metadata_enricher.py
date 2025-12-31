"""
Metadata Enricher

Extracts and enriches metadata from retrieved passages including:
- Quranic verse numbers (Surah:Ayah)
- Hadith IDs (Collection:Number)
- Source types and provenance information
"""

import re
from typing import Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class EnrichedMetadata:
    """Enriched metadata for a retrieved passage."""
    
    # Identification
    doc_id: str  # Persistent unique ID
    source_type: str  # quran, hadith, book, fatwa, tafsir
    
    # New human-readable ID
    citation_id: Optional[str] = None 
    
    # References
    verse_ref: Optional[str] = None  # e.g., "البقرة:43"
    hadith_ref: Optional[str] = None  # e.g., "البخاري:645"
    
    # Attribution
    author: Optional[str] = None
    scholar: Optional[str] = None
    madhab: Optional[str] = None
    
    # Trust indicators
    authenticity: Optional[str] = None  # صحيح, حسن, ضعيف
    language: str = "ar"
    
    # Provenance
    collection: Optional[str] = None
    chapter: Optional[str] = None


class MetadataEnricher:
    # ... (patterns remain same)
    
    # Quranic reference patterns
    QURAN_PATTERNS = [
        # سورة النساء آية 102 or سورة آل عمران: 102
        # Supports multi-word surahs like (آل عمران) by allowing a word and an optional following word
        re.compile(r'\bسورة\s+((?:\w+\s+)?\w+)\s*(?:[:\-]|آية)\s*(\d+)\b(?:\s*-\s*(\d+))?', re.UNICODE),
        # {وَاركَعُوا} [البقرة: 43]
        re.compile(r'\[((?:\w+\s+)?\w+)\s*[:\-]\s*(\d+)\b(?:\s*-\s*(\d+))?\]', re.UNICODE),
        # النساء: 102 (at end of sentence or in context)
        # Avoid matching technical IDs
        re.compile(r'(?<!\w)((?:\w+\s+)?\w+)\s*:\s*(\d+)\b(?!\s*[_0-9a-f]{6})', re.UNICODE),
    ]
    
    # Hadith reference patterns
    HADITH_PATTERNS = [
        # رواه البخاري: 645 or رواه البخاري (645)
        re.compile(r'رواه?\s+(البخاري|مسلم|الترمذي|أبو داود|النسائي|ابن ماجه|أحمد)\s*[:\-\(\[]\s*(\d+)', re.UNICODE),
        # صحيح البخاري: 645
        re.compile(r'صحيح\s+(البخاري|مسلم)\s*[:\-]\s*(\d+)', re.UNICODE),
        # البخاري رقم 645
        re.compile(r'(البخاري|مسلم|الترمذي)\s+(?:رقم|حديث)?\s*(\d+)', re.UNICODE),
    ]
    
    # Madhab indicators
    MADHAB_PATTERNS = {
        'hanafi': re.compile(r'الحنفي|حنفي|أبو حنيفة|الأحناف', re.UNICODE),
        'maliki': re.compile(r'المالكي|مالكي|مالك\s+بن|المالكية', re.UNICODE),
        'shafii': re.compile(r'الشافعي|شافعي|الإمام الشافعي|الشافعية', re.UNICODE),
        'hanbali': re.compile(r'الحنبلي|حنبلي|ابن قدامة|أحمد\s+بن\s+حنبل|الحنابلة', re.UNICODE),
    }
    
    # Authenticity indicators
    AUTHENTICITY_PATTERNS = {
        'sahih': re.compile(r'صحيح|متفق عليه', re.UNICODE),
        'hasan': re.compile(r'حسن(?!\s+البصري)', re.UNICODE),
        'daif': re.compile(r'ضعيف', re.UNICODE),
    }
    
    def enrich(self, content: str, metadata: dict, doc_id: str) -> EnrichedMetadata:
        """
        Enrich a passage with extracted metadata.
        
        Args:
            content: The passage text
            metadata: Existing metadata from retrieval
            doc_id: Unique document ID
            
        Returns:
            EnrichedMetadata with all extracted information
        """
        source_type = metadata.get('type', 'unknown').lower()
        
        enriched = EnrichedMetadata(
            doc_id=doc_id,
            source_type=source_type,
            author=metadata.get('author'),
            scholar=metadata.get('scholar'),
            madhab=metadata.get('madhab'),
            collection=metadata.get('collection') or metadata.get('source'),
            chapter=metadata.get('chapter') or metadata.get('section'),
            language=metadata.get('language', 'ar'),
        )
        
        # Extract verse references
        verse_refs = self._extract_quran_refs(content)
        if verse_refs:
            enriched.verse_ref = verse_refs[0]  # Primary reference
        
        # Extract hadith references
        hadith_refs = self._extract_hadith_refs(content)
        if hadith_refs:
            enriched.hadith_ref = hadith_refs[0]  # Primary reference
            
        # Detect madhab if not in metadata
        if not enriched.madhab:
            enriched.madhab = self._detect_madhab(content)
            
        # Detect authenticity for hadiths
        if source_type == 'hadith':
            enriched.authenticity = self._detect_authenticity(content)
            if not enriched.authenticity:
                enriched.authenticity = self._infer_authenticity(enriched.collection, enriched.hadith_ref)
            
        # Generate Readable Citation ID
        enriched.citation_id = self._generate_citation_id(enriched, doc_id)
        
        return enriched

    def _infer_authenticity(self, collection: Optional[str], ref: Optional[str]) -> str:
        """Infer authenticity from collection name if explicit grade missing."""
        if not collection:
            # Try to guess from ref if present (e.g. "Bukhari:645")
            if ref:
                collection = ref
            else:
                return "needs_verification"
                
        text = collection.lower()
        
        # Bukhari / Muslim -> Sahih (Agreed Upon)
        if any(x in text for x in ['bukhari', 'muslim', 'البخاري', 'مسلم', 'متفق عليه']):
            return "sahih_agreed"
            
        # Sunan -> Hasan (Default assumption for RAG display as requested)
        if any(x in text for x in ['dawud', 'tirmidhi', 'nasai', 'داود', 'الترمذي', 'النسائي', 'ماجه']):
            return "hasan"
            
        return "needs_verification"

    def _generate_citation_id(self, meta: EnrichedMetadata, original_id: str) -> str:
        """Generate human-readable citation ID."""
        
        def sanitize(text: str) -> str:
            # Replace spaces and common separators with underscore
            text = text.replace(" ", "_").replace(":", "_").replace("-", "_")
            # Remove parentheses/brackets
            text = re.sub(r'[()\[\]]', '', text)
            return text

        if meta.source_type == 'quran':
             if meta.verse_ref:
                 # Clean verse ref (e.g. البقرة:43 -> quran_surah_43)
                 # Since we don't have a reliable arabic->english map for all surahs loaded yet,
                 # we will keep the arabic surah name but ensure it's "safe" (no spaces/colons)
                 # Ideally we'd map "البقرة" -> "baqarah".
                 # For now: quran_2_43 if we have numbers? No, we extract text "البقرة".
                 # Let's just normalize usage.
                 safe_ref = sanitize(meta.verse_ref)
                 return f"quran_{safe_ref}"
                 
        if meta.hadith_ref:
            # e.g. "البخاري:645" -> hadith_bukhari_645
            ref = meta.hadith_ref
            
            # Simple map for common collections
            mapping = {
                "البخاري": "bukhari", 
                "مسلم": "muslim", 
                "الترمذي": "tirmidhi", 
                "أبو داود": "abudawud", 
                "النسائي": "nasai", 
                "ابن ماجه": "ibnmajah", 
                "أحمد": "ahmad",
                "مالك": "malika",
                "موطأ": "muwatta" 
            }
            
            for ar, en in mapping.items():
                if ar in ref:
                    ref = ref.replace(ar, en)
                    break
            
            safe_ref = sanitize(ref)
            # Ensure it starts with hadith_ or just use it if it has collection
            if "hadith" not in safe_ref.lower():
                return f"hadith_{safe_ref}"
            return safe_ref
        
        # Fallback: clean up the original ID 
        parts = original_id.rsplit('_', 1)
        if len(parts) == 2 and len(parts[1]) == 6: # likely hash
            return parts[0]
            
        return original_id
    
    def _extract_quran_refs(self, text: str) -> list[str]:
        """Extract Quranic verse references from text."""
        refs = []
        for pattern in self.QURAN_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    surah = match[0]
                    ayah_start = match[1]
                    ayah_end = match[2] if len(match) > 2 and match[2] else None
                    
                    if ayah_end:
                        refs.append(f"{surah}:{ayah_start}-{ayah_end}")
                    else:
                        refs.append(f"{surah}:{ayah_start}")
                else:
                    refs.append(str(match))
        return refs
    
    def _extract_hadith_refs(self, text: str) -> list[str]:
        """Extract hadith references from text."""
        refs = []
        for pattern in self.HADITH_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    collection = match[0]
                    number = match[1]
                    refs.append(f"{collection}:{number}")
        return refs
    
    def _detect_madhab(self, text: str) -> Optional[str]:
        """Detect madhab from text content."""
        for madhab, pattern in self.MADHAB_PATTERNS.items():
            if pattern.search(text):
                return madhab
        return None
    
    def _detect_authenticity(self, text: str) -> Optional[str]:
        """Detect hadith authenticity grade."""
        for grade, pattern in self.AUTHENTICITY_PATTERNS.items():
            if pattern.search(text):
                return grade
        return None


def generate_doc_id(source_type: str, collection: str, index: int, content_hash: str = None) -> str:
    """
    Generate a persistent document ID.
    
    Format: {source_type}_{collection}_{index}_{hash}
    Example: hadith_bukhari_645_a1b2c3
    """
    import hashlib
    
    if content_hash is None and collection:
        # Generate hash from collection + index
        content_hash = hashlib.md5(f"{collection}_{index}".encode()).hexdigest()[:6]
    elif content_hash is None:
        content_hash = hashlib.md5(f"{source_type}_{index}".encode()).hexdigest()[:6]
    
    collection_clean = (collection or "unknown").replace(" ", "_").lower()[:20]
    return f"{source_type}_{collection_clean}_{index}_{content_hash}"
