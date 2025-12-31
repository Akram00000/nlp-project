"""
Reference Extractor - Extract hadith and Quranic references from text.

Uses pattern matching and NER to identify references to hadiths and ayahs.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HadithReference:
    """A hadith reference found in text."""
    
    text: str  # The reference text
    narrator: Optional[str] = None  # e.g., "أبو هريرة"
    source: Optional[str] = None  # e.g., "البخاري"
    number: Optional[str] = None  # Hadith number if available
    match_score: float = 1.0
    
    def to_query(self) -> str:
        """Convert to search query for hadith database."""
        parts = []
        if self.narrator:
            parts.append(self.narrator)
        if self.text:
            # Take first significant part
            words = self.text.split()[:10]
            parts.append(' '.join(words))
        return ' '.join(parts)


@dataclass
class AyahReference:
    """A Quranic verse reference."""
    
    surah: Optional[str] = None  # Surah name
    surah_number: Optional[int] = None
    ayah: Optional[int] = None
    text: Optional[str] = None  # Quoted text


class ReferenceExtractor:
    """
    Extract references to hadiths and Quranic verses.
    
    Patterns for hadith references:
    - "قال رسول الله صلى الله عليه وسلم: ..."
    - "عن أبي هريرة ... رواه البخاري"
    - "في صحيح مسلم عن ..."
    
    Patterns for ayah references:
    - {وَإِذَا قِيلَ لَهُمْ} [البقرة: 13]
    - قوله تعالى: {آية}
    """
    
    # Hadith indicators
    HADITH_PATTERNS = [
        r'قال رسول الله صلى الله عليه وسلم[:\s]+([^.]+)',
        r'قال النبي صلى الله عليه وسلم[:\s]+([^.]+)',
        r'عن ([^،]+)[،\s]+قال[:\s]+([^.]+)',
        r'رواه\s+(البخاري|مسلم|أبو داود|الترمذي|النسائي|ابن ماجه|أحمد)',
        r'في\s+(صحيح البخاري|صحيح مسلم|سنن أبي داود|جامع الترمذي)',
        r'أخرجه\s+(البخاري|مسلم|أبو داود|الترمذي|النسائي|ابن ماجه)',
    ]
    
    # Narrators (common sahaba)
    COMMON_NARRATORS = [
        'أبي هريرة', 'أبو هريرة',
        'عائشة', 'ابن عمر', 'ابن عباس',
        'أنس بن مالك', 'جابر بن عبد الله',
        'أبي سعيد الخدري', 'أبو سعيد الخدري',
    ]
    
    # Hadith sources
    HADITH_SOURCES = [
        'البخاري', 'مسلم', 'أبو داود', 'أبي داود',
        'الترمذي', 'النسائي', 'ابن ماجه', 'أحمد',
        'صحيح البخاري', 'صحيح مسلم',
    ]
    
    # Ayah pattern: {text} [surah: ayah]
    AYAH_PATTERN = r'{([^}]+)}\s*(?:\[([^:\]]+)(?::?\s*(\d+))?\])?'
    
    def extract_all(self, text: str) -> dict:
        """
        Extract all references from text.
        
        Returns:
            dict with 'hadiths' and 'ayahs' keys
        """
        return {
            'hadiths': self.extract_hadiths(text),
            'ayahs': self.extract_ayahs(text),
        }
    
    def extract_hadiths(self, text: str) -> list[dict]:
        """Extract hadith references."""
        references = []
        
        # Check each pattern
        for pattern in self.HADITH_PATTERNS:
            matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
            
            for match in matches:
                ref = self._parse_hadith_match(match, text)
                if ref:
                    references.append({
                        'text': ref.text,
                        'narrator': ref.narrator,
                        'source': ref.source,
                        'number': ref.number,
                        'query': ref.to_query(),
                    })
        
        # Deduplicate by query
        seen = set()
        unique_refs = []
        for ref in references:
            query = ref['query']
            if query not in seen:
                seen.add(query)
                unique_refs.append(ref)
        
        return unique_refs
    
    def extract_ayahs(self, text: str) -> list[dict]:
        """Extract Quranic verse references."""
        references = []
        
        matches = re.finditer(self.AYAH_PATTERN, text)
        
        for match in matches:
            ayah_text = match.group(1).strip()
            surah = match.group(2).strip() if match.group(2) else None
            ayah_num = int(match.group(3)) if match.group(3) else None
            
            ref = {
                'text': ayah_text,
                'surah': surah,
                'ayah': ayah_num,
            }
            references.append(ref)
        
        return references
    
    def _parse_hadith_match(self, match: re.Match, full_text: str) -> Optional[HadithReference]:
        """Parse a regex match into HadithReference."""
        
        # Get surrounding context
        start = max(0, match.start() - 100)
        end = min(len(full_text), match.end() + 200)
        context = full_text[start:end]
        
        # Extract main text
        hadith_text = match.group(1) if match.lastindex and match.lastindex >= 1 else match.group(0)
        hadith_text = hadith_text.strip()
        
        # Extract narrator
        narrator = None
        for nar in self.COMMON_NARRATORS:
            if nar in context:
                narrator = nar
                break
        
        # Extract source
        source = None
        for src in self.HADITH_SOURCES:
            if src in context:
                source = src
                break
        
        # Look for hadith number
        number = None
        num_pattern = r'رقم[:\s]+(\d+)'
        num_match = re.search(num_pattern, context)
        if num_match:
            number = num_match.group(1)
        
        return HadithReference(
            text=hadith_text[:200],  # Limit length
            narrator=narrator,
            source=source,
            number=number,
        )
    
    def has_hadith_reference(self, text: str) -> bool:
        """Quick check if text contains any hadith reference."""
        indicators = [
            'رسول الله صلى الله عليه وسلم',
            'النبي صلى الله عليه وسلم',
            'رواه', 'أخرجه',
            'صحيح البخاري', 'صحيح مسلم',
        ]
        return any(ind in text for ind in indicators)
    
    def has_ayah_reference(self, text: str) -> bool:
        """Quick check if text contains Quranic reference."""
        return '{' in text and '}' in text
