"""
Metadata Handler - Parse and enrich fatwa metadata for citations.

Handles extraction of citation-relevant metadata from various fatwa formats.
"""

import json
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class FatwaMetadata:
    """Structured fatwa metadata for citations."""
    
    fatwa_id: Optional[str] = None
    title: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    scholar: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    date: Optional[str] = None
    category: Optional[str] = None
    language: str = "ar"
    
    # Additional metadata
    extra: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "fatwa_id": self.fatwa_id,
            "title": self.title,
            "question": self.question,
            "answer": self.answer,
            "scholar": self.scholar,
            "source": self.source,
            "url": self.url,
            "date": self.date,
            "category": self.category,
            "language": self.language,
            **self.extra
        }


class MetadataHandler:
    """
    Handle fatwa metadata extraction and enrichment.
    
    Supports:
    - Various JSON/JSONL formats
    - Metadata normalization
    - Missing field handling
    - Scholar name mapping
    """
    
    # Known scholar name mappings (Arabic -> standardized)
    SCHOLAR_MAPPINGS = {
        "ابن باز": "الشيخ عبدالعزيز بن باز",
        "بن باز": "الشيخ عبدالعزيز بن باز",
        "binbaz": "الشيخ عبدالعزيز بن باز",
        "ابن عثيمين": "الشيخ محمد بن صالح العثيمين",
        "العثيمين": "الشيخ محمد بن صالح العثيمين",
        "الفوزان": "الشيخ صالح الفوزان",
        "اللجنة الدائمة": "اللجنة الدائمة للإفتاء",
    }
    
    # Known source mappings
    SOURCE_MAPPINGS = {
        "binbaz.org": "موقع الشيخ ابن باز الرسمي",
        "islamqa": "الإسلام سؤال وجواب",
        "islamweb": "إسلام ويب",
    }
    
    # Field name variations to check
    FIELD_VARIATIONS = {
        "fatwa_id": ["fatwa_id", "id", "fatwaId", "reference_id", "ref"],
        "title": ["title", "عنوان", "subject", "topic"],
        "question": ["question", "سؤال", "query", "q"],
        "answer": ["answer", "جواب", "response", "a", "content", "text"],
        "scholar": ["scholar", "mufti", "عالم", "مفتي", "sheikh", "الشيخ"],
        "source": ["source", "مصدر", "website", "origin", "reference"],
        "url": ["url", "link", "رابط", "href"],
        "date": ["date", "تاريخ", "published_date", "publish_date", "created_at"],
        "category": ["category", "تصنيف", "topic", "section", "باب"],
    }
    
    def __init__(self):
        self._cache: dict[str, FatwaMetadata] = {}
    
    def extract_metadata(self, raw_data: dict) -> FatwaMetadata:
        """
        Extract structured metadata from raw fatwa data.
        
        Args:
            raw_data: Raw fatwa data (e.g., from JSONL)
            
        Returns:
            Structured FatwaMetadata object
        """
        metadata = FatwaMetadata()
        
        # Extract each field using variations
        for field_name, variations in self.FIELD_VARIATIONS.items():
            value = self._extract_field(raw_data, variations)
            if value is not None:
                setattr(metadata, field_name, value)
        
        # Normalize scholar name
        if metadata.scholar:
            metadata.scholar = self._normalize_scholar(metadata.scholar)
        
        # Normalize source
        if metadata.source:
            metadata.source = self._normalize_source(metadata.source)
        
        # Extract URL from source if not present
        if not metadata.url and metadata.source:
            metadata.url = self._extract_url_from_source(metadata.source)
        
        # Store any extra fields
        known_fields = set()
        for variations in self.FIELD_VARIATIONS.values():
            known_fields.update(variations)
        
        for key, value in raw_data.items():
            if key.lower() not in known_fields and value is not None:
                metadata.extra[key] = value
        
        # Detect language
        text_to_check = metadata.answer or metadata.question or ""
        metadata.language = self._detect_language(text_to_check)
        
        return metadata
    
    def _extract_field(self, data: dict, variations: list[str]) -> Optional[str]:
        """Extract field value checking multiple variations."""
        for var in variations:
            # Check exact match
            if var in data and data[var]:
                return str(data[var])
            # Check case-insensitive
            for key in data:
                if key.lower() == var.lower() and data[key]:
                    return str(data[key])
        return None
    
    def _normalize_scholar(self, scholar: str) -> str:
        """Normalize scholar name to standard form."""
        scholar_lower = scholar.lower().strip()
        
        for key, standard in self.SCHOLAR_MAPPINGS.items():
            if key in scholar_lower or scholar_lower in key:
                return standard
        
        return scholar
    
    def _normalize_source(self, source: str) -> str:
        """Normalize source name."""
        source_lower = source.lower().strip()
        
        for key, standard in self.SOURCE_MAPPINGS.items():
            if key in source_lower:
                return standard
        
        return source
    
    def _extract_url_from_source(self, source: str) -> Optional[str]:
        """Try to extract URL from source field."""
        import re
        
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        match = re.search(url_pattern, source)
        
        if match:
            return match.group(0)
        return None
    
    def _detect_language(self, text: str) -> str:
        """Detect language of text (ar/en)."""
        if not text:
            return "ar"
        
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        return "ar" if arabic_chars > len(text) * 0.3 else "en"
    
    def load_from_jsonl(self, file_path: Path | str) -> list[FatwaMetadata]:
        """
        Load and extract metadata from a JSONL file.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            List of FatwaMetadata objects
        """
        file_path = Path(file_path)
        metadata_list = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        metadata = self.extract_metadata(data)
                        metadata_list.append(metadata)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
                        continue
            
            logger.info(f"Loaded {len(metadata_list)} fatwas from {file_path}")
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
        
        return metadata_list
    
    def enrich_from_payload(self, payload: dict) -> FatwaMetadata:
        """
        Enrich metadata from a Qdrant payload.
        
        Args:
            payload: Document payload from Qdrant
            
        Returns:
            Enriched FatwaMetadata
        """
        return self.extract_metadata(payload)
    
    def validate_metadata(self, metadata: FatwaMetadata) -> dict:
        """
        Validate metadata completeness.
        
        Returns dict with validation results.
        """
        required_fields = ["fatwa_id", "scholar", "source"]
        recommended_fields = ["title", "date", "url", "category"]
        
        missing_required = []
        missing_recommended = []
        
        for field in required_fields:
            if not getattr(metadata, field):
                missing_required.append(field)
        
        for field in recommended_fields:
            if not getattr(metadata, field):
                missing_recommended.append(field)
        
        return {
            "valid": len(missing_required) == 0,
            "missing_required": missing_required,
            "missing_recommended": missing_recommended,
            "completeness": 1 - (len(missing_required) + len(missing_recommended) * 0.5) / 
                           (len(required_fields) + len(recommended_fields))
        }
