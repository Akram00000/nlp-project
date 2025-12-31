"""
Hadith Loader - Load and index hadith collections.

Supports hadith JSON files for creating a searchable hadith database.
"""

import json
from pathlib import Path
from typing import Iterator, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class HadithDocument:
    """Processed hadith document."""
    
    id: str
    content: str  # Hadith text (Arabic)
    metadata: dict
    embedding: Optional[list[float]] = None
    
    @property
    def narrator(self) -> Optional[str]:
        return self.metadata.get('narrator')
    
    @property
    def source(self) -> Optional[str]:
        return self.metadata.get('source')
    
    @property
    def book(self) -> Optional[str]:
        return self.metadata.get('book')
    
    @property
    def number(self) -> Optional[str]:
        return self.metadata.get('number')


class HadithLoader:
    """
    Load hadiths from JSON files.
    
    Supports multiple formats including the GitHub hadith-json repository format.
    
    Expected format (GitHub hadith-json):
    {
        "id": 7187,
        "idInBook": 7187,
        "chapterId": 97,
        "bookId": 1,
        "arabic": "حديث بالعربية",
        "english": {
            "narrator": "Narrated Abu Huraira:",
            "text": "..."
        }
    }
    """
    
    # Mapping of book IDs to book names (based on GitHub hadith-json structure)
    BOOK_NAMES = {
        1: "Sahih Bukhari",
        2: "Sahih Muslim",
        3: "Sunan Abu Dawud",
        4: "Sunan al-Tirmidhi",
        5: "Sunan al-Nasa'i",
        6: "Sunan Ibn Majah",
        7: "Musnad Ahmad",
        8: "Muwatta Malik",
        9: "Sunan al-Darimi"
    }
    
    def __init__(self):
        self.field_mappings = {
            'content': ['arabic', 'text', 'hadith', 'content'],
            'narrator': ['narrator', 'rawi', 'راوي'],
            'source': ['source', 'collection', 'مصدر'],
            'book': ['book', 'kitab', 'كتاب'],
            'number': ['number', 'hadith_number', 'idInBook', 'رقم'],
            'chapter': ['chapter', 'bab', 'chapterId', 'باب'],
            'grade': ['grade', 'grading', 'درجة'],
        }
    
    def load_jsonl(self, file_path: Path | str) -> Iterator[HadithDocument]:
        """Load hadiths from JSONL file."""
        file_path = Path(file_path)
        logger.info(f"Loading hadiths from {file_path}")
        
        count = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        doc = self._process_hadith(data, f"hadith_{file_path.stem}_{line_num}")
                        
                        if doc:
                            yield doc
                            count += 1
                            
                    except Exception as e:
                        logger.warning(f"Error processing hadith line {line_num}: {e}")
            
            logger.info(f"Loaded {count} hadiths from {file_path}")
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    def load_json(self, file_path: Path | str) -> Iterator[HadithDocument]:
        """Load hadiths from JSON file (array format or nested format)."""
        file_path = Path(file_path)
        
        try:
            logger.info(f"Loading hadiths from {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle nested format (GitHub hadith-json structure)
            if isinstance(data, dict) and 'hadiths' in data:
                logger.info(f"Detected nested format with 'hadiths' key")
                data = data['hadiths']
            
            # Ensure we have a list
            if not isinstance(data, list):
                data = [data]
            
            count = 0
            for idx, item in enumerate(data):
                doc = self._process_hadith(item, f"hadith_{file_path.stem}_{idx}")
                if doc:
                    yield doc
                    count += 1
            
            logger.info(f"Loaded {count} hadiths from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    def load_directory(self, dir_path: Path | str, pattern: str = "*.json") -> Iterator[HadithDocument]:
        """Load all hadith JSON files from a directory.
        
        Args:
            dir_path: Directory containing JSON files
            pattern: Glob pattern for JSON files (default: "*.json")
        """
        dir_path = Path(dir_path)
        
        if not dir_path.is_dir():
            logger.error(f"Not a directory: {dir_path}")
            return
        
        # Find all JSON files matching the pattern
        json_files = list(dir_path.glob(pattern))
        logger.info(f"Found {len(json_files)} files matching '{pattern}' in {dir_path}")
        
        for json_file in sorted(json_files):
            logger.info(f"Processing {json_file.name}...")
            yield from self.load_json(json_file)
    
    def _process_hadith(self, data: dict, fallback_id: str) -> Optional[HadithDocument]:
        """Process raw hadith data into HadithDocument."""
        
        # Extract content (try direct arabic field first, then use mappings)
        content = data.get('arabic') or self._extract_field(data, 'content')
        if not content:
            logger.debug(f"No content found for hadith. Keys: {list(data.keys())}")
            return None
        
        # Extract narrator - check nested english.narrator field first
        narrator = None
        if 'english' in data and isinstance(data['english'], dict):
            narrator = data['english'].get('narrator')
        if not narrator:
            narrator = self._extract_field(data, 'narrator')
        
        # Extract source - try to map bookId to book name, or use direct source
        source = None
        if 'bookId' in data and data['bookId'] in self.BOOK_NAMES:
            source = self.BOOK_NAMES[data['bookId']]
        if not source:
            source = self._extract_field(data, 'source')
        
        # Build metadata
        metadata = {
            'narrator': narrator,
            'source': source,
            'book': self._extract_field(data, 'book'),
            'chapter': str(data.get('chapterId')) if 'chapterId' in data else self._extract_field(data, 'chapter'),
            'number': str(data.get('idInBook')) if 'idInBook' in data else self._extract_field(data, 'number'),
            'grade': self._extract_field(data, 'grade'),
            'type': 'hadith',  # Mark as hadith for filtering
        }
        
        # Add English translation if available
        if 'english' in data and isinstance(data['english'], dict):
            english_text = data['english'].get('text')
            if english_text:
                metadata['english_text'] = english_text
        
        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        # Generate ID - prefer numeric ID, fallback to string ID or fallback
        doc_id = str(data.get('id', data.get('_id', fallback_id)))
        
        return HadithDocument(
            id=doc_id,
            content=content,
            metadata=metadata,
        )
    
    def _extract_field(self, data: dict, field_type: str) -> Optional[str]:
        """Extract field using multiple possible key names."""
        variations = self.field_mappings.get(field_type, [])
        
        for var in variations:
            if var in data and data[var]:
                return str(data[var])
        
        return None
    
    def get_statistics(self, docs: list[HadithDocument]) -> dict:
        """Get statistics about loaded hadiths."""
        if not docs:
            return {}
        
        total = len(docs)
        sources = {}
        narrators = {}
        
        for doc in docs:
            if doc.source:
                sources[doc.source] = sources.get(doc.source, 0) + 1
            if doc.narrator:
                narrators[doc.narrator] = narrators.get(doc.narrator, 0) + 1
        
        return {
            'total_hadiths': total,
            'unique_sources': len(sources),
            'top_sources': sorted(sources.items(), key=lambda x: x[1], reverse=True)[:5],
            'unique_narrators': len(narrators),
            'top_narrators': sorted(narrators.items(), key=lambda x: x[1], reverse=True)[:10],
        }
