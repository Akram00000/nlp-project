"""
Fatwa Loader - Load and parse fatwa JSON/JSONL files.

Supports multiple fatwa formats from different sources.
"""

import json
from pathlib import Path
from typing import Iterator, Optional
from dataclasses import dataclass
from loguru import logger

from ..citation.metadata_handler import MetadataHandler, FatwaMetadata


@dataclass
class FatwaDocument:
    """Processed fatwa document ready for indexing."""
    
    id: str
    content: str  # Combined question + answer or just answer
    metadata: dict
    embedding: Optional[list[float]] = None
    
    @property
    def fatwa_metadata(self) -> FatwaMetadata:
        """Get structured metadata."""
        handler = MetadataHandler()
        return handler.extract_metadata(self.metadata)


class FatwaLoader:
    """
    Load fatwas from JSON/JSONL files.
    
    Features:
    - Support for multiple file formats
    - Automatic metadata extraction
    - Chunking for long fatwas
    - Reference extraction (hadiths, ayahs)
    """
    
    def __init__(
        self,
        combine_question_answer: bool = True,
        max_chunk_size: int = 1000,
        extract_references: bool = True,
    ):
        """
        Initialize loader.
        
        Args:
            combine_question_answer: Combine Q&A into single content field
            max_chunk_size: Max characters per chunk (0 = no chunking)
            extract_references: Extract hadith/ayah references
        """
        self.combine_qa = combine_question_answer
        self.max_chunk_size = max_chunk_size
        self.extract_references = extract_references
        self.metadata_handler = MetadataHandler()
    
    def load_jsonl(self, file_path: Path | str) -> Iterator[FatwaDocument]:
        """
        Load fatwas from JSONL file.
        
        Args:
            file_path: Path to JSONL file
            
        Yields:
            FatwaDocument objects
        """
        file_path = Path(file_path)
        logger.info(f"Loading fatwas from {file_path}")
        
        count = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        doc = self._process_fatwa(data, f"{file_path.stem}_{line_num}")
                        
                        if doc:
                            yield doc
                            count += 1
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num}: {e}")
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {e}")
            
            logger.info(f"Loaded {count} fatwas from {file_path}")
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    def load_json(self, file_path: Path | str) -> Iterator[FatwaDocument]:
        """
        Load fatwas from JSON file (array format).
        
        Args:
            file_path: Path to JSON file
            
        Yields:
            FatwaDocument objects
        """
        file_path = Path(file_path)
        logger.info(f"Loading fatwas from {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                data = [data]
            
            count = 0
            for idx, item in enumerate(data):
                doc = self._process_fatwa(item, f"{file_path.stem}_{idx}")
                if doc:
                    yield doc
                    count += 1
            
            logger.info(f"Loaded {count} fatwas from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    def load_directory(
        self,
        directory: Path | str,
        pattern: str = "*.jsonl"
    ) -> Iterator[FatwaDocument]:
        """
        Load all fatwa files from a directory.
        
        Args:
            directory: Directory path
            pattern: File pattern (*.jsonl, *.json, etc.)
            
        Yields:
            FatwaDocument objects
        """
        directory = Path(directory)
        
        for file_path in directory.glob(pattern):
            if file_path.suffix == '.jsonl':
                yield from self.load_jsonl(file_path)
            elif file_path.suffix == '.json':
                yield from self.load_json(file_path)
    
    def _process_fatwa(self, data: dict, fallback_id: str) -> Optional[FatwaDocument]:
        """Process raw fatwa data into FatwaDocument."""
        
        # Extract metadata
        metadata = self.metadata_handler.extract_metadata(data)
        
        # Build content
        content = self._build_content(data, metadata)
        if not content:
            logger.warning(f"Empty content for fatwa {fallback_id}")
            return None
        
        # Generate ID
        doc_id = metadata.fatwa_id or data.get('_id') or fallback_id
        
        # Extract references if enabled
        if self.extract_references:
            refs = self._extract_references(content)
            if refs:
                metadata.extra['references'] = refs
        
        # Create document
        doc = FatwaDocument(
            id=doc_id,
            content=content,
            metadata=metadata.to_dict(),
        )
        
        # Handle chunking if needed
        if self.max_chunk_size > 0 and len(content) > self.max_chunk_size:
            # Return first chunk for now; full chunking would yield multiple docs
            # TODO: Implement proper multi-chunk support
            doc.content = content[:self.max_chunk_size]
            logger.debug(f"Truncated fatwa {doc_id} to {self.max_chunk_size} chars")
        
        return doc
    
    def _build_content(self, data: dict, metadata: FatwaMetadata) -> str:
        """Build content field from question and answer."""
        
        if self.combine_qa:
            parts = []
            
            # Add question
            if metadata.question:
                parts.append(f"السؤال: {metadata.question}")
            
            # Add answer
            if metadata.answer:
                parts.append(f"الجواب: {metadata.answer}")
            
            return "\n\n".join(parts)
        else:
            # Just use answer
            return metadata.answer or data.get('content', data.get('text', ''))
    
    def _extract_references(self, text: str) -> dict:
        """
        Extract hadith and ayah references from text.
        
        Returns dict with 'hadiths' and 'ayahs' lists.
        """
        from .reference_extractor import ReferenceExtractor
        
        extractor = ReferenceExtractor()
        return extractor.extract_all(text)
    
    def get_statistics(self, docs: list[FatwaDocument]) -> dict:
        """Get statistics about loaded fatwas."""
        if not docs:
            return {}
        
        total = len(docs)
        scholars = {}
        sources = {}
        with_refs = 0
        
        for doc in docs:
            meta = doc.metadata
            
            scholar = meta.get('scholar')
            if scholar:
                scholars[scholar] = scholars.get(scholar, 0) + 1
            
            source = meta.get('source')
            if source:
                sources[source] = sources.get(source, 0) + 1
            
            if meta.get('references'):
                with_refs += 1
        
        return {
            'total_fatwas': total,
            'unique_scholars': len(scholars),
            'top_scholars': sorted(scholars.items(), key=lambda x: x[1], reverse=True)[:5],
            'unique_sources': len(sources),
            'top_sources': sorted(sources.items(), key=lambda x: x[1], reverse=True)[:5],
            'with_references': with_refs,
            'reference_percentage': (with_refs / total * 100) if total > 0 else 0,
        }
