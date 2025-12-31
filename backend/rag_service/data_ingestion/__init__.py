"""
Data Ingestion Module - Load fatwas, hadiths, and books into vector database.

Handles:
- Loading JSON/JSONL fatwa files
- Extracting and indexing hadiths
- Loading and chunking OCR book files
- Creating embeddings
- Uploading to Qdrant
"""

from .fatwa_loader import FatwaLoader, FatwaDocument
from .hadith_loader import HadithLoader, HadithDocument
from .book_loader import BookLoader, BookDocument, BookMetadata
from .indexer import VectorIndexer
from .reference_extractor import ReferenceExtractor

__all__ = [
    "FatwaLoader",
    "FatwaDocument",
    "HadithLoader",
    "HadithDocument",
    "BookLoader",
    "BookDocument",
    "BookMetadata",
    "VectorIndexer",
    "ReferenceExtractor",
]
