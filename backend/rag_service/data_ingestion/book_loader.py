"""
Book Loader - Load and process Arabic Fiqh books from OCR text files.

Handles:
- Loading OCR text files from madhab-organized directories
- Semantic chunking with page/chapter boundary awareness
- Optional OCR preprocessing (cleanup)
- Metadata extraction from filename and content
"""

import re
from pathlib import Path
from typing import Iterator, Optional
from dataclasses import dataclass, field
from loguru import logger

from ..utils.preprocessing import preprocess_ocr_text


@dataclass
class BookMetadata:
    """Metadata for a book chunk."""
    
    book_id: str
    title: str
    author: Optional[str]
    madhab: str  # Hanafi, Maliki, Shafii, Hanbali, or general
    volume: int
    page_start: int
    page_end: int
    chapter: Optional[str] = None
    section: Optional[str] = None
    source_file: str = ""
    type: str = "book"  # For filtering in retrieval
    
    def to_dict(self) -> dict:
        return {
            "book_id": self.book_id,
            "title": self.title,
            "author": self.author,
            "madhab": self.madhab,
            "volume": self.volume,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "chapter": self.chapter,
            "section": self.section,
            "source_file": self.source_file,
            "type": self.type,
        }


@dataclass
class BookDocument:
    """Processed book chunk ready for indexing."""
    
    id: str
    content: str
    metadata: dict
    embedding: Optional[list[float]] = None
    
    @property
    def book_metadata(self) -> BookMetadata:
        """Get structured metadata."""
        return BookMetadata(
            book_id=self.metadata.get("book_id", ""),
            title=self.metadata.get("title", ""),
            author=self.metadata.get("author"),
            madhab=self.metadata.get("madhab", ""),
            volume=self.metadata.get("volume", 1),
            page_start=self.metadata.get("page_start", 0),
            page_end=self.metadata.get("page_end", 0),
            chapter=self.metadata.get("chapter"),
            section=self.metadata.get("section"),
            source_file=self.metadata.get("source_file", ""),
            type=self.metadata.get("type", "book"),
        )


# Known book mappings (book_id -> book info)
KNOWN_BOOKS = {
    "33781": {
        "title": "مختصر اختلاف العلماء",
        "author": "الطحاوي",
        "full_author": "أبو جعفر أحمد بن محمد بن سلامة الطحاوي",
    },
    "48422": {
        "title": "تبيين الحقائق شرح كنز الدقائق",
        "author": "الزيلعي",
        "full_author": "فخر الدين الزيلعي الحنفي",
    },
    "77361": {
        "title": "البحر الرائق شرح كنز الدقائق",
        "author": "ابن نجيم",
        "full_author": "زين الدين ابن نجيم الحنفي",
    },
    "thskd": {
        "title": "تهذيب سنن أبي داود",
        "author": "ابن القيم",
        "full_author": "شمس الدين ابن القيم الجوزية",
    },
    "28163": {
        "title": "المبسوط",
        "author": "السرخسي",
        "full_author": "شمس الأئمة السرخسي",
    },
    "khrchi": {
        "title": "شرح مختصر خليل",
        "author": "الخرشي",
        "full_author": "أبو عبدالله الخرشي المالكي",
    },
}


class BookLoader:
    """
    Load Arabic Fiqh books from OCR text files.
    
    Features:
    - Madhab-aware directory scanning
    - Semantic chunking (respects pages, chapters, sections)
    - Optional OCR preprocessing
    - Metadata extraction from filenames and content
    - ~1000 token chunks with overlap
    
    Directory structure expected:
    OCR_folder/
        Hanafi/
            01_33781.txt  (volume_bookid.txt)
            02_33781.txt
        Maliki/
            khrchi-1.txt
        nawazel/
            subfolder/
    """
    
    # Arabic fiqh structural markers (for chunking)
    CHAPTER_PATTERNS = [
        r'^(كتاب\s+.+)$',           # كتاب الصلاة
        r'^(باب\s+.+)$',            # باب ما جاء في...
        r'^(فصل\s+.*)$',            # فصل في...
        r'^(مسألة\s*[:：]?\s*.*)$',  # مسألة: ...
        r'^(فرع\s*[:：]?\s*.*)$',    # فرع: ...
    ]
    
    # Madhab folder names
    MADHAB_FOLDERS = {
        'hanafi': 'الحنفي',
        'maliki': 'المالكي',
        'shafii': 'الشافعي',
        'hanbali': 'الحنبلي',
        'nawazel': 'النوازل',
    }
    
    def __init__(
        self,
        target_chunk_tokens: int = 1000,
        chunk_overlap_tokens: int = 100,
        preprocess: bool = False,
        extract_chapters: bool = True,
    ):
        """
        Initialize book loader.
        
        Args:
            target_chunk_tokens: Target tokens per chunk (~4 chars/token for Arabic)
            chunk_overlap_tokens: Overlap between chunks
            preprocess: Apply OCR preprocessing (cleanup artifacts)
            extract_chapters: Try to extract chapter/section info
        """
        self.target_chunk_tokens = target_chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.preprocess = preprocess
        self.extract_chapters = extract_chapters
        
        # Approximate chars per token for Arabic
        self.chars_per_token = 4
        self.target_chunk_chars = target_chunk_tokens * self.chars_per_token
        self.overlap_chars = chunk_overlap_tokens * self.chars_per_token
    
    def load_directory(
        self,
        directory: Path | str,
        madhab_filter: Optional[str] = None,
    ) -> Iterator[BookDocument]:
        """
        Load all books from a directory tree.
        
        Args:
            directory: Root directory containing madhab folders
            madhab_filter: Only load from specific madhab folder
            
        Yields:
            BookDocument objects
        """
        directory = Path(directory)
        
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return
        
        # Scan madhab folders
        for madhab_folder in directory.iterdir():
            if not madhab_folder.is_dir():
                continue
            
            madhab_name = madhab_folder.name.lower()
            
            # Apply filter
            if madhab_filter and madhab_name != madhab_filter.lower():
                continue
            
            logger.info(f"Loading books from {madhab_name} folder")
            yield from self._load_madhab_folder(madhab_folder, madhab_name)
    
    def _load_madhab_folder(
        self,
        folder: Path,
        madhab: str,
    ) -> Iterator[BookDocument]:
        """Load all books from a madhab folder."""
        
        # Handle nested structure (subfolders with txt files)
        txt_files = list(folder.glob("*.txt"))
        
        if txt_files:
            for txt_file in sorted(txt_files):
                yield from self.load_book(txt_file, madhab)
        
        # Check subfolders
        for subfolder in folder.iterdir():
            if subfolder.is_dir():
                for txt_file in sorted(subfolder.glob("*.txt")):
                    yield from self.load_book(txt_file, madhab)
    
    def load_book(
        self,
        file_path: Path | str,
        madhab: Optional[str] = None,
    ) -> Iterator[BookDocument]:
        """
        Load and chunk a single book file.
        
        Args:
            file_path: Path to OCR text file
            madhab: Madhab name (inferred from path if not provided)
            
        Yields:
            BookDocument chunks
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return
        
        logger.info(f"Loading book: {file_path.name}")
        
        # Infer madhab from path if not provided
        if madhab is None:
            madhab = self._infer_madhab(file_path)
        
        # Parse filename for metadata
        volume, book_id = self._parse_filename(file_path.name)
        
        # Read content
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return
        
        # Apply preprocessing if enabled
        if self.preprocess:
            content = preprocess_ocr_text(content)
        
        # Parse pages
        pages = self._parse_pages(content)
        
        if not pages:
            logger.warning(f"No pages found in {file_path.name}")
            return
        
        # Extract book info from first pages
        book_info = self._extract_book_info(pages[:5], book_id)
        
        # Chunk the book
        chunks = self._chunk_book(pages, book_info, volume, madhab, file_path.name)
        
        for chunk in chunks:
            yield chunk
        
        logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
    
    def _parse_filename(self, filename: str) -> tuple[int, str]:
        """Parse volume and book_id from filename."""
        name = Path(filename).stem
        
        # Pattern: 01_33781 (volume_bookid)
        match = re.match(r'^(\d+)_(.+)$', name)
        if match:
            return int(match.group(1)), match.group(2)
        
        # Pattern: khrchi-1 (bookid-volume)
        match = re.match(r'^(.+)-(\d+)$', name)
        if match:
            return int(match.group(2)), match.group(1)
        
        # Pattern: Nawazel-01
        match = re.match(r'^(.+)-(\d+)$', name, re.IGNORECASE)
        if match:
            return int(match.group(2)), match.group(1).lower()
        
        # Default: volume 1, filename as id
        return 1, name
    
    def _infer_madhab(self, file_path: Path) -> str:
        """Infer madhab from file path."""
        path_str = str(file_path).lower()
        
        for madhab in self.MADHAB_FOLDERS.keys():
            if madhab in path_str:
                return madhab
        
        return "general"
    
    def _parse_pages(self, content: str) -> list[dict]:
        """Parse content into pages using --- PAGE n --- markers."""
        pages = []
        
        # Split by page markers
        pattern = r'---\s*PAGE\s+(\d+)\s*---'
        parts = re.split(pattern, content)
        
        # parts will be: [preamble, page_num, content, page_num, content, ...]
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                page_num = int(parts[i])
                page_content = parts[i + 1].strip()
                
                if page_content:
                    pages.append({
                        'number': page_num,
                        'content': page_content,
                    })
        
        return pages
    
    def _extract_book_info(
        self,
        first_pages: list[dict],
        book_id: str,
    ) -> dict:
        """Extract book title and author from first pages."""
        
        # Check known books first
        if book_id in KNOWN_BOOKS:
            return KNOWN_BOOKS[book_id].copy()
        
        info = {
            'title': '',
            'author': None,
            'full_author': None,
        }
        
        # Try to extract from content
        combined = '\n'.join(p['content'] for p in first_pages[:3])
        
        # Look for common patterns
        # Title often on first page in larger text
        title_patterns = [
            r'^(.{10,60})\s*$',  # Standalone line
            r'كتاب\s+(.{5,50})',  # كتاب ...
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, combined, re.MULTILINE)
            if match:
                potential_title = match.group(1).strip()
                if len(potential_title) > 5 and not potential_title.startswith('---'):
                    info['title'] = potential_title
                    break
        
        # Look for author patterns
        author_patterns = [
            r'تصنيف\s+(.+?)(?:\n|$)',
            r'تأليف\s+(.+?)(?:\n|$)',
            r'للشيخ\s+(.+?)(?:\n|$)',
            r'للإمام\s+(.+?)(?:\n|$)',
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, combined)
            if match:
                info['author'] = match.group(1).strip()[:100]
                break
        
        if not info['title']:
            info['title'] = f"كتاب {book_id}"
        
        return info
    
    def _chunk_book(
        self,
        pages: list[dict],
        book_info: dict,
        volume: int,
        madhab: str,
        source_file: str,
    ) -> list[BookDocument]:
        """Chunk book into semantic units."""
        chunks = []
        
        current_chunk = []
        current_chars = 0
        current_chapter = None
        current_section = None
        chunk_page_start = pages[0]['number'] if pages else 1
        
        for page in pages:
            page_content = page['content']
            page_num = page['number']
            
            # Detect chapter/section markers
            if self.extract_chapters:
                detected = self._detect_structure(page_content)
                if detected['chapter']:
                    current_chapter = detected['chapter']
                if detected['section']:
                    current_section = detected['section']
            
            # Split page into paragraphs
            paragraphs = self._split_into_paragraphs(page_content)
            
            for para in paragraphs:
                para_len = len(para)
                
                # Check if adding this paragraph exceeds limit
                if current_chars + para_len > self.target_chunk_chars and current_chunk:
                    # Save current chunk
                    chunk_text = '\n\n'.join(current_chunk)
                    chunk_doc = self._create_chunk_document(
                        content=chunk_text,
                        book_info=book_info,
                        volume=volume,
                        madhab=madhab,
                        page_start=chunk_page_start,
                        page_end=page_num,
                        chapter=current_chapter,
                        section=current_section,
                        source_file=source_file,
                        chunk_index=len(chunks),
                    )
                    chunks.append(chunk_doc)
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = [overlap_text] if overlap_text else []
                    current_chars = len(overlap_text) if overlap_text else 0
                    chunk_page_start = page_num
                
                current_chunk.append(para)
                current_chars += para_len
        
        # Don't forget last chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunk_doc = self._create_chunk_document(
                content=chunk_text,
                book_info=book_info,
                volume=volume,
                madhab=madhab,
                page_start=chunk_page_start,
                page_end=pages[-1]['number'] if pages else chunk_page_start,
                chapter=current_chapter,
                section=current_section,
                source_file=source_file,
                chunk_index=len(chunks),
            )
            chunks.append(chunk_doc)
        
        return chunks
    
    def _detect_structure(self, text: str) -> dict:
        """Detect chapter and section markers in text."""
        result = {'chapter': None, 'section': None}
        
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            
            # Check chapter patterns
            for pattern in self.CHAPTER_PATTERNS[:2]:  # كتاب, باب
                match = re.match(pattern, line)
                if match:
                    result['chapter'] = match.group(1)[:100]
                    break
            
            # Check section patterns
            for pattern in self.CHAPTER_PATTERNS[2:]:  # فصل, مسألة, فرع
                match = re.match(pattern, line)
                if match:
                    result['section'] = match.group(1)[:100]
                    break
        
        return result
    
    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs."""
        # Split by double newlines or structural markers
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _get_overlap_text(self, chunks: list[str]) -> str:
        """Get overlap text from end of chunks."""
        if not chunks:
            return ""
        
        # Get last chunk and take overlap_chars from end
        last_text = chunks[-1]
        if len(last_text) <= self.overlap_chars:
            return last_text
        
        # Try to break at sentence boundary
        overlap = last_text[-self.overlap_chars:]
        sentence_end = overlap.rfind('.')
        if sentence_end > len(overlap) // 2:
            return overlap[sentence_end + 1:].strip()
        
        return overlap
    
    def _create_chunk_document(
        self,
        content: str,
        book_info: dict,
        volume: int,
        madhab: str,
        page_start: int,
        page_end: int,
        chapter: Optional[str],
        section: Optional[str],
        source_file: str,
        chunk_index: int,
    ) -> BookDocument:
        """Create a BookDocument from chunk data."""
        
        book_id = book_info.get('book_id', source_file.split('_')[-1].replace('.txt', ''))
        
        # Generate unique ID
        doc_id = f"book_{madhab}_{book_id}_v{volume}_p{page_start}-{page_end}_c{chunk_index}"
        
        metadata = BookMetadata(
            book_id=book_id,
            title=book_info.get('title', ''),
            author=book_info.get('author') or book_info.get('full_author'),
            madhab=madhab,
            volume=volume,
            page_start=page_start,
            page_end=page_end,
            chapter=chapter,
            section=section,
            source_file=source_file,
            type="book",
        )
        
        return BookDocument(
            id=doc_id,
            content=content,
            metadata=metadata.to_dict(),
        )
    
    def get_statistics(self, documents: list[BookDocument]) -> dict:
        """Get statistics about loaded books."""
        if not documents:
            return {
                'total_chunks': 0,
                'unique_books': 0,
                'madhabs': [],
                'avg_chunk_size': 0,
            }
        
        books = set()
        madhabs = set()
        total_chars = 0
        
        for doc in documents:
            books.add(doc.metadata.get('book_id', ''))
            madhabs.add(doc.metadata.get('madhab', ''))
            total_chars += len(doc.content)
        
        return {
            'total_chunks': len(documents),
            'unique_books': len(books),
            'madhabs': list(madhabs),
            'avg_chunk_size': total_chars // len(documents) if documents else 0,
            'avg_tokens_estimate': (total_chars // len(documents)) // self.chars_per_token if documents else 0,
        }
