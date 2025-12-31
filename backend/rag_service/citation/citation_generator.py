"""
Citation Generator - Generate formatted citations for fatwa and book sources.

Creates properly formatted citations for Islamic scholarly works,
supporting multiple citation styles and formats for fatwas and books.
"""

from typing import Optional
from dataclasses import dataclass
from loguru import logger

from .metadata_handler import MetadataHandler, FatwaMetadata


@dataclass
class Citation:
    """A formatted citation."""
    
    index: int  # Citation number [1], [2], etc.
    document_id: str
    formatted: str  # Full formatted citation
    short: str  # Short reference
    scholar: Optional[str]
    source: Optional[str]
    url: Optional[str]
    relevance_score: float
    source_type: str = "fatwa"  # fatwa, hadith, or book
    
    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "document_id": self.document_id,
            "formatted": self.formatted,
            "short": self.short,
            "scholar": self.scholar,
            "source": self.source,
            "url": self.url,
            "relevance_score": self.relevance_score,
            "source_type": self.source_type,
        }


class CitationGenerator:
    """
    Generate citations for fatwa sources.
    
    Supports multiple citation styles:
    - islamic_scholarly: Traditional Islamic citation format
    - simple: Basic source attribution
    - detailed: Full metadata citation
    
    Features:
    - Automatic metadata extraction
    - Multiple format styles
    - Missing data handling
    - URL linking
    """
    
    def __init__(
        self,
        format_style: str = "islamic_scholarly",
        include_urls: bool = True,
        include_scores: bool = False,
    ):
        """
        Initialize citation generator.
        
        Args:
            format_style: Citation format (islamic_scholarly, simple, detailed)
            include_urls: Whether to include URLs in citations
            include_scores: Whether to include relevance scores
        """
        self.format_style = format_style
        self.include_urls = include_urls
        self.include_scores = include_scores
        self.metadata_handler = MetadataHandler()
    
    def generate_citations(
        self,
        documents: list["RetrievalResult"],  # Forward reference
        format_style: Optional[str] = None,
    ) -> list[dict]:
        """
        Generate citations for retrieved documents.
        
        Args:
            documents: List of RetrievalResult objects
            format_style: Override default format style
            
        Returns:
            List of citation dictionaries
        """
        style = format_style or self.format_style
        citations = []
        
        for i, doc in enumerate(documents, 1):
            # Determine source type
            source_type = doc.metadata.get("type", "fatwa")
            
            # Generate formatted citation based on source type
            if source_type == "book":
                formatted = self._format_book_citation(doc.metadata, i)
                short = self._format_book_short(doc.metadata, i)
                scholar = doc.metadata.get("author")
                source = doc.metadata.get("title")
            elif source_type == "hadith":
                formatted = self._format_hadith_citation(doc.metadata, i)
                short = self._format_hadith_short(doc.metadata, i)
                scholar = doc.metadata.get("narrator")
                source = doc.metadata.get("source")
            else:
                # Fatwa (default)
                metadata = self.metadata_handler.enrich_from_payload(doc.metadata)
                
                if style == "islamic_scholarly":
                    formatted = self._format_islamic_scholarly(metadata, i)
                elif style == "simple":
                    formatted = self._format_simple(metadata, i)
                elif style == "detailed":
                    formatted = self._format_detailed(metadata, i)
                else:
                    formatted = self._format_simple(metadata, i)
                
                short = self._format_short(metadata, i)
                scholar = metadata.scholar
                source = metadata.source
            
            citation = Citation(
                index=i,
                document_id=doc.id,
                formatted=formatted,
                short=short,
                scholar=scholar,
                source=source,
                url=doc.metadata.get("url"),
                relevance_score=doc.score,
                source_type=source_type,
            )
            
            citations.append(citation.to_dict())
        
        return citations
    
    def _format_book_citation(self, metadata: dict, index: int) -> str:
        """
        Format citation for a book source.
        
        Format: [Index] [المؤلف], "[الكتاب]", المذهب [المجلد/الجزء]، ص[الصفحة]
        """
        parts = []
        
        # Author
        author = metadata.get("author")
        if author:
            parts.append(author)
        else:
            parts.append("مؤلف غير محدد")
        
        # Book title
        title = metadata.get("title")
        if title:
            parts.append(f'"{title}"')
        
        # Madhab
        madhab = metadata.get("madhab")
        if madhab and madhab != "general":
            madhab_arabic = {
                "hanafi": "المذهب الحنفي",
                "maliki": "المذهب المالكي",
                "shafii": "المذهب الشافعي",
                "hanbali": "المذهب الحنبلي",
                "nawazel": "النوازل",
            }.get(madhab.lower(), madhab)
            parts.append(madhab_arabic)
        
        # Volume and page
        volume = metadata.get("volume")
        page_start = metadata.get("page_start")
        page_end = metadata.get("page_end")
        
        if volume and volume > 1:
            parts.append(f"الجزء {volume}")
        
        if page_start:
            if page_end and page_end != page_start:
                parts.append(f"ص{page_start}-{page_end}")
            else:
                parts.append(f"ص{page_start}")
        
        # Chapter if available
        chapter = metadata.get("chapter")
        if chapter:
            chapter_short = chapter[:50] + "..." if len(chapter) > 50 else chapter
            parts.append(f"({chapter_short})")
        
        return f"[{index}] " + "، ".join(parts)
    
    def _format_book_short(self, metadata: dict, index: int) -> str:
        """Generate short reference for book."""
        title = metadata.get("title", "")
        if title:
            short_title = title.split()[0] if title else "كتاب"
            page = metadata.get("page_start", "")
            return f"[{index}: {short_title}، ص{page}]" if page else f"[{index}: {short_title}]"
        return f"[{index}]"
    
    def _format_hadith_citation(self, metadata: dict, index: int) -> str:
        """
        Format citation for a hadith source.
        
        Format: [Index] رواه [Source]، كتاب [Book]، رقم [Number]
        """
        parts = []
        
        # Source (collection)
        source = metadata.get("source")
        if source:
            parts.append(f"رواه {source}")
        
        # Book/chapter within collection
        book = metadata.get("book")
        if book:
            parts.append(f"كتاب {book}")
        
        # Hadith number
        number = metadata.get("number")
        if number:
            parts.append(f"رقم {number}")
        
        # Grade if available
        grade = metadata.get("grade")
        if grade:
            parts.append(f"({grade})")
        
        return f"[{index}] " + "، ".join(parts) if parts else f"[{index}] حديث"
    
    def _format_hadith_short(self, metadata: dict, index: int) -> str:
        """Generate short reference for hadith."""
        source = metadata.get("source", "حديث")
        number = metadata.get("number", "")
        if source and number:
            return f"[{index}: {source} #{number}]"
        return f"[{index}: {source}]" if source else f"[{index}]"
    
    def _format_islamic_scholarly(self, metadata: FatwaMetadata, index: int) -> str:
        """
        Format citation in Islamic scholarly style.
        
        Format: [Index] الشيخ [Scholar Name], "[Title/Topic]", [Source], [Date], [URL]
        """
        parts = []
        
        # Scholar
        if metadata.scholar:
            parts.append(metadata.scholar)
        else:
            parts.append("عالم غير محدد")
        
        # Title or question summary
        if metadata.title:
            title = metadata.title[:100] + "..." if len(metadata.title) > 100 else metadata.title
            parts.append(f'"{title}"')
        elif metadata.question:
            question = metadata.question[:80] + "..." if len(metadata.question) > 80 else metadata.question
            parts.append(f'"{question}"')
        
        # Source
        if metadata.source:
            parts.append(metadata.source)
        
        # Date
        if metadata.date:
            parts.append(metadata.date)
        
        # Fatwa ID
        if metadata.fatwa_id:
            parts.append(f"رقم الفتوى: {metadata.fatwa_id}")
        
        # URL
        if self.include_urls and metadata.url:
            parts.append(metadata.url)
        
        # Relevance score (if enabled)
        if self.include_scores:
            parts.append(f"(درجة الصلة: {metadata.extra.get('score', 'N/A')})")
        
        return f"[{index}] " + "، ".join(parts)
    
    def _format_simple(self, metadata: FatwaMetadata, index: int) -> str:
        """
        Format simple citation.
        
        Format: [Index] Scholar - Source
        """
        scholar = metadata.scholar or "Unknown Scholar"
        source = metadata.source or "Unknown Source"
        
        citation = f"[{index}] {scholar} - {source}"
        
        if metadata.fatwa_id:
            citation += f" (#{metadata.fatwa_id})"
        
        return citation
    
    def _format_detailed(self, metadata: FatwaMetadata, index: int) -> str:
        """
        Format detailed citation with all available information.
        """
        lines = [f"[{index}] فتوى / Fatwa:"]
        
        if metadata.scholar:
            lines.append(f"    العالم / Scholar: {metadata.scholar}")
        
        if metadata.title:
            lines.append(f"    العنوان / Title: {metadata.title}")
        
        if metadata.source:
            lines.append(f"    المصدر / Source: {metadata.source}")
        
        if metadata.fatwa_id:
            lines.append(f"    رقم الفتوى / Fatwa ID: {metadata.fatwa_id}")
        
        if metadata.date:
            lines.append(f"    التاريخ / Date: {metadata.date}")
        
        if metadata.category:
            lines.append(f"    التصنيف / Category: {metadata.category}")
        
        if self.include_urls and metadata.url:
            lines.append(f"    الرابط / URL: {metadata.url}")
        
        return "\n".join(lines)
    
    def _format_short(self, metadata: FatwaMetadata, index: int) -> str:
        """Generate short reference for inline citation."""
        if metadata.scholar:
            # Use last part of scholar name
            name_parts = metadata.scholar.split()
            short_name = name_parts[-1] if name_parts else metadata.scholar
            return f"[{index}: {short_name}]"
        
        if metadata.source:
            return f"[{index}: {metadata.source[:20]}]"
        
        return f"[{index}]"
    
    def format_references_section(
        self,
        citations: list[dict],
        title_ar: str = "المراجع",
        title_en: str = "References"
    ) -> str:
        """
        Format a complete references section.
        
        Args:
            citations: List of citation dictionaries
            title_ar: Arabic title for section
            title_en: English title for section
            
        Returns:
            Formatted references section
        """
        if not citations:
            return ""
        
        lines = [
            "",
            "─" * 40,
            f"{title_ar} / {title_en}:",
            ""
        ]
        
        for citation in citations:
            lines.append(citation.get("formatted", f"[{citation.get('index', '?')}] Unknown"))
        
        return "\n".join(lines)
    
    def inject_inline_citations(
        self,
        text: str,
        documents: list["RetrievalResult"],
    ) -> str:
        """
        Attempt to inject inline citation markers into generated text.
        
        This is a best-effort approach - works better when LLM is prompted
        to include citation markers.
        
        Args:
            text: Generated response text
            documents: Source documents
            
        Returns:
            Text with citation markers
        """
        # This is a simplified implementation
        # In practice, you'd want more sophisticated matching
        
        # Add reference section at end
        citations = self.generate_citations(documents)
        references = self.format_references_section(citations)
        
        return text + references
    
    def validate_citations(
        self,
        citations: list[dict],
        min_completeness: float = 0.5
    ) -> dict:
        """
        Validate citation quality.
        
        Returns validation results.
        """
        total = len(citations)
        if total == 0:
            return {"valid": False, "message": "No citations provided"}
        
        complete_count = 0
        issues = []
        
        for citation in citations:
            has_scholar = citation.get("scholar") is not None
            has_source = citation.get("source") is not None
            
            if has_scholar and has_source:
                complete_count += 1
            elif not has_scholar:
                issues.append(f"Citation [{citation.get('index')}] missing scholar")
            elif not has_source:
                issues.append(f"Citation [{citation.get('index')}] missing source")
        
        completeness = complete_count / total
        
        return {
            "valid": completeness >= min_completeness,
            "completeness": completeness,
            "complete_citations": complete_count,
            "total_citations": total,
            "issues": issues[:5],  # Limit issues reported
        }
