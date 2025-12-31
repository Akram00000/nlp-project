"""
Context Manager Module - Handles context window management.

Responsible for organizing retrieved documents, managing context window limits,
and implementing smart truncation strategies for different LLM providers.
"""

from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from .retriever import RetrievalResult


@dataclass
class ContextConfig:
    """Configuration for context management."""
    
    max_context_tokens: int = 4000  # Reserve space for response
    max_documents: int = 5
    chunk_overlap: int = 100  # Overlap when splitting
    include_metadata: bool = True
    document_separator: str = "\n\n---\n\n"
    prioritize_by_score: bool = True


@dataclass
class FormattedContext:
    """Formatted context ready for LLM."""
    
    text: str
    documents_used: list[RetrievalResult]
    total_tokens_estimate: int
    truncated: bool = False
    

class ContextManager:
    """
    Manages context window for RAG generation.
    
    Features:
    - Token estimation and limit management
    - Smart truncation strategies
    - Document formatting with metadata
    - Relevance-based prioritization
    """
    
    # Approximate tokens per character for estimation
    TOKENS_PER_CHAR_ARABIC = 0.5  # Arabic tends to have more tokens per char
    TOKENS_PER_CHAR_ENGLISH = 0.25
    
    def __init__(self, config: Optional[ContextConfig] = None):
        self.config = config or ContextConfig()
    
    def format_context(
        self,
        documents: list[RetrievalResult],
        max_tokens: Optional[int] = None,
        include_metadata: Optional[bool] = None,
    ) -> FormattedContext:
        """
        Format retrieved documents into context string.
        
        Args:
            documents: Retrieved documents to format
            max_tokens: Override max context tokens
            include_metadata: Override metadata inclusion
            
        Returns:
            FormattedContext with formatted text and metadata
        """
        max_tokens = max_tokens or self.config.max_context_tokens
        include_metadata = include_metadata if include_metadata is not None else self.config.include_metadata
        
        if not documents:
            return FormattedContext(
                text="",
                documents_used=[],
                total_tokens_estimate=0,
            )
        
        # Sort by score if configured
        if self.config.prioritize_by_score:
            documents = sorted(documents, key=lambda x: x.score, reverse=True)
        
        # Limit number of documents
        documents = documents[:self.config.max_documents]
        
        # Format documents and track token usage
        formatted_docs = []
        used_docs = []
        total_tokens = 0
        truncated = False
        
        for i, doc in enumerate(documents):
            # Format single document
            formatted = self._format_document(doc, i + 1, include_metadata)
            doc_tokens = self._estimate_tokens(formatted)
            
            # Check if we can fit this document
            if total_tokens + doc_tokens > max_tokens:
                # Try to truncate the document to fit
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 200:  # Minimum useful size
                    truncated_doc = self._truncate_document(
                        doc, i + 1, include_metadata, remaining_tokens
                    )
                    if truncated_doc:
                        formatted_docs.append(truncated_doc)
                        used_docs.append(doc)
                        truncated = True
                break
            
            formatted_docs.append(formatted)
            used_docs.append(doc)
            total_tokens += doc_tokens
        
        # Combine documents
        context_text = self.config.document_separator.join(formatted_docs)
        
        return FormattedContext(
            text=context_text,
            documents_used=used_docs,
            total_tokens_estimate=self._estimate_tokens(context_text),
            truncated=truncated,
        )
    
    def _format_document(
        self,
        doc: RetrievalResult,
        index: int,
        include_metadata: bool
    ) -> str:
        """Format a single document with optional metadata."""
        parts = []
        
        # Document header
        parts.append(f"[المصدر {index}]" if self._is_arabic(doc.content) else f"[Source {index}]")
        
        # Metadata if enabled
        if include_metadata:
            metadata_parts = []
            
            if doc.scholar:
                metadata_parts.append(f"العالم/Scholar: {doc.scholar}")
            if doc.source:
                metadata_parts.append(f"المصدر/Source: {doc.source}")
            if doc.title:
                metadata_parts.append(f"العنوان/Title: {doc.title}")
            if doc.date:
                metadata_parts.append(f"التاريخ/Date: {doc.date}")
            if doc.fatwa_id:
                metadata_parts.append(f"رقم الفتوى/Fatwa ID: {doc.fatwa_id}")
            
            # Relevance score
            metadata_parts.append(f"درجة الصلة/Relevance: {doc.score:.2f}")
            
            if metadata_parts:
                parts.append(" | ".join(metadata_parts))
        
        # Document content
        parts.append(doc.content)
        
        return "\n".join(parts)
    
    def _truncate_document(
        self,
        doc: RetrievalResult,
        index: int,
        include_metadata: bool,
        max_tokens: int
    ) -> Optional[str]:
        """Truncate document to fit within token limit."""
        # Calculate overhead from metadata
        header = f"[المصدر {index}]" if self._is_arabic(doc.content) else f"[Source {index}]"
        overhead_tokens = self._estimate_tokens(header) + 50  # Buffer for metadata
        
        available_tokens = max_tokens - overhead_tokens
        if available_tokens < 100:
            return None
        
        # Estimate characters to keep
        is_arabic = self._is_arabic(doc.content)
        tokens_per_char = self.TOKENS_PER_CHAR_ARABIC if is_arabic else self.TOKENS_PER_CHAR_ENGLISH
        chars_to_keep = int(available_tokens / tokens_per_char)
        
        # Truncate content
        truncated_content = doc.content[:chars_to_keep]
        
        # Try to end at a sentence boundary
        for end_char in ['. ', '。', '؟ ', '! ', '\n']:
            last_boundary = truncated_content.rfind(end_char)
            if last_boundary > len(truncated_content) * 0.7:  # Keep at least 70%
                truncated_content = truncated_content[:last_boundary + 1]
                break
        
        truncated_content += " [...]"
        
        # Create truncated document
        truncated_doc = RetrievalResult(
            id=doc.id,
            content=truncated_content,
            score=doc.score,
            metadata=doc.metadata,
        )
        
        return self._format_document(truncated_doc, index, include_metadata)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        if not text:
            return 0
        
        # Use different ratios for Arabic vs English
        is_arabic = self._is_arabic(text)
        tokens_per_char = self.TOKENS_PER_CHAR_ARABIC if is_arabic else self.TOKENS_PER_CHAR_ENGLISH
        
        return int(len(text) * tokens_per_char)
    
    def _is_arabic(self, text: str) -> bool:
        """Check if text is primarily Arabic."""
        if not text:
            return False
        
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        return arabic_chars > len(text) * 0.3
    
    def get_available_tokens(
        self,
        provider_max_tokens: int,
        reserved_for_response: int = 1000
    ) -> int:
        """Calculate available tokens for context given provider limits."""
        return provider_max_tokens - reserved_for_response
    
    def split_into_chunks(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: Optional[int] = None
    ) -> list[str]:
        """
        Split text into overlapping chunks.
        
        Useful for processing long documents before embedding.
        
        Args:
            text: Text to split
            chunk_size: Approximate tokens per chunk
            overlap: Token overlap between chunks
            
        Returns:
            List of text chunks
        """
        overlap = overlap if overlap is not None else self.config.chunk_overlap
        
        # Convert token sizes to character sizes
        is_arabic = self._is_arabic(text)
        tokens_per_char = self.TOKENS_PER_CHAR_ARABIC if is_arabic else self.TOKENS_PER_CHAR_ENGLISH
        
        char_chunk_size = int(chunk_size / tokens_per_char)
        char_overlap = int(overlap / tokens_per_char)
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + char_chunk_size
            
            # Try to end at a sentence boundary
            if end < len(text):
                for end_char in ['. ', '。', '؟ ', '! ', '\n']:
                    boundary = text.rfind(end_char, start, end + 100)
                    if boundary > start + char_chunk_size * 0.7:
                        end = boundary + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - char_overlap
        
        return chunks
