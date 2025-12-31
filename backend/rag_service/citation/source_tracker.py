"""
Source Tracker - Track document usage for citation generation.

Maintains provenance chain for all sources used in generating responses.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from loguru import logger


@dataclass
class TrackedSource:
    """A tracked source document."""
    
    document_id: str
    content_hash: str
    relevance_score: float
    metadata: dict
    retrieved_at: datetime
    used_in_response: bool = False
    chunk_indices: list[int] = field(default_factory=list)  # If document was chunked


@dataclass 
class QueryRecord:
    """Record of a query and its sources."""
    
    query_id: str
    query_text: str
    timestamp: datetime
    sources: list[TrackedSource] = field(default_factory=list)
    response_text: Optional[str] = None
    finalized: bool = False


class SourceTracker:
    """
    Track sources used in RAG responses.
    
    Features:
    - Track which documents are retrieved for each query
    - Record relevance scores and metadata
    - Maintain audit trail for citations
    - Support for query history
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize source tracker.
        
        Args:
            max_history: Maximum number of queries to keep in history
        """
        self.max_history = max_history
        self._queries: dict[str, QueryRecord] = {}
        self._query_order: list[str] = []  # For LRU eviction
    
    def start_query(self, query_text: str) -> str:
        """
        Start tracking a new query.
        
        Args:
            query_text: The user's query
            
        Returns:
            Unique query ID for tracking
        """
        query_id = str(uuid.uuid4())
        
        record = QueryRecord(
            query_id=query_id,
            query_text=query_text,
            timestamp=datetime.now(),
        )
        
        self._queries[query_id] = record
        self._query_order.append(query_id)
        
        # Evict old queries if needed
        self._evict_if_needed()
        
        logger.debug(f"Started tracking query: {query_id}")
        return query_id
    
    def add_source(
        self,
        query_id: str,
        document: "RetrievalResult",  # Forward reference
        used_chunks: Optional[list[int]] = None,
    ):
        """
        Add a source document to a query's tracking.
        
        Args:
            query_id: ID from start_query()
            document: Retrieved document
            used_chunks: Indices of chunks used (if document was split)
        """
        if query_id not in self._queries:
            logger.warning(f"Unknown query ID: {query_id}")
            return
        
        # Create content hash for deduplication
        content_hash = self._hash_content(document.content)
        
        tracked = TrackedSource(
            document_id=document.id,
            content_hash=content_hash,
            relevance_score=document.score,
            metadata=document.metadata.copy(),
            retrieved_at=datetime.now(),
            chunk_indices=used_chunks or [],
        )
        
        self._queries[query_id].sources.append(tracked)
    
    def mark_used(self, query_id: str, document_ids: list[str]):
        """
        Mark documents as actually used in the response.
        
        Args:
            query_id: Query ID
            document_ids: IDs of documents that were used
        """
        if query_id not in self._queries:
            return
        
        for source in self._queries[query_id].sources:
            if source.document_id in document_ids:
                source.used_in_response = True
    
    def finalize_query(self, query_id: str, response_text: str):
        """
        Finalize tracking for a query.
        
        Args:
            query_id: Query ID
            response_text: The generated response
        """
        if query_id not in self._queries:
            return
        
        record = self._queries[query_id]
        record.response_text = response_text
        record.finalized = True
        
        logger.debug(f"Finalized query {query_id} with {len(record.sources)} sources")
    
    def get_sources(self, query_id: str) -> list[TrackedSource]:
        """Get all sources for a query."""
        if query_id not in self._queries:
            return []
        return self._queries[query_id].sources
    
    def get_used_sources(self, query_id: str) -> list[TrackedSource]:
        """Get only sources that were used in the response."""
        sources = self.get_sources(query_id)
        return [s for s in sources if s.used_in_response]
    
    def get_query_record(self, query_id: str) -> Optional[QueryRecord]:
        """Get full query record."""
        return self._queries.get(query_id)
    
    def get_recent_queries(self, limit: int = 10) -> list[QueryRecord]:
        """Get most recent queries."""
        recent_ids = self._query_order[-limit:]
        return [self._queries[qid] for qid in reversed(recent_ids) if qid in self._queries]
    
    def export_audit_log(self, query_id: str) -> dict:
        """
        Export full audit log for a query.
        
        Useful for debugging and compliance.
        """
        record = self.get_query_record(query_id)
        if not record:
            return {}
        
        return {
            "query_id": record.query_id,
            "query_text": record.query_text,
            "timestamp": record.timestamp.isoformat(),
            "sources": [
                {
                    "document_id": s.document_id,
                    "relevance_score": s.relevance_score,
                    "used": s.used_in_response,
                    "metadata": s.metadata,
                    "retrieved_at": s.retrieved_at.isoformat(),
                }
                for s in record.sources
            ],
            "response_generated": record.response_text is not None,
            "finalized": record.finalized,
        }
    
    def _hash_content(self, content: str) -> str:
        """Create hash of content for deduplication."""
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _evict_if_needed(self):
        """Evict oldest queries if over limit."""
        while len(self._query_order) > self.max_history:
            oldest_id = self._query_order.pop(0)
            if oldest_id in self._queries:
                del self._queries[oldest_id]
