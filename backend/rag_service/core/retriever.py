"""
Retriever Module - Vector search and document retrieval.

Handles connection to Qdrant vector database and implements
semantic search with filtering and re-ranking capabilities.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from loguru import logger


@dataclass
class RetrievalResult:
    """A single retrieved document with metadata."""
    
    id: str
    content: str
    score: float
    metadata: dict = field(default_factory=dict)
    collection: str = ""  # Which collection this result came from
    
    # Fatwa-specific fields extracted from metadata
    @property
    def scholar(self) -> Optional[str]:
        return self.metadata.get("scholar") or self.metadata.get("mufti")
    
    @property
    def source(self) -> Optional[str]:
        return self.metadata.get("source") or self.metadata.get("website")
    
    @property
    def fatwa_id(self) -> Optional[str]:
        return self.metadata.get("fatwa_id") or self.metadata.get("id")
    
    @property
    def date(self) -> Optional[str]:
        return self.metadata.get("date") or self.metadata.get("published_date")
    
    @property
    def url(self) -> Optional[str]:
        return self.metadata.get("url") or self.metadata.get("link")
    
    @property
    def title(self) -> Optional[str]:
        return self.metadata.get("title") or self.metadata.get("question", "")[:100]
    
    @property
    def category(self) -> Optional[str]:
        return self.metadata.get("category") or self.metadata.get("topic")
    
    @property
    def source_type(self) -> str:
        """Get the type of source (fatwa, hadith, book)."""
        return self.metadata.get("type", "fatwa")
    
    # Book-specific fields
    @property
    def madhab(self) -> Optional[str]:
        return self.metadata.get("madhab")
    
    @property
    def author(self) -> Optional[str]:
        return self.metadata.get("author")
    
    @property
    def volume(self) -> Optional[int]:
        return self.metadata.get("volume")
    
    @property
    def page_start(self) -> Optional[int]:
        return self.metadata.get("page_start")
    
    @property
    def page_end(self) -> Optional[int]:
        return self.metadata.get("page_end")


@dataclass
class RetrievalQuery:
    """Query parameters for retrieval."""
    
    text: str
    top_k: int = 5
    similarity_threshold: float = 0.5
    filters: dict = field(default_factory=dict)  # Metadata filters
    rerank: bool = True
    madhab: Optional[str] = None  # Filter by madhab (for books)
    collections: list[str] = field(default_factory=lambda: ["fatwas"])  # Collections to search


class Retriever:
    """
    Document retriever using Qdrant vector database.
    
    Supports:
    - Semantic search using embeddings
    - Metadata filtering (scholar, date, category, etc.)
    - Re-ranking of results
    - Hybrid search (optional)
    """
    
    def __init__(
        self,
        collection_name: str = "fatwas",
        embedding_model: Optional[Any] = None,
        qdrant_client: Optional[Any] = None,
        reranker: Optional[Any] = None,
    ):
        self.collection_name = collection_name
        self._embedding_model = embedding_model
        self._qdrant_client = qdrant_client
        self._reranker = reranker
        self._initialized = False
        
    def initialize(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        embedding_model_name: str = "Omartificial-Intelligence-Space/GATE-AraBert-v1",
        reranker_path: Optional[str] = None,
    ):
        """
        Initialize Qdrant client and embedding model.
        
        Call this before using retrieve() or call retrieve() with auto-init.
        """
        # Initialize Qdrant client
        if self._qdrant_client is None:
            try:
                from qdrant_client import QdrantClient
                
                self._qdrant_client = QdrantClient(
                    host=host,
                    port=port,
                    api_key=api_key,
                    https=False,  # Use HTTP for local Qdrant
                    prefer_grpc=False,  # Use REST API instead of gRPC
                )
                logger.info(f"Connected to Qdrant at {host}:{port}")
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {e}")
                raise
        
        # Initialize embedding model
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                
                self._embedding_model = SentenceTransformer(embedding_model_name)
                logger.info(f"Loaded embedding model: {embedding_model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        
        # Initialize re-ranker if path provided
        if reranker_path and self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                
                self._reranker = CrossEncoder(reranker_path)
                logger.info(f"Loaded re-ranker: {reranker_path}")
            except Exception as e:
                logger.warning(f"Failed to load re-ranker: {e}. Continuing without re-ranking.")
        
        self._initialized = True
    
    def _ensure_initialized(self):
        """Ensure retriever is initialized, auto-initialize if not."""
        if not self._initialized:
            from ..config.settings import settings
            
            self.initialize(
                host=settings.qdrant.host,
                port=settings.qdrant.port,
                api_key=settings.qdrant.api_key,
                embedding_model_name=settings.embedding.model_name,
                reranker_path=settings.retrieval.rerank_model,
            )
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        self._ensure_initialized()
        return self._embedding_model.encode(texts).tolist()
    
    def retrieve(
        self,
        query: str | RetrievalQuery,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filters: Optional[dict] = None,
        rerank: bool = True,
        madhab: Optional[str] = None,
        collections: Optional[list[str]] = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text or RetrievalQuery object
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            filters: Metadata filters (e.g., {"scholar": "Ibn Baz"})
            rerank: Whether to re-rank results
            madhab: Filter by madhab (hanafi, maliki, shafii, hanbali)
            collections: Collections to search (default: just this retriever's collection)
            
        Returns:
            List of RetrievalResult objects sorted by relevance
        """
        self._ensure_initialized()
        
        # Parse query parameters
        if isinstance(query, str):
            query_text = query
        else:
            query_text = query.text
            top_k = top_k or query.top_k
            similarity_threshold = similarity_threshold if similarity_threshold is not None else query.similarity_threshold
            filters = filters or query.filters
            rerank = query.rerank
            madhab = madhab or query.madhab
            collections = collections or query.collections
        
        # Apply defaults from settings
        from ..config.settings import settings
        top_k = top_k or settings.retrieval.top_k
        similarity_threshold = similarity_threshold if similarity_threshold is not None else settings.retrieval.similarity_threshold
        
        # Default to this retriever's collection
        if not collections:
            collections = [self.collection_name]
        
        # Add madhab filter if specified
        if filters is None:
            filters = {}
        if madhab:
            filters["madhab"] = madhab.lower()
        
        # Generate query embedding
        query_embedding = self.embed([query_text])[0]
        
        # Build Qdrant filter
        qdrant_filter = self._build_filter(filters) if filters else None
        
        # Retrieve more documents than needed for re-ranking
        search_limit = top_k * 3 if rerank and self._reranker else top_k
        
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            all_results = []
            
            # Search each collection
            for collection in collections:
                try:
                    results = self._qdrant_client.query_points(
                        collection_name=collection,
                        query=query_embedding,
                        limit=search_limit,
                        query_filter=qdrant_filter,
                        score_threshold=similarity_threshold,
                    )
                    
                    # Convert to RetrievalResult objects
                    for hit in results.points:
                        result = RetrievalResult(
                            id=str(hit.id),
                            content=hit.payload.get("content", hit.payload.get("text", "")),
                            score=hit.score,
                            metadata=hit.payload,
                            collection=collection,
                        )
                        all_results.append(result)
                        
                except Exception as e:
                    logger.warning(f"Failed to search collection {collection}: {e}")
            
            # Sort all results by score
            all_results.sort(key=lambda x: x.score, reverse=True)
            
            # Re-rank if enabled and reranker available
            if rerank and self._reranker and all_results:
                all_results = self._rerank_results(query_text, all_results)
            
            # Return top_k results
            return all_results[:top_k]
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise
    
    async def retrieve_async(
        self,
        query: str | RetrievalQuery,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filters: Optional[dict] = None,
        rerank: bool = True,
    ) -> list[RetrievalResult]:
        """Async version of retrieve (runs sync in executor)."""
        import asyncio
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.retrieve(query, top_k, similarity_threshold, filters, rerank)
        )
    
    def _build_filter(self, filters: dict) -> Any:
        """Build Qdrant filter from dictionary."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        conditions = []
        for key, value in filters.items():
            if value is not None:
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
        
        if conditions:
            return Filter(must=conditions)
        return None
    
    def _rerank_results(
        self,
        query: str,
        results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Re-rank results using cross-encoder."""
        if not self._reranker or not results:
            return results
        
        # Create query-document pairs
        pairs = [(query, r.content) for r in results]
        
        # Get re-ranking scores
        scores = self._reranker.predict(pairs)
        
        # Update scores and sort
        for result, score in zip(results, scores):
            result.score = float(score)
        
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    def health_check(self) -> bool:
        """Check if Qdrant is accessible."""
        try:
            self._ensure_initialized()
            collections = self._qdrant_client.get_collections()
            return True
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")
            return False
    
    def get_collection_info(self) -> Optional[dict]:
        """Get information about the fatwas collection."""
        try:
            self._ensure_initialized()
            info = self._qdrant_client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
            }
        except Exception as e:
            logger.warning(f"Failed to get collection info: {e}")
            return None
