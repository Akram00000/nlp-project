"""
Linked Retrieval - Retrieve fatwas, hadiths, and books with cross-references.

Implements multi-stage retrieval:
1. Primary: Retrieve relevant fatwas
2. Secondary: Retrieve hadiths mentioned in fatwas
3. Tertiary: Retrieve relevant book passages for detailed fiqh context
"""

from typing import Optional
from dataclasses import dataclass
from loguru import logger

from ..core.retriever import Retriever, RetrievalResult
from ..data_ingestion.reference_extractor import ReferenceExtractor


@dataclass
class LinkedRetrievalResult:
    """Result with primary and linked documents."""
    
    primary_docs: list[RetrievalResult]  # Fatwas
    linked_hadiths: list[RetrievalResult]  # Referenced hadiths
    linked_books: list[RetrievalResult]  # Related book passages
    total_documents: int
    
    @property
    def all_documents(self) -> list[RetrievalResult]:
        """Get all documents (fatwas + hadiths + books) for context."""
        return self.primary_docs + self.linked_hadiths + self.linked_books
    
    @property
    def fatwas(self) -> list[RetrievalResult]:
        return self.primary_docs
    
    @property
    def hadiths(self) -> list[RetrievalResult]:
        return self.linked_hadiths
    
    @property
    def books(self) -> list[RetrievalResult]:
        return self.linked_books


class LinkedRetriever:
    """
    Retriever that follows references to fetch linked documents.
    
    Supports:
    - Fatwas (primary search)
    - Hadiths (linked via references or semantic similarity)
    - Books (linked via mentions or madhab-filtered semantic search)
    
    Usage:
    ```python
    retriever = LinkedRetriever()
    result = retriever.retrieve_with_links("ما حكم الصلاة؟")
    
    # Use in RAG
    all_docs = result.all_documents  # Fatwas + Hadiths + Books
    
    # With madhab filter
    result = retriever.retrieve_with_links("ما حكم...", madhab="hanafi")
    ```
    """
    
    # Book reference patterns in fatwas
    BOOK_REFERENCE_PATTERNS = [
        r'قال في\s+(.+?)(?:[:\s]|$)',       # قال في المغني
        r'ذكره?\s+(?:في\s+)?(.+?)(?:[:\s]|$)',  # ذكره في الأم
        r'انظر\s+(.+?)(?:[:\s]|$)',          # انظر البداية
        r'كما في\s+(.+?)(?:[:\s]|$)',        # كما في المجموع
        r'ورد في\s+(.+?)(?:[:\s]|$)',        # ورد في الفتاوى
    ]
    
    def __init__(
        self,
        fatwa_collection: str = "fatwas",
        hadith_collection: str = "hadiths",
        book_collection: str = "books",
        fatwa_retriever: Optional[Retriever] = None,
        hadith_retriever: Optional[Retriever] = None,
        book_retriever: Optional[Retriever] = None,
    ):
        """
        Initialize linked retriever.
        
        Args:
            fatwa_collection: Fatwa collection name
            hadith_collection: Hadith collection name
            book_collection: Book collection name
            fatwa_retriever: Pre-configured fatwa retriever
            hadith_retriever: Pre-configured hadith retriever
            book_retriever: Pre-configured book retriever
        """
        self.fatwa_collection = fatwa_collection
        self.hadith_collection = hadith_collection
        self.book_collection = book_collection
        
        # Create retrievers
        self.fatwa_retriever = fatwa_retriever or Retriever(
            collection_name=fatwa_collection
        )
        
        self.hadith_retriever = hadith_retriever or Retriever(
            collection_name=hadith_collection
        )
        
        self.book_retriever = book_retriever or Retriever(
            collection_name=book_collection
        )
        
        # If fatwa retriever is initialized, share models
        if fatwa_retriever and fatwa_retriever._initialized:
            self._share_models_if_needed()
        
        self.ref_extractor = ReferenceExtractor()
    
    def _share_models_if_needed(self):
        """Share embedding model and reranker between retrievers to avoid double loading."""
        if self.fatwa_retriever._initialized:
            for retriever in [self.hadith_retriever, self.book_retriever]:
                if not retriever._initialized:
                    logger.info(f"Sharing embedding model with {retriever.collection_name} retriever")
                    retriever._embedding_model = self.fatwa_retriever._embedding_model
                    retriever._qdrant_client = self.fatwa_retriever._qdrant_client
                    retriever._reranker = self.fatwa_retriever._reranker
                    retriever._initialized = True
    
    def retrieve_with_links(
        self,
        query: str,
        top_k_fatwas: int = 5,
        top_k_hadiths_per_fatwa: int = 2,
        top_k_books: int = 3,
        retrieve_hadiths: bool = True,
        retrieve_books: bool = True,
        use_semantic_fallback: bool = True,
        madhab: Optional[str] = None,
    ) -> LinkedRetrievalResult:
        """
        Retrieve fatwas and their linked hadiths and books.
        
        Args:
            query: User query
            top_k_fatwas: Number of fatwas to retrieve
            top_k_hadiths_per_fatwa: Max hadiths per fatwa
            top_k_books: Number of book passages to retrieve
            retrieve_hadiths: Whether to retrieve linked hadiths
            retrieve_books: Whether to retrieve book passages
            use_semantic_fallback: Use semantic search if pattern extraction fails
            madhab: Filter books by madhab (hanafi, maliki, shafii, hanbali)
            
        Returns:
            LinkedRetrievalResult with fatwas, hadiths, and books
        """
        # Ensure retrievers share models if not yet initialized
        if self.fatwa_retriever._initialized:
            self._share_models_if_needed()
        
        # Step 1: Retrieve primary fatwas
        fatwas = self.fatwa_retriever.retrieve(
            query=query,
            top_k=top_k_fatwas,
        )
        
        logger.info(f"Retrieved {len(fatwas)} fatwas for query")
        
        # Step 2: Retrieve linked hadiths
        linked_hadiths = []
        
        if retrieve_hadiths and fatwas:
            self._share_models_if_needed()
            
            # Try pattern-based extraction first
            hadith_queries = self._extract_hadith_queries(fatwas)
            
            if hadith_queries:
                logger.info(f"Found {len(hadith_queries)} hadith references via pattern extraction")
                
                for hadith_query in hadith_queries[:top_k_hadiths_per_fatwa * len(fatwas)]:
                    try:
                        results = self.hadith_retriever.retrieve(
                            query=hadith_query,
                            top_k=1,
                            similarity_threshold=0.15,
                        )
                        linked_hadiths.extend(results)
                    except Exception as e:
                        logger.warning(f"Failed to retrieve hadith for query '{hadith_query[:50]}': {e}")
                
                # Deduplicate
                linked_hadiths = self._deduplicate_results(linked_hadiths)
                linked_hadiths = linked_hadiths[:top_k_fatwas * top_k_hadiths_per_fatwa]
                logger.info(f"Retrieved {len(linked_hadiths)} unique hadiths via patterns")
            
            elif use_semantic_fallback:
                logger.info("No hadith references found via patterns, using semantic search")
                linked_hadiths = self._retrieve_hadiths_by_topic(
                    query=query,
                    fatwas=fatwas,
                    top_k=min(3, top_k_hadiths_per_fatwa),
                )
                logger.info(f"Retrieved {len(linked_hadiths)} hadiths via semantic search")
        
        # Step 3: Retrieve book passages
        linked_books = []
        
        if retrieve_books:
            self._share_models_if_needed()
            
            # Try pattern-based extraction first
            book_queries = self._extract_book_queries(fatwas)
            
            if book_queries:
                logger.info(f"Found {len(book_queries)} book references via pattern extraction")
                
                for book_query in book_queries[:top_k_books]:
                    try:
                        results = self.book_retriever.retrieve(
                            query=book_query,
                            top_k=1,
                            similarity_threshold=0.15,
                            madhab=madhab,
                        )
                        linked_books.extend(results)
                    except Exception as e:
                        logger.warning(f"Failed to retrieve book for query '{book_query[:50]}': {e}")
                
                linked_books = self._deduplicate_results(linked_books)
                logger.info(f"Retrieved {len(linked_books)} unique book passages via patterns")
            
            # Always do semantic search for books (they provide detailed fiqh context)
            if len(linked_books) < top_k_books:
                remaining = top_k_books - len(linked_books)
                logger.info(f"Retrieving {remaining} more book passages via semantic search")
                
                try:
                    semantic_books = self.book_retriever.retrieve(
                        query=query,
                        top_k=remaining,
                        similarity_threshold=0.15,
                        madhab=madhab,
                    )
                    
                    # Add only books not already in results
                    existing_ids = {b.id for b in linked_books}
                    for book in semantic_books:
                        if book.id not in existing_ids:
                            linked_books.append(book)
                    
                    logger.info(f"Total book passages: {len(linked_books)}")
                except Exception as e:
                    logger.warning(f"Semantic book retrieval failed: {e}")
        
        return LinkedRetrievalResult(
            primary_docs=fatwas,
            linked_hadiths=linked_hadiths,
            linked_books=linked_books,
            total_documents=len(fatwas) + len(linked_hadiths) + len(linked_books),
        )
    
    def _deduplicate_results(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Deduplicate results by ID."""
        seen_ids = set()
        unique = []
        for r in results:
            if r.id not in seen_ids:
                seen_ids.add(r.id)
                unique.append(r)
        return unique
    
    def _extract_hadith_queries(self, fatwas: list[RetrievalResult]) -> list[str]:
        """Extract hadith search queries from fatwa content."""
        queries = []
        
        for fatwa in fatwas:
            # Extract references
            refs = self.ref_extractor.extract_hadiths(fatwa.content)
            
            for ref in refs:
                query = ref.get('query')
                if query and len(query) > 10:  # Meaningful query
                    queries.append(query)
        
        return queries
    
    def _extract_book_queries(self, fatwas: list[RetrievalResult]) -> list[str]:
        """Extract book reference queries from fatwa content."""
        import re
        queries = []
        
        for fatwa in fatwas:
            content = fatwa.content
            
            for pattern in self.BOOK_REFERENCE_PATTERNS:
                matches = re.findall(pattern, content)
                for match in matches:
                    # Clean up the match
                    book_ref = match.strip()
                    if len(book_ref) > 3 and len(book_ref) < 100:
                        queries.append(book_ref)
        
        return queries
    
    def _retrieve_hadiths_by_topic(
        self,
        query: str,
        fatwas: list[RetrievalResult],
        top_k: int = 3,
    ) -> list[RetrievalResult]:
        """
        Retrieve topically relevant hadiths using semantic search.
        
        Fallback when pattern extraction doesn't find explicit references.
        Uses the original user query to find hadiths about the same topic.
        
        Args:
            query: Original user query
            fatwas: Retrieved fatwas (for context, not currently used)
            top_k: Number of hadiths to retrieve
            
        Returns:
            List of topically relevant hadiths
        """
        try:
            # Search hadiths using the original query
            # Lower threshold since we're doing topical matching
            hadiths = self.hadith_retriever.retrieve(
                query=query,
                top_k=top_k,
                similarity_threshold=0.15,  # Lower threshold for semantic relevance
            )
            
            return hadiths
            
        except Exception as e:
            logger.error(f"Semantic hadith retrieval failed: {e}")
            return []
    
    def health_check(self) -> dict:
        """Check health of all retrievers."""
        return {
            'fatwa_retriever': self.fatwa_retriever.health_check(),
            'hadith_retriever': self.hadith_retriever.health_check(),
            'book_retriever': self.book_retriever.health_check(),
        }
