"""
RAG Engine - Main orchestrator for Retrieval-Augmented Generation.

Coordinates retrieval, context management, and generation to answer
queries about Islamic fatwas with proper citations.
"""

from dataclasses import dataclass, field
from typing import Optional, AsyncIterator
from loguru import logger

from .retriever import Retriever, RetrievalResult
from .generator import Generator, GenerationConfig
from .context_manager import ContextManager, ContextConfig
from ..providers.base import BaseLLMProvider, Message, LLMResponse
from ..citation.citation_generator import CitationGenerator
from ..citation.source_tracker import SourceTracker
from .linked_retrieval import LinkedRetriever, LinkedRetrievalResult


@dataclass
class RAGConfig:
    """Configuration for RAG Engine."""
    
    # Retrieval settings
    top_k: int = 5
    similarity_threshold: float = 0.5
    rerank: bool = True
    
    # Context settings
    max_context_tokens: int = 4000
    include_metadata_in_context: bool = True
    
    # Generation settings
    max_response_tokens: int = 1000
    temperature: float = 0.7
    
    # Citation settings
    include_citations: bool = True
    citation_format: str = "islamic_scholarly"
    
    # Linked retrieval settings (hadith fetching)
    use_linked_retrieval: bool = True
    hadith_collection: str = "hadiths"
    max_hadiths_per_fatwa: int = 2
    
    # Safety settings
    enable_safety_checks: bool = True


@dataclass
class RAGResponse:
    """Complete response from RAG Engine."""
    
    answer: str
    citations: list[dict] = field(default_factory=list)
    sources: list[RetrievalResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    @property
    def answer_with_citations(self) -> str:
        """Get answer with inline citations formatted."""
        if not self.citations:
            return self.answer
        
        # Append references section
        references = "\n\n---\nالمراجع / References:\n"
        for i, citation in enumerate(self.citations, 1):
            references += f"[{i}] {citation.get('formatted', citation.get('source', 'Unknown'))}\n"
        
        return self.answer + references


class RAGEngine:
    """
    Main RAG Engine for Islamic Fatwa Q&A.
    
    Pipeline:
    1. Query preprocessing
    2. Document retrieval from vector DB
    3. Context formatting and management
    4. LLM generation with context
    5. Citation generation
    6. Response post-processing
    
    Features:
    - Provider-agnostic LLM support
    - Automatic citation generation
    - Arabic/English support
    - Safety checks
    - Streaming responses
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        provider: Optional[BaseLLMProvider] = None,
        provider_name: Optional[str] = None,
        retriever: Optional[Retriever] = None,
        linked_retriever: Optional[LinkedRetriever] = None,
    ):
        """
        Initialize RAG Engine.
        
        Args:
            config: RAG configuration
            provider: Pre-configured LLM provider
            provider_name: Name of provider to use
            retriever: Pre-configured retriever
            linked_retriever: Pre-configured linked retriever for hadith fetching
        """
        self.config = config or RAGConfig()
        
        # Initialize components
        self.retriever = retriever or Retriever()
        self.linked_retriever = linked_retriever
        
        # Create linked retriever if enabled - REUSE the main retriever for fatwas!
        if self.config.use_linked_retrieval and not self.linked_retriever:
            # Create hadith retriever that will share embedding model and reranker
            hadith_retriever = Retriever(
                collection_name=self.config.hadith_collection,
                embedding_model=None,  # Will be set after main retriever initializes
                qdrant_client=None,  # Will be set after main retriever initializes
                reranker=None,  # Will be set after main retriever initializes
            )
            
            self.linked_retriever = LinkedRetriever(
                fatwa_collection="fatwas",
                hadith_collection=self.config.hadith_collection,
                fatwa_retriever=self.retriever,  # Reuse main retriever!
                hadith_retriever=hadith_retriever,  # Separate retriever for hadiths
            )
        
        self.generator = Generator(provider=provider, provider_name=provider_name)
        self.context_manager = ContextManager(ContextConfig(
            max_context_tokens=self.config.max_context_tokens,
            max_documents=self.config.top_k,
            include_metadata=self.config.include_metadata_in_context,
        ))
        self.citation_generator = CitationGenerator(format_style=self.config.citation_format)
        self.source_tracker = SourceTracker()
        
        # Pre-load models during initialization instead of on first query
        logger.info("Initializing retrievers and loading models...")
        self.retriever.initialize()
        
        # Share models with hadith retriever if using linked retrieval
        if self.linked_retriever and self.linked_retriever.hadith_retriever:
            self.linked_retriever._share_models_if_needed()
        
        logger.info("RAG Engine initialization complete")
        
        # System prompt for fatwa Q&A
        self._system_prompt: Optional[str] = None
    
    def set_system_prompt(self, prompt: str):
        """Set custom system prompt."""
        self._system_prompt = prompt
    
    def set_provider(self, provider: BaseLLMProvider | str):
        """Switch LLM provider."""
        self.generator.set_provider(provider)
    
    def query(
        self,
        question: str,
        filters: Optional[dict] = None,
        language: Optional[str] = None,
        **kwargs
    ) -> RAGResponse:
        """
        Process a query and generate a response.
        
        Args:
            question: User's question
            filters: Metadata filters for retrieval (scholar, category, etc.)
            language: Override language detection (ar/en)
            **kwargs: Additional generation parameters
            
        Returns:
            RAGResponse with answer, citations, and metadata
        """
        logger.info(f"Processing query: {question[:100]}...")
        
        # 1. Preprocess query
        processed_query = self._preprocess_query(question, language)
        
        # 2. Retrieve relevant documents (with linked hadiths if enabled)
        if self.config.use_linked_retrieval and self.linked_retriever:
            linked_result = self.linked_retriever.retrieve_with_links(
                query=processed_query,
                top_k_fatwas=self.config.top_k,
                top_k_hadiths_per_fatwa=self.config.max_hadiths_per_fatwa,
            )
            documents = linked_result.all_documents
            logger.info(f"Retrieved {len(linked_result.primary_docs)} fatwas + {len(linked_result.linked_hadiths)} hadiths")
        else:
            documents = self.retriever.retrieve(
                query=processed_query,
                top_k=self.config.top_k,
                similarity_threshold=self.config.similarity_threshold,
                filters=filters,
                rerank=self.config.rerank,
            )
        
        if not documents:
            logger.warning("No documents retrieved for query")
            return RAGResponse(
                answer=self._get_no_results_message(language),
                citations=[],
                sources=[],
                metadata={"query": question, "documents_retrieved": 0}
            )
        
        logger.info(f"Retrieved {len(documents)} total documents for context")
        
        # 3. Track sources
        query_id = self.source_tracker.start_query(question)
        for doc in documents:
            self.source_tracker.add_source(query_id, doc)
        
        # 4. Format context
        formatted_context = self.context_manager.format_context(
            documents,
            include_metadata=self.config.include_metadata_in_context,
        )
        
        # 5. Build messages
        messages = self._build_messages(question, formatted_context.text, language)
        
        # 6. Generate response
        gen_config = GenerationConfig(
            max_tokens=self.config.max_response_tokens,
            temperature=self.config.temperature,
        )
        
        response = self.generator.generate(messages, config=gen_config, **kwargs)
        
        # 7. Generate citations
        citations = []
        if self.config.include_citations:
            citations = self.citation_generator.generate_citations(
                formatted_context.documents_used
            )
        
        # 8. Finalize tracking
        self.source_tracker.finalize_query(query_id, response.content)
        
        return RAGResponse(
            answer=response.content,
            citations=citations,
            sources=formatted_context.documents_used,
            metadata={
                "query": question,
                "processed_query": processed_query,
                "documents_retrieved": len(documents),
                "documents_used": len(formatted_context.documents_used),
                "context_truncated": formatted_context.truncated,
                "tokens_used": response.total_tokens,
                "provider": response.provider,
                "model": response.model,
            }
        )
    
    async def query_async(
        self,
        question: str,
        filters: Optional[dict] = None,
        language: Optional[str] = None,
        **kwargs
    ) -> RAGResponse:
        """Async version of query."""
        logger.info(f"Processing async query: {question[:100]}...")
        
        processed_query = self._preprocess_query(question, language)
        
        # Async retrieval
        documents = await self.retriever.retrieve_async(
            query=processed_query,
            top_k=self.config.top_k,
            similarity_threshold=self.config.similarity_threshold,
            filters=filters,
            rerank=self.config.rerank,
        )
        
        if not documents:
            return RAGResponse(
                answer=self._get_no_results_message(language),
                citations=[],
                sources=[],
                metadata={"query": question, "documents_retrieved": 0}
            )
        
        # Track sources
        query_id = self.source_tracker.start_query(question)
        for doc in documents:
            self.source_tracker.add_source(query_id, doc)
        
        # Format context
        formatted_context = self.context_manager.format_context(documents)
        
        # Build messages and generate
        messages = self._build_messages(question, formatted_context.text, language)
        gen_config = GenerationConfig(
            max_tokens=self.config.max_response_tokens,
            temperature=self.config.temperature,
        )
        
        response = await self.generator.generate_async(messages, config=gen_config, **kwargs)
        
        # Generate citations
        citations = []
        if self.config.include_citations:
            citations = self.citation_generator.generate_citations(
                formatted_context.documents_used
            )
        
        self.source_tracker.finalize_query(query_id, response.content)
        
        return RAGResponse(
            answer=response.content,
            citations=citations,
            sources=formatted_context.documents_used,
            metadata={
                "query": question,
                "documents_retrieved": len(documents),
                "documents_used": len(formatted_context.documents_used),
                "tokens_used": response.total_tokens,
                "provider": response.provider,
            }
        )
    
    async def query_stream(
        self,
        question: str,
        filters: Optional[dict] = None,
        language: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream response tokens for a query.
        
        Note: Citations are not available in streaming mode until completion.
        Use query() or query_async() if citations are needed.
        
        Yields:
            Response tokens as they're generated
        """
        processed_query = self._preprocess_query(question, language)
        
        documents = await self.retriever.retrieve_async(
            query=processed_query,
            top_k=self.config.top_k,
            similarity_threshold=self.config.similarity_threshold,
            filters=filters,
            rerank=self.config.rerank,
        )
        
        if not documents:
            yield self._get_no_results_message(language)
            return
        
        formatted_context = self.context_manager.format_context(documents)
        messages = self._build_messages(question, formatted_context.text, language)
        
        async for token in self.generator.stream(messages, **kwargs):
            yield token
    
    def _preprocess_query(self, query: str, language: Optional[str] = None) -> str:
        """Preprocess query for retrieval."""
        from ..utils.preprocessing import preprocess_arabic_text, detect_language
        
        # Detect language if not provided
        if language is None:
            language = detect_language(query)
        
        # Apply Arabic preprocessing if needed
        if language == "ar":
            query = preprocess_arabic_text(query)
        
        return query.strip()
    
    def _build_messages(
        self,
        question: str,
        context: str,
        language: Optional[str] = None
    ) -> list[Message]:
        """Build messages for LLM generation."""
        from ..utils.prompts import get_system_prompt, build_rag_prompt
        
        messages = []
        
        # System prompt
        system_prompt = self._system_prompt or get_system_prompt(language)
        messages.append(Message(role="system", content=system_prompt))
        logger.debug(f"System prompt length: {len(system_prompt)} chars")
        
        # User message with context
        user_message = build_rag_prompt(question, context, language)
        messages.append(Message(role="user", content=user_message))
        
        # Log prompt details
        logger.info(f"Prompt token estimate: ~{(len(system_prompt) + len(user_message)) // 4} tokens")
        logger.debug(f"Context length: {len(context)} chars")
        logger.debug(f"User message length: {len(user_message)} chars")
        logger.debug(f"Full user message preview (first 500 chars): {user_message[:500]}...")
        
        return messages
    
    def _get_no_results_message(self, language: Optional[str] = None) -> str:
        """Get appropriate no results message."""
        if language == "ar":
            return "عذراً، لم أتمكن من العثور على معلومات ذات صلة بسؤالك في قاعدة البيانات. يرجى إعادة صياغة السؤال أو استشارة عالم متخصص."
        return "Sorry, I couldn't find relevant information for your question in the database. Please try rephrasing your question or consult a qualified scholar."
    
    def health_check(self) -> dict:
        """Check health of all components."""
        return {
            "retriever": self.retriever.health_check(),
            "generator": self.generator.health_check(),
            "overall": self.retriever.health_check() and self.generator.health_check(),
        }
    
    def get_collection_info(self) -> Optional[dict]:
        """Get information about the fatwas collection."""
        return self.retriever.get_collection_info()
