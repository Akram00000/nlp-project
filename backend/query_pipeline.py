#!/usr/bin/env python3
"""
RAG Pipeline Query Script - Execute full pipeline and analyze results.

Usage:
    python query_pipeline.py "ما حكم صلاة الجمعة؟"
    python query_pipeline.py "ما حكم الزكاة؟" --madhab hanafi --output results.json
    python query_pipeline.py "what is the ruling on fasting?" --no-books --top-k 10
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from loguru import logger

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rag_service.core.rag_engine import RAGEngine, RAGConfig, RAGResponse
from rag_service.core.retriever import Retriever, RetrievalResult
from rag_service.core.linked_retrieval import LinkedRetriever, LinkedRetrievalResult
from rag_service.core.context_manager import ContextManager, ContextConfig
from rag_service.citation.citation_generator import CitationGenerator
from rag_service.providers.factory import ProviderFactory
from rag_service.utils.preprocessing import detect_language


@dataclass
class SourceAnalysis:
    """Analysis of a single retrieved source."""
    id: str
    type: str  # fatwa, hadith, book
    score: float
    collection: str
    content_preview: str
    
    # Type-specific fields
    scholar: Optional[str] = None
    source: Optional[str] = None
    fatwa_id: Optional[str] = None
    
    # Hadith fields
    narrator: Optional[str] = None
    hadith_source: Optional[str] = None
    hadith_number: Optional[str] = None
    
    # Book fields
    book_title: Optional[str] = None
    author: Optional[str] = None
    madhab: Optional[str] = None
    volume: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    chapter: Optional[str] = None


@dataclass
class CitationAnalysis:
    """Analysis of generated citations."""
    index: int
    document_id: str
    source_type: str
    formatted: str
    short: str
    relevance_score: float


@dataclass
class PipelineAnalysis:
    """Complete analysis of the RAG pipeline execution."""
    
    # Query info
    query: str
    processed_query: str
    detected_language: str
    timestamp: str
    
    # Retrieval stats
    total_documents_retrieved: int
    fatwas_retrieved: int
    hadiths_retrieved: int
    books_retrieved: int
    
    # Sources breakdown
    sources: list[SourceAnalysis] = field(default_factory=list)
    
    # Context stats
    documents_used_in_context: int = 0
    context_truncated: bool = False
    estimated_context_tokens: int = 0
    
    # Generation stats
    provider: Optional[str] = None
    model: Optional[str] = None
    tokens_used: Optional[int] = None
    
    # Citations
    citations: list[CitationAnalysis] = field(default_factory=list)
    citation_format: str = "islamic_scholarly"
    
    # Response
    answer: str = ""
    answer_with_citations: str = ""
    
    # Filters applied
    madhab_filter: Optional[str] = None
    other_filters: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
    
    def print_summary(self):
        """Print a human-readable summary."""
        print(self.to_text())
    
    def to_text(self) -> str:
        """Generate a detailed text report of the analysis."""
        lines = []
        
        lines.append("\n" + "=" * 70)
        lines.append("RAG PIPELINE ANALYSIS REPORT")
        lines.append("=" * 70)
        
        lines.append(f"\n📝 Query: {self.query}")
        lines.append(f"   Language: {self.detected_language}")
        lines.append(f"   Timestamp: {self.timestamp}")
        
        if self.madhab_filter:
            lines.append(f"   Madhab Filter: {self.madhab_filter}")
        
        lines.append(f"\n📚 RETRIEVAL STATISTICS")
        lines.append(f"   Total Documents: {self.total_documents_retrieved}")
        lines.append(f"   ├── Fatwas: {self.fatwas_retrieved}")
        lines.append(f"   ├── Hadiths: {self.hadiths_retrieved}")
        lines.append(f"   └── Books: {self.books_retrieved}")
        
        lines.append(f"\n📄 CONTEXT")
        lines.append(f"   Documents in Context: {self.documents_used_in_context}")
        lines.append(f"   Estimated Tokens: {self.estimated_context_tokens}")
        lines.append(f"   Truncated: {'Yes' if self.context_truncated else 'No'}")
        
        if self.provider:
            lines.append(f"\n🤖 GENERATION")
            lines.append(f"   Provider: {self.provider}")
            lines.append(f"   Model: {self.model}")
            lines.append(f"   Tokens Used: {self.tokens_used}")
        
        lines.append(f"\n📖 CITATIONS ({len(self.citations)} total)")
        for cit in self.citations:
            lines.append(f"   [{cit.index}] {cit.source_type}: {cit.short}")
        
        lines.append(f"\n📋 DETAILED SOURCES BREAKDOWN")
        
        # Group by type
        fatwas = [s for s in self.sources if s.type == "fatwa"]
        hadiths = [s for s in self.sources if s.type == "hadith"]
        books = [s for s in self.sources if s.type == "book"]
        
        if fatwas:
            lines.append(f"\n   FATWAS ({len(fatwas)}):")
            for i, s in enumerate(fatwas, 1):
                lines.append(f"\n   [{i}] Score: {s.score:.4f}")
                if s.scholar:
                    lines.append(f"       Scholar: {s.scholar}")
                if s.source:
                    lines.append(f"       Source: {s.source}")
                if s.fatwa_id:
                    lines.append(f"       Fatwa ID: {s.fatwa_id}")
                lines.append(f"       Content: {s.content_preview}")
        
        if hadiths:
            lines.append(f"\n   HADITHS ({len(hadiths)}):")
            for i, s in enumerate(hadiths, 1):
                lines.append(f"\n   [{i}] Score: {s.score:.4f}")
                if s.hadith_source:
                    lines.append(f"       Source: {s.hadith_source}")
                if s.hadith_number:
                    lines.append(f"       Number: {s.hadith_number}")
                if s.narrator:
                    lines.append(f"       Narrator: {s.narrator}")
                lines.append(f"       Content: {s.content_preview}")
        
        if books:
            lines.append(f"\n   BOOKS ({len(books)}):")
            for i, s in enumerate(books, 1):
                lines.append(f"\n   [{i}] Score: {s.score:.4f}")
                if s.book_title:
                    lines.append(f"       Title: {s.book_title}")
                if s.author:
                    lines.append(f"       Author: {s.author}")
                if s.madhab:
                    lines.append(f"       Madhab: {s.madhab}")
                if s.volume:
                    lines.append(f"       Volume: {s.volume}")
                if s.page_start:
                    pages = f"pp.{s.page_start}-{s.page_end}" if s.page_end and s.page_end != s.page_start else f"p.{s.page_start}"
                    lines.append(f"       Pages: {pages}")
                if s.chapter:
                    lines.append(f"       Chapter: {s.chapter}")
                lines.append(f"       Content: {s.content_preview}")
        
        lines.append(f"\n💬 COMPLETE ANSWER")
        lines.append("-" * 70)
        lines.append(self.answer if self.answer else "(No answer generated)")
        lines.append("-" * 70)
        
        lines.append("\n" + "=" * 70 + "\n")
        
        return "\n".join(lines)


def analyze_source(doc: RetrievalResult) -> SourceAnalysis:
    """Analyze a single retrieved document."""
    source_type = doc.metadata.get("type", "fatwa")
    
    analysis = SourceAnalysis(
        id=doc.id,
        type=source_type,
        score=doc.score,
        collection=doc.collection or "unknown",
        content_preview=doc.content if doc.content else "",  # Store full content for detailed report
    )
    
    if source_type == "fatwa":
        analysis.scholar = doc.metadata.get("scholar")
        analysis.source = doc.metadata.get("source")
        analysis.fatwa_id = doc.metadata.get("fatwa_id")
    
    elif source_type == "hadith":
        analysis.narrator = doc.metadata.get("narrator")
        analysis.hadith_source = doc.metadata.get("source")
        analysis.hadith_number = doc.metadata.get("number")
    
    elif source_type == "book":
        analysis.book_title = doc.metadata.get("title")
        analysis.author = doc.metadata.get("author")
        analysis.madhab = doc.metadata.get("madhab")
        analysis.volume = doc.metadata.get("volume")
        analysis.page_start = doc.metadata.get("page_start")
        analysis.page_end = doc.metadata.get("page_end")
        analysis.chapter = doc.metadata.get("chapter")
    
    return analysis


def run_pipeline(
    query: str,
    madhab: Optional[str] = None,
    top_k_fatwas: int = 5,
    top_k_hadiths: int = 2,
    top_k_books: int = 3,
    retrieve_books: bool = True,
    retrieve_hadiths: bool = True,
    provider: Optional[str] = None,
    generate: bool = True,
    max_context_tokens: int = 4000,
    max_generation_tokens: int = 2000,
) -> PipelineAnalysis:
    """
    Run the full RAG pipeline and return analysis.
    
    Args:
        query: User's question
        madhab: Filter books by madhab (hanafi, maliki, shafii, hanbali)
        top_k_fatwas: Number of fatwas to retrieve
        top_k_hadiths: Number of hadiths per fatwa
        top_k_books: Number of book passages to retrieve
        retrieve_books: Whether to retrieve book passages
        retrieve_hadiths: Whether to retrieve hadiths
        provider: LLM provider name (lmstudio, gemini, etc.)
        generate: Whether to generate response (False = retrieval only)
        max_context_tokens: Maximum context tokens
        max_generation_tokens: Maximum tokens for LLM generation
        
    Returns:
        PipelineAnalysis with full breakdown
    """
    timestamp = datetime.now().isoformat()
    detected_lang = detect_language(query)
    
    logger.info(f"Starting pipeline for query: {query[:50]}...")
    logger.info(f"Settings: madhab={madhab}, books={retrieve_books}, hadiths={retrieve_hadiths}")
    
    # Initialize analysis
    analysis = PipelineAnalysis(
        query=query,
        processed_query=query,  # Will be updated
        detected_language=detected_lang,
        timestamp=timestamp,
        total_documents_retrieved=0,
        fatwas_retrieved=0,
        hadiths_retrieved=0,
        books_retrieved=0,
        madhab_filter=madhab,
    )
    
    # Initialize retrievers
    logger.info("Initializing retrievers...")
    
    fatwa_retriever = Retriever(collection_name="fatwas")
    hadith_retriever = Retriever(collection_name="hadiths")
    book_retriever = Retriever(collection_name="books")
    
    # Initialize main retriever (loads embedding model)
    fatwa_retriever.initialize()
    
    # Share models with other retrievers
    for retriever in [hadith_retriever, book_retriever]:
        retriever._embedding_model = fatwa_retriever._embedding_model
        retriever._qdrant_client = fatwa_retriever._qdrant_client
        retriever._reranker = fatwa_retriever._reranker
        retriever._initialized = True
    
    # Create linked retriever
    linked_retriever = LinkedRetriever(
        fatwa_retriever=fatwa_retriever,
        hadith_retriever=hadith_retriever,
        book_retriever=book_retriever,
    )
    
    # Run linked retrieval
    logger.info("Running linked retrieval...")
    
    result: LinkedRetrievalResult = linked_retriever.retrieve_with_links(
        query=query,
        top_k_fatwas=top_k_fatwas,
        top_k_hadiths_per_fatwa=top_k_hadiths,
        top_k_books=top_k_books,
        retrieve_hadiths=retrieve_hadiths,
        retrieve_books=retrieve_books,
        madhab=madhab,
    )
    
    # Analyze retrieved documents
    all_docs = result.all_documents
    
    analysis.fatwas_retrieved = len(result.fatwas)
    analysis.hadiths_retrieved = len(result.hadiths)
    analysis.books_retrieved = len(result.books)
    analysis.total_documents_retrieved = result.total_documents
    
    # Analyze each source
    for doc in all_docs:
        analysis.sources.append(analyze_source(doc))
    
    logger.info(f"Retrieved: {analysis.fatwas_retrieved} fatwas, "
                f"{analysis.hadiths_retrieved} hadiths, {analysis.books_retrieved} books")
    
    # Format context
    context_manager = ContextManager(ContextConfig(
        max_context_tokens=max_context_tokens,
        include_metadata=True,
    ))
    
    formatted_context = context_manager.format_context(all_docs, include_metadata=True)
    
    analysis.documents_used_in_context = len(formatted_context.documents_used)
    analysis.context_truncated = formatted_context.truncated
    analysis.estimated_context_tokens = formatted_context.total_tokens_estimate
    
    # Generate citations
    citation_generator = CitationGenerator(format_style="islamic_scholarly")
    citations = citation_generator.generate_citations(formatted_context.documents_used)
    
    for cit in citations:
        analysis.citations.append(CitationAnalysis(
            index=cit["index"],
            document_id=cit["document_id"],
            source_type=cit.get("source_type", "fatwa"),
            formatted=cit["formatted"],
            short=cit["short"],
            relevance_score=cit["relevance_score"],
        ))
    
    # Generate response if requested
    if generate:
        logger.info("Generating response...")
        
        try:
            # Try to get provider
            from rag_service.providers.factory import get_provider
            llm_provider = get_provider(provider)
            provider = llm_provider.config.provider_type.value
            logger.info(f"Using provider: {provider}")
            
            if llm_provider:
                from rag_service.core.generator import Generator, GenerationConfig
                from rag_service.providers.base import Message
                
                generator = Generator(provider=llm_provider)
                
                # Build prompt
                system_prompt = """أنت عالم إسلامي متخصص في الفقه والفتاوى. أجب على الأسئلة بناءً على المصادر المقدمة فقط.
استشهد بالمصادر باستخدام أرقام [1]، [2]، إلخ.
إذا لم تجد إجابة في المصادر، قل ذلك بوضوح.

You are an Islamic scholar specializing in fiqh and fatwas. Answer questions based only on the provided sources.
Cite sources using numbers [1], [2], etc.
If you cannot find an answer in the sources, say so clearly."""
                
                user_prompt = f"""السؤال / Question:
{query}

المصادر / Sources:
{formatted_context.text}

الجواب / Answer:"""
                
                messages = [
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=user_prompt),
                ]
                
                response = generator.generate(messages, GenerationConfig(
                    max_tokens=max_generation_tokens,
                    temperature=0.7,
                ))
                
                analysis.answer = response.content
                analysis.provider = response.provider
                analysis.model = response.model
                analysis.tokens_used = response.total_tokens
                
                # Build answer with citations
                references = "\n\n---\nالمراجع / References:\n"
                for cit in citations:
                    references += f"{cit['formatted']}\n"
                analysis.answer_with_citations = analysis.answer + references
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            analysis.answer = f"[Generation failed: {e}]"
    else:
        analysis.answer = "[Generation skipped - retrieval only mode]"
    
    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG pipeline and analyze results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python query_pipeline.py "ما حكم صلاة الجمعة؟"
  python query_pipeline.py "ما حكم الزكاة؟" --madhab hanafi
  python query_pipeline.py "ruling on fasting" --output results.json
  python query_pipeline.py "ما حكم الربا؟" --no-generate --top-k 10
        """
    )
    
    parser.add_argument("query", help="The question to ask")
    parser.add_argument("--madhab", "-m", choices=["hanafi", "maliki", "shafii", "hanbali"],
                       help="Filter books by madhab")
    parser.add_argument("--top-k", "-k", type=int, default=5,
                       help="Number of fatwas to retrieve (default: 5)")
    parser.add_argument("--top-k-hadiths", type=int, default=2,
                       help="Number of hadiths per fatwa (default: 2)")
    parser.add_argument("--top-k-books", type=int, default=3,
                       help="Number of book passages (default: 3)")
    parser.add_argument("--no-books", action="store_true",
                       help="Disable book retrieval")
    parser.add_argument("--no-hadiths", action="store_true",
                       help="Disable hadith retrieval")
    parser.add_argument("--no-generate", action="store_true",
                       help="Skip LLM generation (retrieval only)")
    parser.add_argument("--provider", "-p", choices=ProviderFactory.list_providers(),
                       help="LLM provider to use")
    parser.add_argument("--output", "-o", type=str,
                       help="Output file for JSON analysis")
    parser.add_argument("--output-txt", "-t", type=str,
                       help="Output file for detailed text report")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress detailed output")
    parser.add_argument("--max-context", type=int, default=4000,
                       help="Maximum context tokens (default: 4000)")
    parser.add_argument("--max-generation", type=int, default=2000,
                       help="Maximum generation tokens (default: 2000)")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    
    # Run pipeline
    analysis = run_pipeline(
        query=args.query,
        madhab=args.madhab,
        top_k_fatwas=args.top_k,
        top_k_hadiths=args.top_k_hadiths,
        top_k_books=args.top_k_books,
        retrieve_books=not args.no_books,
        retrieve_hadiths=not args.no_hadiths,
        generate=not args.no_generate,
        provider=args.provider,
        max_context_tokens=args.max_context,
        max_generation_tokens=args.max_generation,
    )
    
    # Print summary
    if not args.quiet:
        analysis.print_summary()
    
    # Save to JSON file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(analysis.to_json(), encoding="utf-8")
        print(f"\n✅ JSON analysis saved to: {output_path}")
    
    # Save to text file if requested
    if args.output_txt:
        txt_path = Path(args.output_txt)
        txt_path.write_text(analysis.to_text(), encoding="utf-8")
        print(f"\n✅ Text report saved to: {txt_path}")
    
    return analysis


if __name__ == "__main__":
    main()
