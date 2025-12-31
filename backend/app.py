"""
Islamic RAG Web Application - FastAPI Backend

Run with: python app.py
Then open: http://localhost:8000
"""

import sys
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, asdict

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from loguru import logger
import uvicorn

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rag_service.core.retriever import Retriever
from rag_service.core.linked_retrieval import LinkedRetriever
from rag_service.core.context_manager import ContextManager, ContextConfig
from rag_service.citation.citation_generator import CitationGenerator
from rag_service.providers.factory import get_provider
from rag_service.providers.base import Message
from rag_service.core.generator import Generator, GenerationConfig
from rag_service.utils.preprocessing import detect_language
from rag_service.translation.query_translator import translate_query_to_arabic


# ============================================================
# API Models
# ============================================================

class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str
    madhab: Optional[str] = None
    top_k: int = 5


class SourceInfo(BaseModel):
    """Information about a source document."""
    chunk_id: str  # Unique identifier for source linking
    type: str
    title: Optional[str] = None
    author: Optional[str] = None
    scholar: Optional[str] = None
    source_name: Optional[str] = None
    content_preview: str
    full_content: str  # Full content for popup display
    score: float


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str
    language: str
    sources: List[SourceInfo]
    total_sources: int
    translated_query: Optional[str] = None  # Show translated query if applicable


# ============================================================
# Application Setup
# ============================================================

app = FastAPI(
    title="Islamic RAG API",
    description="Retrieve and generate answers from Islamic texts using RAG",
    version="1.0.0"
)

# Ensure static directory exists
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ============================================================
# Global State (Lazy Initialization)
# ============================================================

class RAGState:
    """Holds initialized RAG components."""
    def __init__(self):
        self.initialized = False
        self.fatwa_retriever = None
        self.hadith_retriever = None
        self.book_retriever = None
        self.linked_retriever = None
        self.llm_provider = None
        self.generator = None
    
    def initialize(self):
        if self.initialized:
            return
        
        logger.info("Initializing RAG components...")
        
        # Initialize retrievers
        self.fatwa_retriever = Retriever(collection_name="fatwas")
        self.hadith_retriever = Retriever(collection_name="hadiths")
        self.book_retriever = Retriever(collection_name="books")
        
        self.fatwa_retriever.initialize()
        
        # Share models
        for retriever in [self.hadith_retriever, self.book_retriever]:
            retriever._embedding_model = self.fatwa_retriever._embedding_model
            retriever._qdrant_client = self.fatwa_retriever._qdrant_client
            retriever._reranker = self.fatwa_retriever._reranker
            retriever._initialized = True
        
        self.linked_retriever = LinkedRetriever(
            fatwa_retriever=self.fatwa_retriever,
            hadith_retriever=self.hadith_retriever,
            book_retriever=self.book_retriever,
        )
        
        # Initialize LLM
        try:
            self.llm_provider = get_provider()
            self.generator = Generator(provider=self.llm_provider)
            logger.info(f"LLM provider initialized: {self.llm_provider.name}")
        except Exception as e:
            logger.warning(f"LLM provider not available: {e}")
            self.llm_provider = None
            self.generator = None
        
        self.initialized = True
        logger.info("RAG components initialized successfully!")


rag_state = RAGState()


# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
async def serve_frontend():
    """Serve the main HTML page."""
    index_path = static_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)


@app.get("/api/health")
async def health_check():
    """Check API health status."""
    return {
        "status": "healthy",
        "rag_initialized": rag_state.initialized,
        "llm_available": rag_state.llm_provider is not None
    }


@app.post("/api/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the Islamic RAG system.
    
    Returns an answer generated from retrieved sources.
    """
    # Initialize on first request
    rag_state.initialize()
    
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    detected_lang = detect_language(query)
    logger.info(f"Processing query: {query[:50]}... (lang={detected_lang})")
    
    # Translate English queries to Arabic for better retrieval
    translated_query = None
    retrieval_query = query
    if detected_lang == "en" and rag_state.llm_provider:
        translated_query = translate_query_to_arabic(
            query=query,
            llm_provider=rag_state.llm_provider,
            detected_lang=detected_lang
        )
        if translated_query != query:
            retrieval_query = translated_query
            logger.info(f"Using translated query for retrieval: {retrieval_query[:50]}...")
    
    # Retrieve documents using Arabic query
    result = rag_state.linked_retriever.retrieve_with_links(
        query=retrieval_query,
        top_k_fatwas=request.top_k,
        top_k_hadiths_per_fatwa=2,
        top_k_books=3,
        madhab=request.madhab,
    )
    
    all_docs = result.all_documents
    logger.info(f"Retrieved {len(all_docs)} documents")

    
    # Build sources list with chunk_id for linking
    sources = []
    sources_for_verification = []  # For citation verification
    
    for i, doc in enumerate(all_docs):
        doc_type = doc.metadata.get("type", "fatwa")
        chunk_id = f"{doc_type}_{i}"  # Unique chunk ID for this response
        
        # Extract metadata
        title = doc.metadata.get("title", doc.metadata.get("question", ""))[:100]
        source_name = doc.metadata.get("source", doc.metadata.get("website", ""))
        author = doc.metadata.get("author", doc.metadata.get("scholar", ""))
        
        source_info = SourceInfo(
            chunk_id=chunk_id,
            type=doc_type,
            title=title if title else None,
            source_name=source_name if source_name else None,
            author=author if author and doc_type == "book" else None,
            scholar=author if author and doc_type == "fatwa" else None,
            content_preview=doc.content[:300] + "..." if len(doc.content) > 300 else doc.content,
            full_content=doc.content,  # Full content for popup
            score=doc.score,
        )
        sources.append(source_info)
        
        # For verification
        sources_for_verification.append({
            "chunk_id": chunk_id,
            "text": doc.content,
            "source_type": doc_type,
            "metadata": doc.metadata,
        })
    
    # Generate answer using scholarly prompts
    answer = ""
    if rag_state.generator and all_docs:
        from rag_service.utils.scholarly_prompts import SCHOLARLY_SYSTEM_PROMPT
        from rag_service.utils.fiqh_glossary import inject_glossary_definitions, build_glossary_footnotes
        from rag_service.utils.citation_verifier import (
            verify_answer_citations,
            convert_source_links_to_html,
            build_source_context_for_prompt,
        )
        
        # Build context with chunk_id for proper citations
        sources_context = build_source_context_for_prompt(sources_for_verification)
        
        user_prompt = f"""السؤال / Question:
{query}

المصادر المتاحة / Available Sources:
{sources_context}

⚠️ IMPORTANT: Use ONLY the chunk_id values above for citations!
Format: [source:chunk_id]description[/source]
Example: [source:fatwa_0]فتوى من إسلام ويب[/source]

أجب وفق الصيغة المحددة."""

        messages = [
            Message(role="system", content=SCHOLARLY_SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ]
        
        try:
            response = rag_state.generator.generate(messages, GenerationConfig(
                max_tokens=3000,
                temperature=0.5,
            ))
            answer = response.content
            
            # Post-processing: Remove thought block
            import re
            thought_match = re.search(r'<thought>.*?</thought>', answer, re.DOTALL | re.IGNORECASE)
            if thought_match:
                answer = answer[:thought_match.start()] + answer[thought_match.end():]
                answer = answer.strip()
            
            # Post-processing: Verify and fix hallucinated citations
            answer, hallucinations = verify_answer_citations(answer, sources_for_verification)
            if hallucinations:
                logger.warning(f"Fixed {len(hallucinations)} hallucinated citations")
            
            # Post-processing: Convert [source:ID]text[/source] to clickable HTML
            def replace_source_tag(match):
                source_id = match.group(1)
                display_text = match.group(2)
                return f'<a href="#" class="source-link" data-source-id="{source_id}" onclick="showSourceModal(\'{source_id}\'); return false;">{display_text}</a>'
            
            answer = re.sub(r'\[source:([^\]]+)\]([^\[]+)\[/source\]', replace_source_tag, answer)
            
            # Also handle old format [[source_id:ID|Text]]
            def replace_old_source_link(match):
                parts = match.group(1).split('|')
                source_id = parts[0].replace('source_id:', '')
                display_text = parts[1] if len(parts) > 1 else source_id
                return f'<a href="#" class="source-link" data-source-id="{source_id}" onclick="showSourceModal(\'{source_id}\'); return false;">{display_text}</a>'
            
            answer = re.sub(r'\[\[([^\]]+)\]\]', replace_old_source_link, answer)
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            answer = "عذراً، حدث خطأ في توليد الإجابة."
    elif not all_docs:
        answer = "لم يتم العثور على مصادر ذات صلة."
    else:
        answer = "خدمة التوليد غير متاحة حالياً."
    
    return QueryResponse(
        answer=answer,
        language=detected_lang,
        sources=sources,
        total_sources=len(sources),
        translated_query=translated_query if translated_query != query else None,
    )



# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":
    logger.info("Starting Islamic RAG Web Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
