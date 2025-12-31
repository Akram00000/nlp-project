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
    type: str
    title: Optional[str] = None
    author: Optional[str] = None
    scholar: Optional[str] = None
    content_preview: str
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

    
    # Build sources list
    sources = []
    for doc in all_docs:
        doc_type = doc.metadata.get("type", "fatwa")
        source_info = SourceInfo(
            type=doc_type,
            content_preview=doc.content[:300] + "..." if len(doc.content) > 300 else doc.content,
            score=doc.score,
        )
        
        if doc_type == "fatwa":
            source_info.scholar = doc.metadata.get("scholar")
        elif doc_type == "book":
            source_info.title = doc.metadata.get("title")
            source_info.author = doc.metadata.get("author")
        elif doc_type == "hadith":
            source_info.title = doc.metadata.get("source")
        
        sources.append(source_info)
    
    # Generate answer
    answer = ""
    if rag_state.generator and all_docs:
        context_manager = ContextManager(ContextConfig(max_context_tokens=4000))
        formatted_context = context_manager.format_context(all_docs, include_metadata=True)
        
        system_prompt = """أنت عالم إسلامي متخصص. أجب على الأسئلة بناءً على المصادر المقدمة فقط.
You are an Islamic scholar. Answer based only on the provided sources."""

        user_prompt = f"""السؤال / Question:
{query}

المصادر / Sources:
{formatted_context.text}

الجواب / Answer:"""

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]
        
        try:
            response = rag_state.generator.generate(messages, GenerationConfig(
                max_tokens=2000,
                temperature=0.7,
            ))
            answer = response.content
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            answer = f"عذراً، حدث خطأ في توليد الإجابة. / Sorry, generation failed: {str(e)}"
    elif not all_docs:
        answer = "لم يتم العثور على مصادر ذات صلة. / No relevant sources found."
    else:
        answer = "خدمة التوليد غير متاحة حالياً. / Generation service unavailable."
    
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
