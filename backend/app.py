"""
Islamic RAG Web Application - FastAPI Backend

Run with: python app.py
Then open: http://localhost:8000
"""

import sys
import json
import asyncio
import re
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, asdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
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
    history: Optional[List[dict]] = None  # Chat history for context


class ChatMessage(BaseModel):
    """A single message in chat history."""
    role: str  # 'user' or 'assistant'
    content: str


class TranslateRequest(BaseModel):
    """Request model for translation."""
    text: str
    sources: Optional[List[dict]] = None  # Source metadata for smart translation


class TranslateResponse(BaseModel):
    """Response model for translation."""
    original_text: str
    translated_text: str
    provider: Optional[str] = None
    cached: bool = False


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
            self.llm_provider = get_provider(provider_name="gemini")
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
# Streaming Endpoint for Non-blocking Responses
# ============================================================

@app.post("/api/query/stream")
async def query_rag_stream(request: QueryRequest, http_request: Request):
    """
    Stream the RAG response using Server-Sent Events (SSE).
    
    Sends events:
    - sources: JSON with retrieved sources
    - token: Each token of the generated response
    - done: Final message with complete response
    - error: If an error occurs
    """
    # Initialize on first request
    rag_state.initialize()
    
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    async def generate_stream():
        try:
            # Status: Starting
            yield f"data: {json.dumps({'type': 'status', 'status': 'analyzing', 'message': 'جاري تحليل السؤال...'})}\n\n"
            
            detected_lang = detect_language(query)
            logger.info(f"[STREAM] Processing query: {query[:50]}... (lang={detected_lang})")
            
            # Translate English queries to Arabic for better retrieval
            translated_query = None
            retrieval_query = query
            if detected_lang == "en" and rag_state.llm_provider:
                yield f"data: {json.dumps({'type': 'status', 'status': 'translating', 'message': 'جاري ترجمة السؤال...'})}\n\n"
                translated_query = translate_query_to_arabic(
                    query=query,
                    llm_provider=rag_state.llm_provider,
                    detected_lang=detected_lang
                )
                if translated_query != query:
                    retrieval_query = translated_query
                    logger.info(f"[STREAM] Using translated query: {retrieval_query[:50]}...")
            
            # Send language detection event
            yield f"data: {json.dumps({'type': 'meta', 'language': detected_lang, 'translated_query': translated_query})}\n\n"
            
            # Status: Retrieving
            yield f"data: {json.dumps({'type': 'status', 'status': 'retrieving', 'message': 'جاري البحث في المصادر...'})}\n\n"
            
            # Retrieve documents
            result = rag_state.linked_retriever.retrieve_with_links(
                query=retrieval_query,
                top_k_fatwas=request.top_k,
                top_k_hadiths_per_fatwa=2,
                top_k_books=3,
                madhab=request.madhab,
            )
            
            all_docs = result.all_documents
            logger.info(f"[STREAM] Retrieved {len(all_docs)} documents")
            
            # Build sources list
            sources = []
            sources_for_verification = []
            
            for i, doc in enumerate(all_docs):
                doc_type = doc.metadata.get("type", "fatwa")
                chunk_id = f"{doc_type}_{i}"
                
                title = doc.metadata.get("title", doc.metadata.get("question", ""))[:100]
                source_name = doc.metadata.get("source", doc.metadata.get("website", ""))
                author = doc.metadata.get("author", doc.metadata.get("scholar", ""))
                
                source_info = {
                    "chunk_id": chunk_id,
                    "type": doc_type,
                    "title": title if title else None,
                    "source_name": source_name if source_name else None,
                    "author": author if author and doc_type == "book" else None,
                    "scholar": author if author and doc_type == "fatwa" else None,
                    "content_preview": doc.content[:300] + "..." if len(doc.content) > 300 else doc.content,
                    "full_content": doc.content,
                    "score": doc.score,
                }
                sources.append(source_info)
                
                sources_for_verification.append({
                    "chunk_id": chunk_id,
                    "text": doc.content,
                    "source_type": doc_type,
                    "metadata": doc.metadata,
                })
            
            # Send sources immediately
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'total': len(sources)})}\n\n"
            
            # Generate answer with streaming
            if rag_state.generator and all_docs:
                # Status: Generating
                yield f"data: {json.dumps({'type': 'status', 'status': 'generating', 'message': 'جاري توليد الإجابة...'})}\n\n"
                
                from rag_service.utils.scholarly_prompts import SCHOLARLY_SYSTEM_PROMPT
                from rag_service.utils.citation_verifier import build_source_context_for_prompt
                
                # Build context and history
                sources_context = build_source_context_for_prompt(sources_for_verification)
                
                # Build messages including chat history
                messages = [Message(role="system", content=SCHOLARLY_SYSTEM_PROMPT)]
                
                # Add chat history (limited to last 10 messages for context window)
                if request.history:
                    history_limit = 10
                    recent_history = request.history[-history_limit:] if len(request.history) > history_limit else request.history
                    for msg in recent_history:
                        messages.append(Message(role=msg.get('role', 'user'), content=msg.get('content', '')))
                
                user_prompt = f"""السؤال / Question:
{query}

المصادر المتاحة / Available Sources:
{sources_context}

⚠️ IMPORTANT: Use ONLY the chunk_id values above for citations!
Format: [source:chunk_id]description[/source]
Example: [source:fatwa_0]فتوى من إسلام ويب[/source]

أجب وفق الصيغة المحددة."""

                messages.append(Message(role="user", content=user_prompt))
                
                # Stream tokens
                full_response = ""
                try:
                    async for token in rag_state.generator.stream(messages, GenerationConfig(
                        max_tokens=3000,
                        temperature=0.5,
                    )):
                        # Check if client disconnected
                        if await http_request.is_disconnected():
                            logger.info("[STREAM] Client disconnected, stopping generation")
                            return
                        
                        full_response += token
                        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                        await asyncio.sleep(0)  # Allow other tasks to run
                    
                    # Post-process the complete response
                    answer = full_response
                    
                    # Remove thought block
                    thought_match = re.search(r'<thought>.*?</thought>', answer, re.DOTALL | re.IGNORECASE)
                    if thought_match:
                        answer = answer[:thought_match.start()] + answer[thought_match.end():]
                        answer = answer.strip()
                    
                    # Verify citations
                    from rag_service.utils.citation_verifier import verify_answer_citations
                    answer, hallucinations = verify_answer_citations(answer, sources_for_verification)
                    if hallucinations:
                        logger.warning(f"[STREAM] Fixed {len(hallucinations)} hallucinated citations")
                    
                    # Convert source tags to HTML
                    def replace_source_tag(match):
                        source_id = match.group(1)
                        display_text = match.group(2)
                        return f'<a href="#" class="source-link" data-source-id="{source_id}" onclick="showSourceModal(\'{source_id}\'); return false;">{display_text}</a>'
                    
                    answer = re.sub(r'\[source:([^\]]+)\]([^\[]+)\[/source\]', replace_source_tag, answer)
                    
                    # Handle old format
                    def replace_old_source_link(match):
                        parts = match.group(1).split('|')
                        source_id = parts[0].replace('source_id:', '')
                        display_text = parts[1] if len(parts) > 1 else source_id
                        return f'<a href="#" class="source-link" data-source-id="{source_id}" onclick="showSourceModal(\'{source_id}\'); return false;">{display_text}</a>'
                    
                    answer = re.sub(r'\[\[([^\]]+)\]\]', replace_old_source_link, answer)
                    
                    # Send done event with final processed answer
                    yield f"data: {json.dumps({'type': 'done', 'answer': answer})}\n\n"
                    
                except Exception as e:
                    logger.error(f"[STREAM] Generation error: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Generation failed'})}\n\n"
                    
            elif not all_docs:
                yield f"data: {json.dumps({'type': 'done', 'answer': 'لم يتم العثور على مصادر ذات صلة.'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'done', 'answer': 'خدمة التوليد غير متاحة حالياً.'})}\n\n"
                
        except Exception as e:
            logger.error(f"[STREAM] Error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )



# ============================================================
# Run Server
# ============================================================

# Translation provider (lazy-loaded)
_translation_provider = None


def get_translation_provider():
    """Get or create translation provider, reusing RAG's LLM provider if available."""
    global _translation_provider
    if _translation_provider is None:
        try:
            from rag_service.translation import TranslationProviderFactory
            
            # First, try to reuse the existing RAG LLM provider (already initialized and working)
            if rag_state.llm_provider:
                logger.info("Creating translation provider from existing RAG LLM provider")
                _translation_provider = TranslationProviderFactory.from_llm_provider(
                    rag_state.llm_provider
                )
            else:
                # Fallback: create a new provider
                _translation_provider = TranslationProviderFactory.create_with_fallback()
            
            if _translation_provider:
                logger.info(f"Translation provider initialized: {_translation_provider.llm_provider.name}")
        except Exception as e:
            logger.warning(f"Could not initialize translation provider: {e}")
    return _translation_provider


@app.post("/api/translate", response_model=TranslateResponse)
async def translate_text(request: TranslateRequest):
    """
    Translate Arabic text to English.
    
    Uses smart translation that:
    - Preserves existing English hadith translations when available
    - Keeps Quranic text in Arabic with translation
    - Uses LLM for remaining content
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    logger.info(f"[TRANSLATE] Translating text ({len(text)} chars)")
    
    try:
        from rag_service.translation import SmartResponseTranslator, Language
        
        # Get or create translator
        provider = get_translation_provider()
        
        if not provider:
            raise HTTPException(
                status_code=503, 
                detail="Translation service unavailable - no LLM provider configured"
            )
        
        translator = SmartResponseTranslator(provider)
        
        # Perform translation
        result = translator.translate_response(
            text=text,
            sources=request.sources,
        )
        
        return TranslateResponse(
            original_text=text,
            translated_text=result.translated_text,
            provider=result.metadata.get("provider"),
            cached=False,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[TRANSLATE] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


if __name__ == "__main__":
    logger.info("Starting Islamic RAG Web Server...")
    uvicorn.run(app, host="0.0.0.0", port=8333)
