
import asyncio
import sys
from pathlib import Path
from loguru import logger

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from rag_service.core.retriever import Retriever, RetrievalQuery
from rag_service.core.linked_retrieval import LinkedRetriever
from rag_service.core.generator import Generator

async def verify_fix():
    # 1. Initialize RAG components
    fatwa_retriever = Retriever(collection_name="fatwas")
    hadith_retriever = Retriever(collection_name="hadiths")
    book_retriever = Retriever(collection_name="books")
    
    fatwa_retriever.initialize()
    for retriever in [hadith_retriever, book_retriever]:
        retriever._embedding_model = fatwa_retriever._embedding_model
        retriever._qdrant_client = fatwa_retriever._qdrant_client
        retriever._reranker = fatwa_retriever._reranker
        retriever._initialized = True
        
    linked_retriever = LinkedRetriever(
        fatwa_retriever=fatwa_retriever,
        hadith_retriever=hadith_retriever,
        book_retriever=book_retriever,
    )
    
    generator = Generator()
    
    # 2. Query
    query = "ما الآيات القرآنية التي استُدل بها على وجوب صلاة الجماعة؟"
    logger.info(f"Querying: {query}")
    
    result = linked_retriever.retrieve_with_links(query=query, top_k_fatwas=5)
    all_docs = result.all_documents
    logger.info(f"Retrieved {len(all_docs)} documents")
    
    # 3. Verify Relevance
    logger.info("Starting relevance verification with updated logic...")
    verified_docs = await generator.verify_relevance(query, all_docs)
    
    logger.info(f"Verified {len(verified_docs)} out of {len(all_docs)} documents")
    
    if len(verified_docs) > 0:
        logger.info("SUCCESS: One or more documents passed verification!")
    else:
        logger.warning("STILL FAILED: No documents passed verification.")
        
    for i, doc in enumerate(all_docs):
        status = "PASSED" if doc in verified_docs else "REJECTED"
        logger.info(f"{i+1}. Doc {doc.id} ({doc.collection}): {status}")
        if status == "PASSED":
             logger.info(f"   Snippet: {doc.content[:150]}...")

if __name__ == "__main__":
    asyncio.run(verify_fix())
