
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from rag_service.core.retriever import Retriever, RetrievalQuery
from loguru import logger

def test_hybrid_search():
    logger.info("Initializing Retriever...")
    retriever = Retriever(collection_name="hadiths")
    retriever.initialize()
    
    query_text = "صلاة الجماعة" # Congregational prayer
    
    logger.info(f"--- Testing Pure Vector Search (Alpha=1.0) ---")
    results_vector = retriever.retrieve(
        RetrievalQuery(text=query_text, top_k=5, alpha=1.0, use_mmr=False)
    )
    for r in results_vector:
        print(f"[{r.score:.4f}] {r.content[:100]}...")
        
    logger.info(f"--- Testing Hybrid Search (Alpha=0.5) ---")
    results_hybrid = retriever.retrieve(
        RetrievalQuery(text=query_text, top_k=5, alpha=0.5, use_mmr=False)
    )
    for r in results_hybrid:
        print(f"[{r.score:.4f}] {r.content[:100]}...")
        
    logger.info(f"--- Testing MMR (Diversity) ---")
    results_mmr = retriever.retrieve(
        RetrievalQuery(text=query_text, top_k=5, alpha=0.5, use_mmr=True)
    )
    for r in results_mmr:
        print(f"[{r.score:.4f}] {r.content[:100]}...")

if __name__ == "__main__":
    test_hybrid_search()
