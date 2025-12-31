
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from rag_service.utils.metadata_enricher import MetadataEnricher
from rag_service.core.context_manager import ContextManager
from rag_service.core.retriever import RetrievalResult

def test_hadith_auth():
    enricher = MetadataEnricher()
    cm = ContextManager()
    
    # Test 1: Bukhari (Explicitly Authentic)
    doc1 = RetrievalResult(
        id="doc1",
        content="Narrated Ibn Umar: Prayer in congregation...",
        score=0.9,
        metadata={"type": "hadith", "source": "bukhari"}
    )
    doc1.enriched_metadata = enricher.enrich(doc1.content, doc1.metadata, doc1.id)
    print(f"Doc1 Auth: {doc1.enriched_metadata.authenticity} (Expected: sahih_agreed)")
    
    # Test 2: Tirmidhi (Inferred Hasan)
    doc2 = RetrievalResult(
        id="doc2",
        content="Narrated Abu Huraira...",
        score=0.8,
        metadata={"type": "hadith", "source": "tirmidhi"}
    )
    doc2.enriched_metadata = enricher.enrich(doc2.content, doc2.metadata, doc2.id)
    print(f"Doc2 Auth: {doc2.enriched_metadata.authenticity} (Expected: hasan)")
    
    # Test 3: Formatting
    formatted = cm._format_document_enriched(doc1, 1, True)
    if "Authenticity: sahih_agreed" in formatted:
        print("Formatting SUCCESS: Authenticity found in context.")
    else:
        print(f"Formatting FAILED. Got: \n{formatted}")

if __name__ == "__main__":
    test_hadith_auth()
