"""
Book Ingestion Example - Load and index Arabic Fiqh books.

This example demonstrates how to:
1. Load OCR book files from the data directory
2. Apply optional preprocessing
3. Index books into a separate Qdrant collection
4. Query books with madhab filtering
"""

from pathlib import Path
from loguru import logger

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_ingestion import BookLoader, VectorIndexer
from core.retriever import Retriever


def load_and_preview_books():
    """Load books and preview chunks without indexing."""
    
    # Path to OCR book data
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "OCR_google_documentAI_pretrained-ocr-v2.1-2024-08-07"
    
    if not data_path.exists():
        logger.error(f"Data path not found: {data_path}")
        return
    
    # Initialize loader
    loader = BookLoader(
        target_chunk_tokens=1000,  # ~1000 tokens per chunk
        chunk_overlap_tokens=100,  # ~100 token overlap
        preprocess=False,  # Set to True for OCR cleanup
        extract_chapters=True,  # Extract chapter/section info
    )
    
    # Load only Hanafi books for preview
    logger.info("Loading Hanafi books for preview...")
    documents = []
    
    for doc in loader.load_directory(data_path, madhab_filter="hanafi"):
        documents.append(doc)
        
        # Preview first 3 documents
        if len(documents) <= 3:
            print(f"\n{'='*60}")
            print(f"Document ID: {doc.id}")
            print(f"Book: {doc.metadata.get('title')}")
            print(f"Author: {doc.metadata.get('author')}")
            print(f"Madhab: {doc.metadata.get('madhab')}")
            print(f"Volume: {doc.metadata.get('volume')}")
            print(f"Pages: {doc.metadata.get('page_start')}-{doc.metadata.get('page_end')}")
            print(f"Chapter: {doc.metadata.get('chapter')}")
            print(f"Content Preview: {doc.content[:300]}...")
        
        # Limit for preview
        if len(documents) >= 50:
            break
    
    # Show statistics
    stats = loader.get_statistics(documents)
    print(f"\n{'='*60}")
    print("Statistics:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Unique books: {stats['unique_books']}")
    print(f"  Madhabs: {stats['madhabs']}")
    print(f"  Avg chunk size: {stats['avg_chunk_size']} chars")
    print(f"  Avg tokens (estimate): {stats['avg_tokens_estimate']}")
    
    return documents


def index_books():
    """Index books into Qdrant."""
    
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "OCR_google_documentAI_pretrained-ocr-v2.1-2024-08-07"
    
    if not data_path.exists():
        logger.error(f"Data path not found: {data_path}")
        return
    
    # Initialize indexer
    indexer = VectorIndexer(batch_size=16)  # Smaller batch for memory efficiency
    indexer.initialize(
        qdrant_host="localhost",
        qdrant_port=6333,
        embedding_model_name="Omartificial-Intelligence-Space/GATE-AraBert-v1",
    )
    
    # Index all books with preprocessing
    logger.info("Indexing books with OCR preprocessing...")
    result = indexer.index_books(
        data_path=data_path,
        collection_name="books",
        madhab_filter=None,  # Index all madhabs
        preprocess=True,  # Apply OCR cleanup
        force_recreate=True,  # Start fresh
    )
    
    print(f"\nIndexing Results:")
    print(f"  Total: {result['total']}")
    print(f"  Indexed: {result['indexed']}")
    print(f"  Failed: {result['failed']}")
    if 'book_stats' in result:
        print(f"  Books: {result['book_stats']['unique_books']}")
        print(f"  Madhabs: {result['book_stats']['madhabs']}")
    
    return result


def search_books():
    """Search books with madhab filtering."""
    
    # Initialize retriever
    retriever = Retriever(collection_name="books")
    retriever.initialize(
        host="localhost",
        port=6333,
        embedding_model_name="Omartificial-Intelligence-Space/GATE-AraBert-v1",
    )
    
    # Search all books
    print("\n" + "="*60)
    print("Search: ما حكم الصلاة على الميت")
    print("="*60)
    
    results = retriever.retrieve(
        query="ما حكم الصلاة على الميت",
        top_k=3,
        similarity_threshold=0.15,
    )
    
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] Score: {doc.score:.4f}")
        print(f"    Book: {doc.metadata.get('title')}")
        print(f"    Author: {doc.metadata.get('author')}")
        print(f"    Madhab: {doc.metadata.get('madhab')}")
        print(f"    Pages: {doc.metadata.get('page_start')}-{doc.metadata.get('page_end')}")
        print(f"    Content: {doc.content[:200]}...")
    
    # Search with madhab filter
    print("\n" + "="*60)
    print("Search (Hanafi only): ما حكم الصلاة على الميت")
    print("="*60)
    
    results = retriever.retrieve(
        query="ما حكم الصلاة على الميت",
        top_k=3,
        similarity_threshold=0.15,
        madhab="hanafi",
    )
    
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] Score: {doc.score:.4f}")
        print(f"    Book: {doc.metadata.get('title')}")
        print(f"    Madhab: {doc.metadata.get('madhab')}")
        print(f"    Content: {doc.content[:200]}...")


def search_across_collections():
    """Search across fatwas and books together."""
    
    from core.linked_retrieval import LinkedRetriever
    
    # Initialize linked retriever
    linked = LinkedRetriever(
        fatwa_collection="fatwas",
        hadith_collection="hadiths",
        book_collection="books",
    )
    
    # Search with book integration
    print("\n" + "="*60)
    print("Linked Search: ما حكم صيام يوم عاشوراء")
    print("="*60)
    
    result = linked.retrieve_with_links(
        query="ما حكم صيام يوم عاشوراء",
        top_k_fatwas=3,
        top_k_hadiths_per_fatwa=2,
        top_k_books=2,
        retrieve_books=True,
        madhab="hanafi",  # Filter books by madhab
    )
    
    print(f"\nTotal documents: {result.total_documents}")
    print(f"  - Fatwas: {len(result.fatwas)}")
    print(f"  - Hadiths: {len(result.hadiths)}")
    print(f"  - Books: {len(result.books)}")
    
    print("\nFatwas:")
    for doc in result.fatwas[:2]:
        print(f"  - {doc.title[:60]}... (score: {doc.score:.4f})")
    
    print("\nBooks:")
    for doc in result.books:
        print(f"  - {doc.metadata.get('title')} (ص{doc.page_start}-{doc.page_end})")
        print(f"    {doc.content[:150]}...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Book Ingestion Example")
    parser.add_argument("--action", choices=["preview", "index", "search", "linked"], 
                       default="preview", help="Action to perform")
    
    args = parser.parse_args()
    
    if args.action == "preview":
        load_and_preview_books()
    elif args.action == "index":
        index_books()
    elif args.action == "search":
        search_books()
    elif args.action == "linked":
        search_across_collections()
