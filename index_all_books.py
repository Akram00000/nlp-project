#!/usr/bin/env python3
"""
Index all books from all madhabs into Qdrant.
"""
import sys
from pathlib import Path

sys.path.insert(0, 'backend')

from rag_service.data_ingestion import VectorIndexer

def main():
    data_path = Path('data/OCR_google_documentAI_pretrained-ocr-v2.1-2024-08-07')
    
    if not data_path.exists():
        print(f"❌ Data path not found: {data_path}")
        return
    
    print(f"📂 Data path: {data_path}")
    print(f"📚 Indexing ALL books from all madhabs...")
    print()
    
    # Initialize indexer
    indexer = VectorIndexer(batch_size=32)
    print("🔧 Initializing indexer...")
    indexer.initialize(
        qdrant_host='localhost',
        qdrant_port=6333,
        embedding_model_name='Omartificial-Intelligence-Space/GATE-AraBert-v1'
    )
    
    print("⚙️  Starting book indexing (this will take several minutes)...")
    print()
    
    # Index all books
    result = indexer.index_books(
        data_path=data_path,
        collection_name='books',
        madhab_filter=None,  # Index ALL madhabs
        preprocess=True,     # Apply OCR cleanup
        force_recreate=True  # Start fresh
    )
    
    # Print results
    print()
    print("="*60)
    print("✅ INDEXING COMPLETE")
    print("="*60)
    print(f"Total chunks processed: {result['total']}")
    print(f"Successfully indexed: {result['indexed']}")
    print(f"Failed: {result['failed']}")
    
    if 'book_stats' in result:
        stats = result['book_stats']
        print(f"\n📊 Book Statistics:")
        print(f"  Unique books: {stats.get('unique_books', 'N/A')}")
        print(f"  Madhabs: {stats.get('madhabs', 'N/A')}")
        print(f"  Average chunk size: {stats.get('avg_chunk_size', 'N/A')} chars")
        print(f"  Average tokens: {stats.get('avg_tokens_estimate', 'N/A')}")
    
    print("\n✨ Books collection is ready for queries!")

if __name__ == '__main__':
    main()
