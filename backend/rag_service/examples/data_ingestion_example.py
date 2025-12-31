"""
Data Ingestion Example - Index your fatwa and hadith data.

This shows how to load and index your JSON files into Qdrant.

Usage:
    python -m backend.rag_service.examples.data_ingestion_example
"""

from pathlib import Path
from backend.rag_service.data_ingestion import VectorIndexer


def example_index_fatwas():
    """Index fatwa files from data directory."""
    
    print("=" * 60)
    print("Indexing Fatwas")
    print("=" * 60)
    
    # Initialize indexer
    indexer = VectorIndexer()
    indexer.initialize(
        qdrant_host="localhost",
        qdrant_port=6333,
        embedding_model_name="Omartificial-Intelligence-Space/GATE-AraBert-v1",
    )
    
    # Path to your fatwa data
    data_path = Path("data")  # Adjust to your data directory
    
    # Index all JSONL files
    result = indexer.index_fatwas(
        data_path=data_path,
        collection_name="fatwas",
        pattern="*.jsonl",  # Will index 165kFatwa.jsonl, binbaz_fatwas.jsonl
        force_recreate=True,  # Set True to recreate collection
    )
    
    print(f"\n✓ Indexed {result['indexed']} fatwas")
    if 'fatwa_stats' in result:
        stats = result['fatwa_stats']
        print(f"  - {stats['unique_scholars']} scholars")
        print(f"  - {stats['unique_sources']} sources")
        print(f"  - {stats['reference_percentage']:.1f}% contain hadith/ayah references")


def example_index_hadiths():
    """Index hadith files (when available)."""
    
    print("\n" + "=" * 60)
    print("Indexing Hadiths")
    print("=" * 60)
    
    indexer = VectorIndexer()
    indexer.initialize()
    
    # Path to hadith data (adjust when you have hadith files)
    hadith_path = Path("data/hadiths")
    
    if not hadith_path.exists():
        print(f"\n⚠️  Hadith directory not found: {hadith_path}")
        print("   Create this directory and add hadith JSON/JSONL files")
        print("   Expected format:")
        print("   {")
        print('     "id": "bukhari_1234",')
        print('     "arabic": "حديث بالعربية",')
        print('     "narrator": "أبو هريرة",')
        print('     "source": "البخاري",')
        print('     "book": "كتاب الإيمان",')
        print('     "number": "1234"')
        print("   }")
        return
    
    result = indexer.index_hadiths(
        data_path=hadith_path,
        collection_name="hadiths",
        pattern="*.jsonl",
        force_recreate=False,
    )
    
    print(f"\n✓ Indexed {result['indexed']} hadiths")
    if 'hadith_stats' in result:
        stats = result['hadith_stats']
        print(f"  - {stats['unique_sources']} sources")
        print(f"  - {stats['unique_narrators']} narrators")


def example_check_references():
    """Check what hadith references are found in fatwas."""
    
    print("\n" + "=" * 60)
    print("Checking Hadith References in Fatwas")
    print("=" * 60)
    
    from backend.rag_service.data_ingestion import FatwaLoader
    from backend.rag_service.data_ingestion.reference_extractor import ReferenceExtractor
    
    # Load a sample of fatwas
    loader = FatwaLoader(extract_references=True)
    fatwas = list(loader.load_jsonl("data/165kFatwa.jsonl"))[:10]
    
    extractor = ReferenceExtractor()
    
    total_hadith_refs = 0
    total_ayah_refs = 0
    
    print(f"\nAnalyzing {len(fatwas)} fatwas...\n")
    
    for fatwa in fatwas:
        refs = extractor.extract_all(fatwa.content)
        
        if refs['hadiths'] or refs['ayahs']:
            print(f"Fatwa {fatwa.id}:")
            
            if refs['hadiths']:
                print(f"  ✓ {len(refs['hadiths'])} hadith reference(s)")
                total_hadith_refs += len(refs['hadiths'])
                # Show first reference
                if refs['hadiths']:
                    h = refs['hadiths'][0]
                    print(f"    Example: {h['query'][:80]}...")
            
            if refs['ayahs']:
                print(f"  ✓ {len(refs['ayahs'])} ayah reference(s)")
                total_ayah_refs += len(refs['ayahs'])
    
    print(f"\nTotal: {total_hadith_refs} hadith refs, {total_ayah_refs} ayah refs")


def example_using_cli():
    """Show CLI commands for indexing."""
    
    print("\n" + "=" * 60)
    print("CLI Usage Examples")
    print("=" * 60)
    
    print("""
You can also use the CLI for indexing:

# Index fatwas
python -m backend.rag_service.data_ingestion.cli index-fatwas data/

# Index hadiths
python -m backend.rag_service.data_ingestion.cli index-hadiths data/hadiths/

# Index everything
python -m backend.rag_service.data_ingestion.cli index-all data/

# Check references
python -m backend.rag_service.data_ingestion.cli check-references data/165kFatwa.jsonl --limit 20

# Force recreate collections
python -m backend.rag_service.data_ingestion.cli index-fatwas data/ --force

# Custom Qdrant host
python -m backend.rag_service.data_ingestion.cli index-fatwas data/ --host remote.example.com --port 6333
    """)


def main():
    """Run all examples."""
    
    print("\n" + "#" * 60)
    print("#  Data Ingestion Examples")
    print("#" * 60)
    
    try:
        # Check references first to see what we'll extract
        example_check_references()
        
        # Index fatwas
        example_index_fatwas()
        
        # Index hadiths (if available)
        example_index_hadiths()
        
        # Show CLI usage
        example_using_cli()
        
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        print("\nMake sure:")
        print("  1. Qdrant is running (docker run -p 6333:6333 qdrant/qdrant)")
        print("  2. Dependencies installed (pip install -r requirements.txt)")
        print("  3. Data files exist in data/ directory")
    
    print("\n" + "#" * 60)
    print("#  Next Steps:")
    print("#  1. Verify collections in Qdrant")
    print("#  2. Test retrieval with RAG engine")
    print("#  3. Check linked retrieval (fatwa → hadith)")
    print("#" * 60)


if __name__ == "__main__":
    main()
