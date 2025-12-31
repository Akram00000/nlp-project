"""
Linked Retrieval Example - Fetch fatwas with referenced hadiths.

Shows how fatwas automatically pull hadiths they mention.

Usage:
    python -m backend.rag_service.examples.linked_retrieval_example
"""

from backend.rag_service.core.linked_retrieval import LinkedRetriever
from backend.rag_service.core.rag_engine import RAGEngine, RAGConfig


def example_basic_linked_retrieval():
    """Basic example of linked retrieval."""
    
    print("=" * 60)
    print("Basic Linked Retrieval")
    print("=" * 60)
    
    # Create linked retriever
    retriever = LinkedRetriever(
        fatwa_collection="fatwas",
        hadith_collection="hadiths",
    )
    
    # Query about prayer
    query = "ما حكم صلاة الجماعة؟"
    
    print(f"\nQuery: {query}")
    print("\nRetrieving...")
    
    result = retriever.retrieve_with_links(
        query=query,
        top_k_fatwas=5,
        top_k_hadiths_per_fatwa=2,
    )
    
    print(f"\n✓ Retrieved:")
    print(f"  - {len(result.primary_docs)} fatwas")
    print(f"  - {len(result.linked_hadiths)} linked hadiths")
    print(f"  - {result.total_documents} total documents")
    
    # Show first fatwa
    if result.primary_docs:
        fatwa = result.primary_docs[0]
        print(f"\nTop Fatwa:")
        print(f"  Scholar: {fatwa.metadata.get('scholar', 'Unknown')}")
        print(f"  Score: {fatwa.score:.3f}")
        print(f"  Content: {fatwa.content[:200]}...")
    
    # Show linked hadiths
    if result.linked_hadiths:
        print(f"\nLinked Hadiths:")
        for i, hadith in enumerate(result.linked_hadiths[:2], 1):
            print(f"\n  {i}. Hadith:")
            print(f"     Source: {hadith.metadata.get('source', 'Unknown')}")
            print(f"     Narrator: {hadith.metadata.get('narrator', 'Unknown')}")
            print(f"     Content: {hadith.content[:150]}...")


def example_rag_with_linked_retrieval():
    """Use linked retrieval in RAG pipeline."""
    
    print("\n" + "=" * 60)
    print("RAG with Linked Retrieval")
    print("=" * 60)
    
    # Configure RAG with linked retrieval enabled
    config = RAGConfig(
        top_k=5,
        use_linked_retrieval=True,  # Enable hadith linking
        hadith_collection="hadiths",
        max_hadiths_per_fatwa=2,
        include_citations=True,
    )
    
    engine = RAGEngine(config=config, provider_name="lmstudio")
    
    print("\n✓ RAG Engine configured with linked retrieval")
    print(f"  - Will retrieve fatwas + referenced hadiths")
    print(f"  - Max {config.max_hadiths_per_fatwa} hadiths per fatwa")
    
    # Query (requires Qdrant + LLM)
    query = "ما حكم صلاة الجماعة؟"
    
    print(f"\nQuery: {query}")
    print("\nNote: Actual query requires:")
    print("  1. Qdrant running with indexed data")
    print("  2. LLM provider running (LM Studio/Gemini)")
    print("\nPipeline flow:")
    print("  1. Retrieve relevant fatwas")
    print("  2. Extract hadith references from fatwas")
    print("  3. Retrieve those hadiths from hadith collection")
    print("  4. Use both fatwas + hadiths as context")
    print("  5. Generate answer with citations")


def example_without_linked_retrieval():
    """Compare: RAG without linked retrieval."""
    
    print("\n" + "=" * 60)
    print("RAG WITHOUT Linked Retrieval (for comparison)")
    print("=" * 60)
    
    # Disable linked retrieval
    config = RAGConfig(
        top_k=5,
        use_linked_retrieval=False,  # Disabled
        include_citations=True,
    )
    
    engine = RAGEngine(config=config)
    
    print("\n✓ RAG Engine without linked retrieval")
    print("  - Only retrieves fatwas")
    print("  - Hadiths NOT fetched even if mentioned")
    print("  - Faster but less comprehensive")


def example_reference_extraction():
    """Show how references are extracted."""
    
    print("\n" + "=" * 60)
    print("Reference Extraction Demo")
    print("=" * 60)
    
    from backend.rag_service.data_ingestion.reference_extractor import ReferenceExtractor
    
    extractor = ReferenceExtractor()
    
    # Sample fatwa text mentioning hadith
    sample_text = """
    صلاة الجماعة واجبة على الرجال في أصح أقوال أهل العلم.
    
    قال رسول الله صلى الله عليه وسلم: "من سمع النداء فلم يأته فلا صلاة له إلا من عذر"
    رواه ابن ماجه وأبو داود.
    
    وقال تعالى: {وَارْكَعُوا مَعَ الرَّاكِعِينَ} [البقرة: 43]
    """
    
    refs = extractor.extract_all(sample_text)
    
    print("\nSample Fatwa Text:")
    print(sample_text)
    
    print("\n✓ Extracted References:")
    
    if refs['hadiths']:
        print(f"\n  Hadiths ({len(refs['hadiths'])}):")
        for h in refs['hadiths']:
            print(f"    - Query: {h['query']}")
            print(f"      Source: {h.get('source', 'N/A')}")
    
    if refs['ayahs']:
        print(f"\n  Quranic Verses ({len(refs['ayahs'])}):")
        for a in refs['ayahs']:
            print(f"    - {a['text']}")
            print(f"      [{a.get('surah', '?')}:{a.get('ayah', '?')}]")


def main():
    """Run all examples."""
    
    print("\n" + "#" * 60)
    print("#  Linked Retrieval Examples")
    print("#" * 60)
    
    try:
        # Show reference extraction
        example_reference_extraction()
        
        # Basic linked retrieval
        example_basic_linked_retrieval()
        
        # RAG with linked retrieval
        example_rag_with_linked_retrieval()
        
        # Comparison without linking
        example_without_linked_retrieval()
        
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        print("\nMake sure:")
        print("  1. Qdrant is running")
        print("  2. Collections 'fatwas' and 'hadiths' exist")
        print("  3. Data has been indexed")
    
    print("\n" + "#" * 60)
    print("#  Key Benefits of Linked Retrieval:")
    print("#")
    print("#  1. More comprehensive answers")
    print("#  2. Primary sources included (hadiths)")
    print("#  3. Better citations with hadith references")
    print("#  4. Transparent sourcing")
    print("#" * 60)


if __name__ == "__main__":
    main()
