"""
Complete Workflow Example - From Data to RAG Response

This shows the complete pipeline:
1. Load JSON files
2. Extract references
3. Index to Qdrant
4. Retrieve with linking
5. Generate answer with citations
"""

from pathlib import Path
from backend.rag_service.data_ingestion import VectorIndexer, FatwaLoader, ReferenceExtractor
from backend.rag_service.core.rag_engine import RAGEngine, RAGConfig
from backend.rag_service.core.linked_retrieval import LinkedRetriever


def step1_analyze_data():
    """Step 1: Analyze your fatwa data before indexing."""
    
    print("STEP 1: Analyzing Data")
    print("=" * 60)
    
    # Load sample fatwas
    loader = FatwaLoader(extract_references=True)
    fatwas = list(loader.load_jsonl("data/165kFatwa.jsonl"))[:100]
    
    # Get statistics
    stats = loader.get_statistics(fatwas)
    
    print(f"\n✓ Sample Analysis (first 100 fatwas):")
    print(f"  Total: {stats['total_fatwas']}")
    print(f"  Scholars: {stats['unique_scholars']}")
    print(f"  Sources: {stats['unique_sources']}")
    print(f"  With References: {stats['with_references']} ({stats['reference_percentage']:.1f}%)")
    
    # Show top scholars
    print(f"\n  Top Scholars:")
    for scholar, count in stats['top_scholars'][:3]:
        print(f"    - {scholar}: {count} fatwas")
    
    # Show hadith extraction example
    extractor = ReferenceExtractor()
    for fatwa in fatwas[:5]:
        if extractor.has_hadith_reference(fatwa.content):
            refs = extractor.extract_hadiths(fatwa.content)
            print(f"\n  Example Hadith Extraction:")
            print(f"    Fatwa ID: {fatwa.id}")
            print(f"    Found: {len(refs)} hadith reference(s)")
            if refs:
                print(f"    Query: {refs[0]['query'][:80]}...")
            break


def step2_index_data():
    """Step 2: Index data into Qdrant."""
    
    print("\n\nSTEP 2: Indexing Data")
    print("=" * 60)
    
    # Initialize indexer
    indexer = VectorIndexer(batch_size=32)
    indexer.initialize(
        qdrant_host="localhost",
        qdrant_port=6333,
    )
    
    print("\n✓ Indexer initialized")
    
    # Index fatwas
    print("\n  Indexing fatwas...")
    fatwa_result = indexer.index_fatwas(
        data_path=Path("data"),
        collection_name="fatwas",
        pattern="*.jsonl",
        force_recreate=False,
    )
    
    print(f"  ✓ Indexed {fatwa_result['indexed']} fatwas")
    
    # Index hadiths (if available)
    hadith_path = Path("data/hadiths")
    if hadith_path.exists():
        print("\n  Indexing hadiths...")
        hadith_result = indexer.index_hadiths(
            data_path=hadith_path,
            collection_name="hadiths",
            pattern="*.jsonl",
        )
        print(f"  ✓ Indexed {hadith_result['indexed']} hadiths")
    else:
        print("\n  ⚠ Hadith directory not found - skipping")
        print(f"    Create {hadith_path} and add hadith files to enable linking")


def step3_test_retrieval():
    """Step 3: Test retrieval with linking."""
    
    print("\n\nSTEP 3: Testing Retrieval")
    print("=" * 60)
    
    # Create linked retriever
    retriever = LinkedRetriever(
        fatwa_collection="fatwas",
        hadith_collection="hadiths",
    )
    
    # Test query
    query = "ما حكم صلاة الجماعة في المسجد؟"
    print(f"\nQuery: {query}")
    
    result = retriever.retrieve_with_links(
        query=query,
        top_k_fatwas=5,
        top_k_hadiths_per_fatwa=2,
    )
    
    print(f"\n✓ Retrieved:")
    print(f"  Primary (Fatwas): {len(result.primary_docs)}")
    print(f"  Linked (Hadiths): {len(result.linked_hadiths)}")
    print(f"  Total: {result.total_documents}")
    
    # Show top result
    if result.primary_docs:
        top = result.primary_docs[0]
        print(f"\n  Top Fatwa:")
        print(f"    Scholar: {top.metadata.get('scholar', 'Unknown')}")
        print(f"    Source: {top.metadata.get('source', 'Unknown')}")
        print(f"    Score: {top.score:.3f}")
        print(f"    Content: {top.content[:150]}...")
    
    # Show linked hadiths
    if result.linked_hadiths:
        print(f"\n  Linked Hadiths:")
        for i, hadith in enumerate(result.linked_hadiths[:2], 1):
            print(f"\n    {i}. {hadith.metadata.get('source', 'Unknown')} - {hadith.metadata.get('narrator', 'Unknown')}")
            print(f"       {hadith.content[:100]}...")


def step4_generate_response():
    """Step 4: Generate complete RAG response."""
    
    print("\n\nSTEP 4: Generating RAG Response")
    print("=" * 60)
    
    # Configure RAG
    config = RAGConfig(
        top_k=5,
        use_linked_retrieval=True,
        hadith_collection="hadiths",
        max_hadiths_per_fatwa=2,
        include_citations=True,
        citation_format="islamic_scholarly",
        max_response_tokens=1000,
        temperature=0.7,
    )
    
    print("\n✓ RAG Engine Configuration:")
    print(f"  Provider: LM Studio (local)")
    print(f"  Top-K: {config.top_k}")
    print(f"  Linked Retrieval: {config.use_linked_retrieval}")
    print(f"  Citations: {config.include_citations}")
    
    # Create engine
    engine = RAGEngine(config=config, provider_name="lmstudio")
    
    # Query
    query = "ما حكم صلاة الجماعة؟"
    print(f"\nQuery: {query}")
    
    print("\nNote: Actual response generation requires:")
    print("  1. Qdrant running with indexed data ✓")
    print("  2. LLM provider running (LM Studio on port 1234)")
    print("\nIf LM Studio is running, uncomment below to test:")
    print("""
    # response = engine.query(query)
    # print(f"\\nAnswer:\\n{response.answer}")
    # print(f"\\nCitations: {len(response.citations)}")
    # print(f"Sources: {len(response.sources)}")
    # 
    # # Full response with citations
    # print(f"\\n{response.answer_with_citations}")
    """)


def step5_integration():
    """Step 5: Integration with other services."""
    
    print("\n\nSTEP 5: Service Integration")
    print("=" * 60)
    
    print("""
✓ RAG Service is ready for integration:

1. API Gateway (Youcef):
   - Endpoint: POST /api/v1/query
   - Request: {"question": "...", "language": "ar"}
   - Response: RAGResponse with answer, citations, sources
   
   from backend.rag_service.core.rag_engine import RAGEngine, RAGConfig
   
   engine = RAGEngine(config=RAGConfig(use_linked_retrieval=True))
   response = engine.query(user_question)
   return {
       "answer": response.answer,
       "citations": response.citations,
       "sources": [s.to_dict() for s in response.sources]
   }

2. Translation Service (Akram):
   - Integrate NLLB-200 provider
   - See: translation/translation_interface.py
   - Automatic query/response translation

3. WSD Service (Iyed):
   - Register WSD hook
   - See: translation/integration.py
   - Disambiguates Arabic queries

4. Frontend (Jasmine):
   - Display answer with inline citations [1], [2], etc.
   - Show references section
   - Highlight hadith sources differently
   - Streaming support available

Example API Response:
{
  "answer": "صلاة الجماعة واجبة... [1][2]",
  "citations": [
    {
      "id": 1,
      "scholar": "الشيخ ابن باز",
      "source": "فتاوى نور على الدرب",
      "type": "fatwa",
      "formatted": "[1] الشيخ ابن باز..."
    },
    {
      "id": 2,
      "narrator": "أبو هريرة",
      "source": "البخاري",
      "type": "hadith",
      "formatted": "[2] رواه البخاري عن أبي هريرة..."
    }
  ],
  "metadata": {
    "primary_sources": 5,
    "linked_hadiths": 3,
    "total_documents": 8
  }
}
    """)


def main():
    """Run complete workflow."""
    
    print("\n" + "#" * 60)
    print("#  Complete RAG Pipeline Workflow")
    print("#  From JSON Files to Full RAG Response")
    print("#" * 60)
    
    try:
        step1_analyze_data()
        step2_index_data()
        step3_test_retrieval()
        step4_generate_response()
        step5_integration()
        
    except Exception as e:
        print(f"\n⚠️ Error: {e}")
        print("\nChecklist:")
        print("  □ Qdrant running (docker run -p 6333:6333 qdrant/qdrant)")
        print("  □ Data files exist in data/ directory")
        print("  □ Dependencies installed (pip install -r requirements.txt)")
        print("  □ LM Studio running on port 1234 (for generation)")
    
    print("\n" + "#" * 60)
    print("#  Workflow Complete!")
    print("#  Next: Test with your actual queries")
    print("#" * 60)


if __name__ == "__main__":
    main()
