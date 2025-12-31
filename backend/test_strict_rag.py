
import re
from rag_service.utils.fact_checker import FactChecker
from rag_service.core.retriever import RetrievalResult
from rag_service.utils.metadata_enricher import EnrichedMetadata

def test_strict_validation():
    checker = FactChecker()
    
    # Mock Docs
    doc1 = RetrievalResult(id="doc1", content="Text 1", metadata={"type":"fatwa"}, score=1.0)
    doc1.enriched_metadata = EnrichedMetadata(doc_id="doc1", citation_id="fatwa_1", source_type="fatwa")
    
    docs = [doc1]
    
    # Case 1: Valid Answer
    print("Test 1: Valid Answer")
    ans_valid = "According to [fatwa_1]: ruling is X."
    res = checker.check_response(ans_valid, docs)
    print(f"Valid? {res['valid']} (Expected: True)")
    
    # Case 2: Invalid Citation
    print("\nTest 2: Invalid Citation")
    ans_invalid = "According to [fatwa_99]: ruling is Y."
    res = checker.check_response(ans_invalid, docs)
    print(f"Valid? {res['valid']} (Expected: False)")
    print(f"Issues: {res['issues']}")
    
    # Case 3: Hallucinated Quran
    print("\nTest 3: Hallucinated Quran")
    # doc1 has no verse ref
    ans_quran = "As stated in Surah Baqarah:255..."
    res = checker.check_response(ans_quran, docs)
    print(f"Valid? {res['valid']} (Expected: False)")
    print(f"Issues: {res['issues']}")

if __name__ == "__main__":
    test_strict_validation()
