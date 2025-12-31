
import asyncio
from unittest.mock import MagicMock
from rag_service.core.generator import Generator

def test_relevance_heuristic():
    gen = Generator()
    
    # Test 1: Relevant Context
    query = "ruling on riba loan"
    context = "Fatwa: Buying house with riba loan is debated."
    is_relevant = gen._check_source_relevance(query, context)
    print(f"Test 1 (Relevant): {is_relevant} (Expected: True)")
    
    # Test 2: Irrelevant Context
    query = "prayer times in london"
    context = "Fatwa: Zakat on gold."
    is_relevant = gen._check_source_relevance(query, context)
    print(f"Test 2 (Irrelevant): {is_relevant} (Expected: False)")
    
    # Test 3: Short query words ignored
    query = "is it haram" # all strictly ignored or short
    context = "haram to eat pork"
    # 'haram' is ignored in set, 'pork' not in query. 
    # Actually 'haram' is in ignored list.
    is_relevant = gen._check_source_relevance(query, context)
    print(f"Test 3 (Stopwords): {is_relevant} (Expected: False or maybe True if lenient)")

if __name__ == "__main__":
    test_relevance_heuristic()
