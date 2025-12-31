
import sys
import os

# Add backend to path
sys.path.append('.')

from rag_service.utils.metadata_enricher import MetadataEnricher

def test_metadata_enrichment_specific():
    enricher = MetadataEnricher()
    
    # Case 1: Women 102 in typical fatwa context
    content1 = "وقوله تعالى في سورة النساء آية 102: {وَإِذَا كُنْتَ فِيهِمْ فَأَقَمْتَ لَهُمُ الصَّلَاةَ...}"
    meta1 = {"type": "quran", "source": "islamweb"}
    enriched1 = enricher.enrich(content1, meta1, "test_1")
    
    print(f"Case 1 (سورة النساء آية 102) -> verse_ref: {enriched1.verse_ref}")
    assert enriched1.verse_ref == "النساء:102"
    
    # Case 2: [النساء: 102] style
    content2 = "قال الله تعالى: {وَإِذَا كُنْتَ فِيهِمْ...} [النساء: 102]"
    enriched2 = enricher.enrich(content2, meta1, "test_2")
    print(f"Case 2 ([النساء: 102]) -> verse_ref: {enriched2.verse_ref}")
    assert enriched2.verse_ref == "النساء:102"
    
    # Case 3: Just the text with colon
    content3 = "النساء: 102 تدل على وجوب الجماعة"
    enriched3 = enricher.enrich(content3, meta1, "test_3")
    print(f"Case 3 (النساء: 102) -> verse_ref: {enriched3.verse_ref}")
    assert enriched3.verse_ref == "النساء:102"

    print("\nSUCCESS: Metadata enrichment correctly identifies An-Nisa 102 across formats.")

if __name__ == "__main__":
    test_metadata_enrichment_specific()
