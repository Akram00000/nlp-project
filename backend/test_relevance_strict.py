
import asyncio
from dataclasses import dataclass
from typing import Optional
import sys
from unittest.mock import MagicMock, AsyncMock

# Add backend to path
sys.path.append('.')

from rag_service.core.generator import Generator, GenerationConfig, ScholarlyResponse
from rag_service.core.retriever import RetrievalResult
from rag_service.utils.metadata_enricher import EnrichedMetadata

@dataclass
class MockMetadata:
    citation_id: str
    source_type: str
    verse_ref: Optional[str] = None
    hadith_ref: Optional[str] = None

async def test_relevance_strict():
    # 1. Setup Mock Provider
    mock_provider = MagicMock()
    mock_provider.name = "mock_llm"
    
    # Mock Response content with thought block and correct citation
    mock_response_text = """
<thought>
I found 2 sources. 
Source 1: [quran_annisa_102] discusses the prayer of fear (congregational prayer in battle), which is highly relevant to "congregational prayer".
Source 2: [quran_albaqarah_240] discusses widow maintenance/divorce, which is completely irrelevant to prayer.
I will use Source 1 and ignore Source 2.
</thought>

**الحكم الشرعي** (Ruling):
أوجب الله تعالى صلاة الجماعة حتى في حال الخوف والقتال، كما ورد في سورة النساء [quran_annisa_102].

**التفصيل والأدلة** (Details and Evidence):
1. سورة النساء، الآية 102: "وَإِذَا كُنْتَ فِيهِمْ فَأَقَمْتَ لَهُمُ الصَّلَاةَ..." [quran_annisa_102]. هذه الآية تدل على مشروعية صلاة الجماعة. أما ما ورد في سورة البقرة آية 240 فهو يتعلق بنفقة المتوفى عنها زوجها ولا علاقة له بالصلاة.
"""
    
    mock_llm_response = MagicMock()
    mock_llm_response.content = mock_response_text
    mock_llm_response.model = "mock-model"
    mock_llm_response.provider = "mock-provider"
    mock_llm_response.usage = {}
    mock_llm_response.raw_response = None
    
    mock_provider.generate_async = AsyncMock(return_value=mock_llm_response)
    
    generator = Generator(provider=mock_provider)
    
    # 2. Mock Documents
    doc1 = RetrievalResult(
        id="doc1",
        content="وَإِذَا كُنْتَ فِيهِمْ فَأَقَمْتَ لَهُمُ الصَّلَاةَ فَلْتَقُمْ طَائِفَةٌ مِنْهُمْ مَعَكَ...",
        score=0.9,
        metadata={"type": "quran"}
    )
    doc1.enriched_metadata = EnrichedMetadata(
        doc_id="doc1",
        source_type="quran",
        citation_id="quran_annisa_102",
        verse_ref="النساء:102"
    )
    
    doc2 = RetrievalResult(
        id="doc2",
        content="وَالَّذِينَ يُتَوَفَّوْنَ مِنْكُمْ وَيَذَرُونَ أَزْوَاجًا وَصِيَّةً لِأَزْوَاجِهِمْ مَتَاعًا إِلَى الْحَوْلِ...",
        score=0.8,
        metadata={"type": "quran"}
    )
    doc2.enriched_metadata = EnrichedMetadata(
        doc_id="doc2",
        source_type="quran",
        citation_id="quran_albaqarah_240",
        verse_ref="البقرة:240"
    )
    
    # 3. Test Generation
    query = "Verses about congregational prayer?"
    response = await generator.generate_scholarly(
        query=query,
        context="",  # ContextManager will be used internally
        context_docs=[doc1, doc2]
    )
    
    print("\n--- TEST RESULTS ---")
    print(f"Thought block found: {bool(response.thought)}")
    if response.thought:
        print(f"Thought preview: {response.thought[:100]}...")
    
    print(f"Contains correct citation: {'quran_annisa_102' in response.content}")
    print(f"Cites irrelevant doc: {'quran_albaqarah_240' in response.content}") # In this mock, I included it in explanation to show it was ignored.
    
    # Check if prompt contains the strict instructions
    args, kwargs = mock_provider.generate_async.call_args
    messages = args[0]
    system_msg = next(m for m in messages if m.role == "system").content
    
    print(f"Strict instructions in prompt: {'RELEVANCE FILTERING' in system_msg}")
    print(f"Chain of Thought requirement in prompt: {'REASONING STEP' in system_msg}")

    if "RELEVANCE FILTERING" in system_msg and bool(response.thought):
        print("\nSUCCESS: Strict RAG instructions implemented and reasoning block extracted.")
    else:
        print("\nFAILURE: Missing instructions or reasoning extraction.")

if __name__ == "__main__":
    asyncio.run(test_relevance_strict())
