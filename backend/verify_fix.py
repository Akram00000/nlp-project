
import asyncio
import logging
from rag_service.core.generator import Generator, GenerationConfig
from rag_service.core.retriever import RetrievalResult, Retriever
from rag_service.utils.fact_checker import FactChecker

# Setup logging to see our new debug logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_service")
logger.setLevel(logging.DEBUG)

async def verify_fix():
    print("Initializing Generator...")
    gen = Generator()
    
    # Mocking Retrieval for deterministic test (or we can use actual retriever if we want integration test)
    # Let's use actual retriever if possible, but we don't know if Qdrant has data.
    # The user said "Retrieval is working correctly". So let's Mock the retrieval output to match strict inputs.
    
    query = "هل يجوز أخذ قرض ربوي لشراء منزل في بلد غير إسلامي؟"
    
    # Mock docs as per User Request scenario
    mock_docs = [
        RetrievalResult(
            id="fatwa_fatwas_2_a53ded", 
            content="السؤال: ما حكم شراء المنازل بقرض بنكي ربوي للمسلمين في غير بلاد الإسلام؟ الجواب: أجاز المجلس الأوروبي للإفتاء والبحوث ذلك للضرورة والحاجة الماسة التي تنزل منزلة الضرورة.",
            score=0.92,
            metadata={"source_type": "fatwa"}
        ),
        RetrievalResult(
            id="fatwa_fatwas_4_43c891", 
            content="الأصل في القروض الربوية التحريم، ولكن في بلاد الغرب حيث لا يوجد بديل إسلامي، رخص بعض العلماء في ذلك لتوفير السكن.",
            score=0.88,
            metadata={"source_type": "fatwa"}
        )
    ]
    
    print(f"\n--- Testing Query: {query} ---")
    
    # We need to Mock the generate_async to return a synthesized answer 
    # because we can't call actual LLM here without API key/Server running accessible to this script env 
    # (actually we might have env vars loaded but let's be safe).
    # BUT, to test "Refusal", we need to see what the Prompt looks like and if Validation passes.
    
    # Let's mock the LLM response to be a "Good Answer" and see if Validation passes.
    # If Validation was the issue, this will pass now.
    
    from unittest.mock import MagicMock, AsyncMock
    
    # Mock LLM response with PREFIX citations to test the fix
    good_response = """
**الحكم الشرعي**: 
شراء منزل بقرض ربوي في بلد غير إسلامي محل خلاف، وقد أجازه بعض العلماء للضرورة [fatwa_fatwas_2].

**التفصيل**:
الأصل هو التحريم، لكن المجلس الأوروبي للإفتاء رخص فيه عند الحاجة الماسة [fatwa_fatwas_2]. وأشار آخرون لعدم وجود بديل [fatwa_fatwas_4].
    """
    # Note: The retrieval result IDs correspond to fatwa_fatwas_2_a53ded and fatwa_fatwas_4_43c891.
    # So [fatwa_fatwas_2] and [fatwa_fatwas_4] should now PASS validation.
    
    # Patch the _call_llm or generate_async method? 
    # Generator.generate_async calls provider.generate.
    # Let's mock provider.generate
    
    mock_provider_response = MagicMock()
    mock_provider_response.content = good_response
    mock_provider_response.model = "test-model"
    mock_provider_response.usage = {}
    
    mock_provider_response.usage = {}
    
    # Mock the provider by setting _provider directly (it's a property)
    mock_provider = MagicMock()
    mock_provider.name = "test-model"
    mock_provider.generate_async = AsyncMock(return_value=mock_provider_response)
    
    gen._provider = mock_provider

    # Run
    # Note: passing context_docs to trigger the logic in generate_scholarly
    # (Checking if new logic uses context_docs properly)
    # The new logic in generator.py uses source_docs = context_docs or []
    
    response = await gen.generate_scholarly(
        query=query,
        context="Should be ignored if docs passed",
        context_docs=mock_docs
    )
    
    print("\n--- Result ---")
    print(f"Content: {response.content}")
    print(f"Valid? {response.valid}")
    print(f"Has Quran Refs: {response.has_quran_refs}")
    print(f"Has Hadith Refs: {response.has_hadith_refs}")
    
    if "المصادر المتاحة لا تحتوي" in response.content:
        print("FAIL: System Refused to Answer!")
    else:
        print("SUCCESS: System Generated Answer!")

if __name__ == "__main__":
    asyncio.run(verify_fix())
