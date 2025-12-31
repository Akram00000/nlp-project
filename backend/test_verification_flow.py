
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from rag_service.core.generator import Generator, GenerationConfig
from rag_service.core.retriever import RetrievalResult
from rag_service.utils.verification_prompts import SOURCE_VERIFICATION_PROMPT
from loguru import logger

# Mock Document
doc1 = RetrievalResult(
    id="doc1",
    content="The Prophet (peace be upon him) said: 'Prayer is the pillar of religion.'",
    score=0.9,
    metadata={"type": "hadith"}
)
doc2 = RetrievalResult(
    id="doc2",
    content="Trading in wine is prohibited in Islam.",
    score=0.8,
    metadata={"type": "hadith"}
)

async def test_flow():
    logger.info("Initializing Generator...")
    # diverse config
    gen = Generator(provider_name="lmstudio") 
    
    query = "What is the ruling on congregational prayer?"
    
    logger.info(f"Testing Verification for Query: '{query}'")
    
    # Mocking verify_relevance to avoid actual LLM calls if needed, 
    # but we want to test the LLM call if possible.
    # If LM Studio is down, this will fail. Assuming it's up.
    
    # We will try to call verify_relevance directly first
    try:
        verified = await gen.verify_relevance(query, [doc1, doc2])
        print(f"Verified Docs Count: {len(verified)}")
        for d in verified:
            print(f" - Kept: {d.content}")
            
        if len(verified) == 1 and verified[0].id == "doc1":
            print("SUCCESS: Irrelevant doc filtered out.")
        else:
            print(f"WARNING: Filtering results unexpected: {[d.id for d in verified]}")
            
    except Exception as e:
        print(f"Verification Check Failed (Is LLM running?): {e}")

    # Test Scholarly Generation with context_docs
    try:
        print("\nTesting Generate Scholarly...")
        response = await gen.generate_scholarly(
            query=query,
            context="CTX", # This will be ignored/overwritten if verification works
            context_docs=[doc1, doc2]
        )
        print("Response Generated:")
        print(response.content[:200] + "...")
        
    except Exception as e:
        print(f"Generation Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_flow())
