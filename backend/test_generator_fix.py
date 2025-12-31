
import asyncio
import sys
from unittest.mock import MagicMock, AsyncMock

# Mock configs before modules load if needed
# But dependencies are just local files.

from rag_service.core.generator import Generator, GenerationConfig
from rag_service.core.retriever import RetrievalResult
from rag_service.core.context_manager import ContextManager

async def test_scholarly_gen_fix():
    print("Initializing Generator...")
    # Mock provider config
    # We need to mock the provider factory or just let it fail at generation step 
    # but succeed at verification step logic (where the bug was).
    
    # Actually, verify_relevance calls the LLM. 
    # generate_async calls the LLM.
    # We should mock verify_relevance to return docs immediately to trigger the "if verified_docs" block.
    
    gen = Generator()
    
    # Mock verify_relevance to bypass LLM call and return docs
    gen.verify_relevance = AsyncMock(return_value=[
        RetrievalResult(id="doc1", content="Verified Content", metadata={"type":"fatwa"}, score=1.0)
    ])
    
    # Mock generate_async to avoid actual LLM call
    mock_response = MagicMock()
    mock_response.content = "Answer with [doc1]"
    gen.generate_async = AsyncMock(return_value=mock_response)
    
    docs = [
        RetrievalResult(id="doc1", content="Verified Content", metadata={"type":"fatwa"}, score=1.0)
    ]
    
    print("Calling generate_scholarly with docs...")
    try:
        response = await gen.generate_scholarly(
            query="test",
            context="original context",
            context_docs=docs
        )
        print("Success! Response generated.")
        print(f"Content: {response.content}")
        
    except NameError as e:
        print(f"FAILED with NameError: {e}")
    except Exception as e:
        # We might fail on other things (validation etc) but as long as it's not NameError on formatted_context
        if "formatted_context" in str(e):
            print(f"FAILED with formatted_context error: {e}")
        else:
            print(f"Failed with other error (might be OK if unrelated): {e}")

if __name__ == "__main__":
    asyncio.run(test_scholarly_gen_fix())
