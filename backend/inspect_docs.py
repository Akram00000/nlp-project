
import asyncio
import sys
import io
from pathlib import Path

# Fix encoding for Windows console/redirection
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from rag_service.core.retriever import Retriever

async def inspect():
    retriever = Retriever(collection_name="fatwas")
    retriever.initialize()
    
    query = "ما الآيات القرآنية التي استُدل بها على وجوب صلاة الجماعة؟"
    results = retriever.retrieve(query=query, top_k=5)
    
    for i, doc in enumerate(results):
        print(f"\n--- Document {i+1}: {doc.id} ---")
        print(f"Score: {doc.score}")
        print(f"Full Content (first 2000 chars):\n")
        print(doc.content[:2000])
        print("\n" + "="*50)

if __name__ == "__main__":
    asyncio.run(inspect())
