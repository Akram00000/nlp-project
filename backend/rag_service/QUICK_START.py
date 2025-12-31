"""
Quick Start Guide - Data Ingestion & Linked Retrieval

Follow these steps to index your fatwa data and use the RAG system.
"""

# STEP 1: Index Your Fatwa Data
# ================================

from pathlib import Path
from backend.rag_service.data_ingestion import VectorIndexer

# Initialize indexer
indexer = VectorIndexer()
indexer.initialize(
    qdrant_host="localhost",
    qdrant_port=6333,
)

# Index your existing fatwa files
result = indexer.index_fatwas(
    data_path=Path("data"),  # Where your 165kFatwa.jsonl and binbaz_fatwas.jsonl are
    collection_name="fatwas",
    pattern="*.jsonl",
    force_recreate=False,
)

print(f"✓ Indexed {result['indexed']} fatwas")


# STEP 2: Index Hadith Data (when available)
# ===========================================

# When you add hadith JSON files:
result = indexer.index_hadiths(
    data_path=Path("data/hadiths"),
    collection_name="hadiths",
    pattern="*.jsonl",
)

print(f"✓ Indexed {result['indexed']} hadiths")


# STEP 3: Use in RAG Pipeline with Linked Retrieval
# ==================================================

from backend.rag_service.core.rag_engine import RAGEngine, RAGConfig

# Configure with linked retrieval enabled
config = RAGConfig(
    top_k=5,
    use_linked_retrieval=True,  # ← Enables automatic hadith fetching
    hadith_collection="hadiths",
    max_hadiths_per_fatwa=2,
    include_citations=True,
)

engine = RAGEngine(config=config, provider_name="lmstudio")

# Query
response = engine.query("ما حكم صلاة الجماعة؟")

print(response.answer_with_citations)
# This will include:
# - Relevant fatwas
# - Hadiths mentioned in those fatwas
# - Proper citations


# ALTERNATIVE: Use CLI for Indexing
# ===================================

"""
# Index fatwas
python -m backend.rag_service.data_ingestion.cli index-fatwas data/

# Index hadiths
python -m backend.rag_service.data_ingestion.cli index-hadiths data/hadiths/

# Check what references are extracted
python -m backend.rag_service.data_ingestion.cli check-references data/165kFatwa.jsonl --limit 20
"""
