"""
Example usage of the RAG Service.

This file demonstrates how to use the RAG service components.
Run this after setting up the environment and dependencies.

Usage:
    python -m backend.rag_service.example_usage
"""

import asyncio
from pathlib import Path


def example_basic_usage():
    """Basic usage example with mock data."""
    print("=" * 60)
    print("RAG Service - Basic Usage Example")
    print("=" * 60)
    
    from backend.rag_service.core.rag_engine import RAGEngine, RAGConfig
    from backend.rag_service.core.retriever import RetrievalResult
    from backend.rag_service.providers.base import Message
    
    # 1. Create RAG Engine with default config
    config = RAGConfig(
        top_k=5,
        include_citations=True,
        temperature=0.7,
    )
    
    engine = RAGEngine(config=config, provider_name="lmstudio")
    
    print("\n✓ RAG Engine created")
    print(f"  Provider: {engine.generator.provider.name}")
    print(f"  Config: top_k={config.top_k}, citations={config.include_citations}")
    
    # Note: Actual query requires:
    # - Qdrant running with indexed fatwas
    # - LLM provider running (e.g., LM Studio)
    
    print("\n[To run actual queries, ensure Qdrant and LLM provider are running]")


def example_provider_switching():
    """Example of switching between LLM providers."""
    print("\n" + "=" * 60)
    print("Provider Switching Example")
    print("=" * 60)
    
    from backend.rag_service.providers import get_provider, ProviderFactory
    from backend.rag_service.providers.base import ProviderConfig, ProviderType
    
    # List available providers
    print("\nAvailable providers:", ProviderFactory.list_providers())
    
    # Create LM Studio provider (local)
    try:
        lmstudio = get_provider("lmstudio")
        print(f"\n✓ LM Studio provider: {lmstudio}")
        print(f"  Health check: {lmstudio.health_check()}")
    except Exception as e:
        print(f"\n✗ LM Studio not available: {e}")
    
    # Create Gemini provider (requires API key)
    try:
        gemini = get_provider("gemini")
        print(f"\n✓ Gemini provider: {gemini}")
    except Exception as e:
        print(f"\n✗ Gemini not available: {e}")
    
    # Custom provider configuration
    custom_config = ProviderConfig(
        provider_type=ProviderType.OPENAI,
        base_url="http://localhost:8000/v1",  # Custom endpoint
        model="my-custom-model",
        max_tokens=2000,
        temperature=0.5,
    )
    print(f"\nCustom config created: {custom_config}")


def example_citation_generation():
    """Example of citation generation."""
    print("\n" + "=" * 60)
    print("Citation Generation Example")
    print("=" * 60)
    
    from backend.rag_service.citation import CitationGenerator, MetadataHandler
    from backend.rag_service.core.retriever import RetrievalResult
    
    # Sample documents (as if retrieved from Qdrant)
    documents = [
        RetrievalResult(
            id="fatwa_001",
            content="صلاة الجماعة واجبة على الرجال في أصح أقوال أهل العلم، وقد دل على ذلك الكتاب والسنة",
            score=0.95,
            metadata={
                "scholar": "ابن باز",
                "source": "فتاوى نور على الدرب",
                "fatwa_id": "12345",
                "url": "https://binbaz.org/fatwas/12345",
                "date": "1420-01-15",
                "category": "صلاة",
            }
        ),
        RetrievalResult(
            id="fatwa_002",
            content="من سمع النداء فلم يأته فلا صلاة له إلا من عذر",
            score=0.87,
            metadata={
                "scholar": "العثيمين",
                "source": "الشرح الممتع",
                "fatwa_id": "67890",
            }
        ),
    ]
    
    # Generate citations
    gen = CitationGenerator(format_style="islamic_scholarly")
    citations = gen.generate_citations(documents)
    
    print("\nGenerated Citations:")
    for citation in citations:
        print(f"\n{citation['formatted']}")
    
    # Format references section
    references = gen.format_references_section(citations)
    print(references)


def example_arabic_preprocessing():
    """Example of Arabic text preprocessing."""
    print("\n" + "=" * 60)
    print("Arabic Preprocessing Example")
    print("=" * 60)
    
    from backend.rag_service.utils.preprocessing import (
        preprocess_arabic_text,
        normalize_arabic,
        remove_diacritics,
        detect_language,
    )
    
    # Sample Arabic text with diacritics
    text_with_diacritics = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
    
    print(f"\nOriginal: {text_with_diacritics}")
    print(f"Without diacritics: {remove_diacritics(text_with_diacritics)}")
    
    # Normalization
    text_variants = "إِنَّا أَعْطَيْنَاكَ الْكَوْثَرَ"
    normalized = normalize_arabic(text_variants)
    print(f"\nOriginal: {text_variants}")
    print(f"Normalized: {normalized}")
    
    # Language detection
    test_texts = [
        "ما حكم صلاة الجماعة؟",
        "What is the ruling on prayer?",
        "Mixed نص text",
    ]
    
    print("\nLanguage Detection:")
    for text in test_texts:
        lang = detect_language(text)
        print(f"  '{text[:30]}...' -> {lang}")


def example_translation_interface():
    """Example of translation integration."""
    print("\n" + "=" * 60)
    print("Translation Integration Example")
    print("=" * 60)
    
    from backend.rag_service.translation import (
        TranslationIntegration,
        MockTranslationProvider,
        Language,
    )
    
    # Create integration with mock provider
    integration = TranslationIntegration(
        provider=MockTranslationProvider()
    )
    
    # Simulate query preprocessing
    query_en = "What is the ruling on congregational prayer?"
    processed_query, original_lang = integration.pre_retrieval_hook(query_en)
    
    print(f"\nOriginal query: {query_en}")
    print(f"Detected language: {original_lang.value}")
    print(f"Processed query: {processed_query}")
    
    # Simulate response postprocessing
    response_ar = "صلاة الجماعة واجبة على الرجال"
    final_response = integration.post_generation_hook(response_ar, original_lang)
    
    print(f"\nGenerated response (AR): {response_ar}")
    print(f"Final response: {final_response}")
    
    # Health check
    health = integration.health_check()
    print(f"\nIntegration health: {health}")


def example_prompts():
    """Example of prompt templates."""
    print("\n" + "=" * 60)
    print("Prompt Templates Example")
    print("=" * 60)
    
    from backend.rag_service.utils.prompts import (
        get_system_prompt,
        build_rag_prompt,
    )
    
    # Get system prompt
    system_ar = get_system_prompt("ar")
    print("\nArabic System Prompt (first 200 chars):")
    print(system_ar[:200] + "...")
    
    # Build RAG prompt
    context = """[المصدر 1]
العالم: الشيخ ابن باز
صلاة الجماعة واجبة على الرجال..."""
    
    query = "ما حكم صلاة الجماعة؟"
    
    rag_prompt = build_rag_prompt(query, context, language="ar")
    print("\nRAG Prompt (first 300 chars):")
    print(rag_prompt[:300] + "...")


def example_validation():
    """Example of validation utilities."""
    print("\n" + "=" * 60)
    print("Validation Example")
    print("=" * 60)
    
    from backend.rag_service.utils.validators import (
        validate_query,
        validate_response,
        validate_citations,
    )
    
    # Validate query
    good_query = "ما حكم صلاة الجماعة في المسجد؟"
    result = validate_query(good_query)
    print(f"\nQuery validation: '{good_query[:30]}...'")
    print(f"  Valid: {result.valid}")
    print(f"  Errors: {result.errors}")
    
    # Validate response
    response = """صلاة الجماعة واجبة [1].
    
    قال الشيخ ابن باز: صلاة الجماعة فرض عين [1].
    
    المراجع:
    [1] فتاوى نور على الدرب"""
    
    result = validate_response(response)
    print(f"\nResponse validation:")
    print(f"  Valid: {result.valid}")
    print(f"  Warnings: {result.warnings}")


async def example_async_usage():
    """Example of async RAG usage."""
    print("\n" + "=" * 60)
    print("Async Usage Example")
    print("=" * 60)
    
    from backend.rag_service.core.generator import Generator
    from backend.rag_service.providers.base import Message
    
    # Note: This requires a running LLM provider
    print("\n[Async streaming requires a running LLM provider]")
    print("Example code:")
    print("""
    async def stream_response():
        generator = Generator(provider_name="lmstudio")
        messages = [
            Message(role="user", content="ما حكم الصلاة؟")
        ]
        
        async for token in generator.stream(messages):
            print(token, end="", flush=True)
    """)


def main():
    """Run all examples."""
    print("\n" + "#" * 60)
    print("#  RAG Service Usage Examples")
    print("#" * 60)
    
    try:
        example_basic_usage()
        example_provider_switching()
        example_citation_generation()
        example_arabic_preprocessing()
        example_translation_interface()
        example_prompts()
        example_validation()
        
        # Async example
        asyncio.run(example_async_usage())
        
    except ImportError as e:
        print(f"\n⚠️  Import error: {e}")
        print("Make sure you've installed dependencies:")
        print("  pip install -r backend/rag_service/requirements.txt")
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
    
    print("\n" + "#" * 60)
    print("#  Examples Complete")
    print("#" * 60)


if __name__ == "__main__":
    main()
