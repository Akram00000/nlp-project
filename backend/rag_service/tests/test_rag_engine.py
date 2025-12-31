"""
Tests for RAG Engine core functionality.

Run with: pytest backend/rag_service/tests/test_rag_engine.py
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from dataclasses import dataclass

# Import modules to test
from ..core.rag_engine import RAGEngine, RAGConfig, RAGResponse
from ..core.retriever import Retriever, RetrievalResult, RetrievalQuery
from ..core.generator import Generator, GenerationConfig
from ..core.context_manager import ContextManager, ContextConfig
from ..providers.base import Message, LLMResponse


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_retrieval_results():
    """Sample retrieval results for testing."""
    return [
        RetrievalResult(
            id="fatwa_001",
            content="صلاة الجماعة واجبة على الرجال في أصح أقوال أهل العلم",
            score=0.95,
            metadata={
                "scholar": "ابن باز",
                "source": "فتاوى نور على الدرب",
                "fatwa_id": "12345",
                "category": "صلاة",
            }
        ),
        RetrievalResult(
            id="fatwa_002", 
            content="من سمع النداء فلم يأته فلا صلاة له إلا من عذر",
            score=0.87,
            metadata={
                "scholar": "ابن عثيمين",
                "source": "الشرح الممتع",
                "fatwa_id": "67890",
            }
        ),
    ]


@pytest.fixture
def mock_llm_response():
    """Sample LLM response."""
    return LLMResponse(
        content="""صلاة الجماعة واجبة على الرجال [1].

قال الشيخ ابن باز رحمه الله: صلاة الجماعة فرض عين [1].

والدليل على ذلك حديث: من سمع النداء فلم يأته فلا صلاة له [2].

المراجع:
[1] فتاوى نور على الدرب - ابن باز
[2] الشرح الممتع - ابن عثيمين""",
        model="test-model",
        provider="test",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    )


@pytest.fixture
def mock_provider():
    """Mock LLM provider."""
    provider = Mock()
    provider.name = "mock"
    provider.model = "mock-model"
    provider.health_check.return_value = True
    return provider


# =============================================================================
# RETRIEVER TESTS
# =============================================================================

class TestRetriever:
    """Tests for Retriever class."""
    
    def test_retrieval_result_properties(self, mock_retrieval_results):
        """Test RetrievalResult property accessors."""
        result = mock_retrieval_results[0]
        
        assert result.scholar == "ابن باز"
        assert result.source == "فتاوى نور على الدرب"
        assert result.fatwa_id == "12345"
        assert result.category == "صلاة"
    
    def test_retrieval_query_defaults(self):
        """Test RetrievalQuery default values."""
        query = RetrievalQuery(text="ما حكم صلاة الجماعة؟")
        
        assert query.top_k == 5
        assert query.similarity_threshold == 0.5
        assert query.rerank == True
        assert query.filters == {}
    
    @patch('rag_service.core.retriever.QdrantClient')
    @patch('rag_service.core.retriever.SentenceTransformer')
    def test_retriever_initialization(self, mock_st, mock_qdrant):
        """Test retriever can be initialized."""
        retriever = Retriever(collection_name="test_fatwas")
        
        assert retriever.collection_name == "test_fatwas"
        assert not retriever._initialized


# =============================================================================
# CONTEXT MANAGER TESTS
# =============================================================================

class TestContextManager:
    """Tests for ContextManager class."""
    
    def test_format_context_empty(self):
        """Test formatting empty document list."""
        cm = ContextManager()
        result = cm.format_context([])
        
        assert result.text == ""
        assert result.documents_used == []
        assert result.total_tokens_estimate == 0
    
    def test_format_context_with_documents(self, mock_retrieval_results):
        """Test formatting documents into context."""
        cm = ContextManager()
        result = cm.format_context(mock_retrieval_results)
        
        assert len(result.documents_used) == 2
        assert "[المصدر 1]" in result.text or "[Source 1]" in result.text
        assert "ابن باز" in result.text
        assert result.total_tokens_estimate > 0
    
    def test_token_estimation_arabic(self):
        """Test token estimation for Arabic text."""
        cm = ContextManager()
        
        arabic_text = "بسم الله الرحمن الرحيم"
        tokens = cm._estimate_tokens(arabic_text)
        
        # Arabic should estimate more tokens per character
        assert tokens > 0
    
    def test_is_arabic_detection(self):
        """Test Arabic text detection."""
        cm = ContextManager()
        
        assert cm._is_arabic("هذا نص عربي") == True
        assert cm._is_arabic("This is English") == False
        assert cm._is_arabic("Mixed نص text") == True  # 30%+ Arabic
    
    def test_split_into_chunks(self):
        """Test text chunking."""
        cm = ContextManager()
        
        long_text = "هذا نص طويل. " * 100
        chunks = cm.split_into_chunks(long_text, chunk_size=200)
        
        assert len(chunks) > 1
        assert all(len(c) > 0 for c in chunks)


# =============================================================================
# GENERATOR TESTS  
# =============================================================================

class TestGenerator:
    """Tests for Generator class."""
    
    def test_normalize_messages_from_dicts(self):
        """Test message normalization from dictionaries."""
        gen = Generator(provider_name="lmstudio")
        
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        
        normalized = gen._normalize_messages(messages)
        
        assert len(normalized) == 2
        assert all(isinstance(m, Message) for m in normalized)
        assert normalized[0].role == "system"
        assert normalized[1].content == "Hello"
    
    def test_extract_params_from_config(self):
        """Test parameter extraction from config."""
        gen = Generator(provider_name="lmstudio")
        
        config = GenerationConfig(
            max_tokens=500,
            temperature=0.5,
        )
        
        params = gen._extract_params(config, {})
        
        assert params["max_tokens"] == 500
        assert params["temperature"] == 0.5
    
    def test_extract_params_kwargs_override(self):
        """Test kwargs override config values."""
        gen = Generator(provider_name="lmstudio")
        
        config = GenerationConfig(max_tokens=500)
        params = gen._extract_params(config, {"max_tokens": 1000})
        
        assert params["max_tokens"] == 1000


# =============================================================================
# RAG ENGINE TESTS
# =============================================================================

class TestRAGEngine:
    """Tests for main RAGEngine class."""
    
    def test_rag_config_defaults(self):
        """Test RAGConfig default values."""
        config = RAGConfig()
        
        assert config.top_k == 5
        assert config.include_citations == True
        assert config.enable_safety_checks == True
    
    def test_rag_response_structure(self, mock_retrieval_results):
        """Test RAGResponse dataclass."""
        response = RAGResponse(
            answer="Test answer",
            citations=[{"index": 1, "formatted": "Test citation"}],
            sources=mock_retrieval_results,
            metadata={"query": "test"},
        )
        
        assert response.answer == "Test answer"
        assert len(response.citations) == 1
        assert len(response.sources) == 2
    
    def test_rag_response_with_citations(self):
        """Test answer_with_citations property."""
        response = RAGResponse(
            answer="This is the answer [1].",
            citations=[
                {"index": 1, "formatted": "Ibn Baz - Fatawa"},
            ],
        )
        
        full_answer = response.answer_with_citations
        
        assert "This is the answer" in full_answer
        assert "المراجع" in full_answer or "References" in full_answer
        assert "Ibn Baz" in full_answer
    
    def test_no_results_message_arabic(self):
        """Test no results message in Arabic."""
        engine = RAGEngine()
        msg = engine._get_no_results_message("ar")
        
        assert "عذراً" in msg
        assert "لم أتمكن" in msg
    
    def test_no_results_message_english(self):
        """Test no results message in English."""
        engine = RAGEngine()
        msg = engine._get_no_results_message("en")
        
        assert "Sorry" in msg
        assert "couldn't find" in msg


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestRAGIntegration:
    """Integration tests for RAG pipeline."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_mock(
        self,
        mock_retrieval_results,
        mock_llm_response,
        mock_provider,
    ):
        """Test full RAG pipeline with mocks."""
        # Setup mocks
        mock_provider.generate.return_value = mock_llm_response
        
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        mock_retriever.health_check.return_value = True
        
        # Create engine with mocks
        engine = RAGEngine(
            retriever=mock_retriever,
            provider=mock_provider,
        )
        
        # Run query
        response = engine.query("ما حكم صلاة الجماعة؟")
        
        # Verify
        assert response.answer is not None
        assert len(response.sources) > 0
        mock_retriever.retrieve.assert_called_once()


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
