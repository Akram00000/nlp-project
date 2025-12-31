"""
Tests for Citation Generator functionality.
"""

import pytest
from ..citation.citation_generator import CitationGenerator, Citation
from ..citation.source_tracker import SourceTracker, TrackedSource
from ..citation.metadata_handler import MetadataHandler, FatwaMetadata
from ..core.retriever import RetrievalResult


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_documents():
    """Sample documents for citation testing."""
    return [
        RetrievalResult(
            id="doc_001",
            content="صلاة الجماعة واجبة على الرجال",
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
            id="doc_002",
            content="الصلاة عماد الدين",
            score=0.85,
            metadata={
                "scholar": "العثيمين",
                "source": "الشرح الممتع",
            }
        ),
    ]


@pytest.fixture
def raw_fatwa_data():
    """Raw fatwa data as would come from JSONL."""
    return {
        "id": "f_001",
        "question": "ما حكم صلاة الجماعة؟",
        "answer": "صلاة الجماعة واجبة...",
        "mufti": "ابن باز",
        "website": "binbaz.org",
        "link": "https://binbaz.org/fatwas/123",
        "publish_date": "2020-01-15",
        "topic": "الصلاة",
    }


# =============================================================================
# METADATA HANDLER TESTS
# =============================================================================

class TestMetadataHandler:
    """Tests for MetadataHandler."""
    
    def test_extract_metadata_basic(self, raw_fatwa_data):
        """Test basic metadata extraction."""
        handler = MetadataHandler()
        metadata = handler.extract_metadata(raw_fatwa_data)
        
        assert metadata.fatwa_id == "f_001"
        assert metadata.question == "ما حكم صلاة الجماعة؟"
        assert metadata.scholar is not None  # Should be normalized
        assert metadata.url == "https://binbaz.org/fatwas/123"
    
    def test_normalize_scholar_name(self):
        """Test scholar name normalization."""
        handler = MetadataHandler()
        
        assert "ابن باز" in handler._normalize_scholar("بن باز")
        assert "ابن باز" in handler._normalize_scholar("binbaz")
    
    def test_field_variations(self):
        """Test extraction with different field names."""
        handler = MetadataHandler()
        
        # Test with 'mufti' instead of 'scholar'
        data1 = {"mufti": "ابن باز", "q": "سؤال", "a": "جواب"}
        meta1 = handler.extract_metadata(data1)
        assert meta1.scholar is not None
        assert meta1.question == "سؤال"
        assert meta1.answer == "جواب"
    
    def test_language_detection(self):
        """Test language detection in metadata."""
        handler = MetadataHandler()
        
        assert handler._detect_language("هذا نص عربي") == "ar"
        assert handler._detect_language("This is English") == "en"
    
    def test_validate_metadata(self):
        """Test metadata validation."""
        handler = MetadataHandler()
        
        complete_meta = FatwaMetadata(
            fatwa_id="123",
            scholar="ابن باز",
            source="فتاوى",
            title="عنوان",
            date="2020-01-01",
            url="https://example.com",
            category="فقه"
        )
        
        result = handler.validate_metadata(complete_meta)
        assert result["valid"] == True
        assert len(result["missing_required"]) == 0
        
        incomplete_meta = FatwaMetadata()
        result2 = handler.validate_metadata(incomplete_meta)
        assert result2["valid"] == False
        assert len(result2["missing_required"]) > 0


# =============================================================================
# SOURCE TRACKER TESTS
# =============================================================================

class TestSourceTracker:
    """Tests for SourceTracker."""
    
    def test_start_query(self):
        """Test starting a new query tracking session."""
        tracker = SourceTracker()
        query_id = tracker.start_query("ما حكم الصلاة؟")
        
        assert query_id is not None
        assert len(query_id) > 0
    
    def test_add_source(self, sample_documents):
        """Test adding sources to tracking."""
        tracker = SourceTracker()
        query_id = tracker.start_query("test query")
        
        for doc in sample_documents:
            tracker.add_source(query_id, doc)
        
        sources = tracker.get_sources(query_id)
        assert len(sources) == 2
    
    def test_finalize_query(self, sample_documents):
        """Test finalizing query tracking."""
        tracker = SourceTracker()
        query_id = tracker.start_query("test query")
        
        for doc in sample_documents:
            tracker.add_source(query_id, doc)
        
        tracker.finalize_query(query_id, "Generated response text")
        
        record = tracker.get_query_record(query_id)
        assert record.finalized == True
        assert record.response_text == "Generated response text"
    
    def test_export_audit_log(self, sample_documents):
        """Test audit log export."""
        tracker = SourceTracker()
        query_id = tracker.start_query("test query")
        
        for doc in sample_documents:
            tracker.add_source(query_id, doc)
        
        tracker.finalize_query(query_id, "Response")
        
        audit = tracker.export_audit_log(query_id)
        
        assert "query_id" in audit
        assert "sources" in audit
        assert len(audit["sources"]) == 2
        assert audit["finalized"] == True
    
    def test_max_history_eviction(self):
        """Test that old queries are evicted."""
        tracker = SourceTracker(max_history=3)
        
        ids = []
        for i in range(5):
            qid = tracker.start_query(f"Query {i}")
            ids.append(qid)
        
        # First 2 should be evicted
        assert tracker.get_query_record(ids[0]) is None
        assert tracker.get_query_record(ids[1]) is None
        
        # Last 3 should still exist
        assert tracker.get_query_record(ids[2]) is not None
        assert tracker.get_query_record(ids[4]) is not None


# =============================================================================
# CITATION GENERATOR TESTS
# =============================================================================

class TestCitationGenerator:
    """Tests for CitationGenerator."""
    
    def test_generate_citations_islamic_scholarly(self, sample_documents):
        """Test Islamic scholarly citation format."""
        gen = CitationGenerator(format_style="islamic_scholarly")
        citations = gen.generate_citations(sample_documents)
        
        assert len(citations) == 2
        assert citations[0]["index"] == 1
        assert "ابن باز" in citations[0]["formatted"]
        assert citations[0]["scholar"] is not None
    
    def test_generate_citations_simple(self, sample_documents):
        """Test simple citation format."""
        gen = CitationGenerator(format_style="simple")
        citations = gen.generate_citations(sample_documents)
        
        assert len(citations) == 2
        assert "[1]" in citations[0]["formatted"]
    
    def test_generate_citations_detailed(self, sample_documents):
        """Test detailed citation format."""
        gen = CitationGenerator(format_style="detailed")
        citations = gen.generate_citations(sample_documents)
        
        # Detailed format should have multiple lines
        assert "\n" in citations[0]["formatted"]
    
    def test_short_reference(self, sample_documents):
        """Test short reference generation."""
        gen = CitationGenerator()
        citations = gen.generate_citations(sample_documents)
        
        # Short should be compact
        assert len(citations[0]["short"]) < 30
        assert "[1" in citations[0]["short"]
    
    def test_format_references_section(self, sample_documents):
        """Test full references section formatting."""
        gen = CitationGenerator()
        citations = gen.generate_citations(sample_documents)
        
        section = gen.format_references_section(citations)
        
        assert "المراجع" in section or "References" in section
        assert "[1]" in section
        assert "[2]" in section
    
    def test_validate_citations(self, sample_documents):
        """Test citation validation."""
        gen = CitationGenerator()
        citations = gen.generate_citations(sample_documents)
        
        response_text = "This uses citation [1] and also [2]."
        result = gen.validate_citations(citations, response_text)
        
        assert result["valid"] == True
        assert result["completeness"] > 0
    
    def test_validate_citations_missing(self):
        """Test validation with missing citations."""
        gen = CitationGenerator()
        
        # Citations that don't exist
        response_text = "This references [5] which doesn't exist."
        citations = [{"index": 1, "formatted": "Test"}]
        
        result = gen.validate_citations(citations, response_text)
        
        assert "5" in str(result["issues"]) or len(result["issues"]) > 0


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
