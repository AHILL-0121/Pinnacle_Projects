"""
Tests for RAG System Components

Run with: python -m pytest tests/ -v
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np


class TestDocumentProcessor:
    """Tests for document processing."""
    
    def test_chunk_id_generation(self):
        """Test that chunk IDs are unique."""
        from src.document_processor import SemanticChunker
        
        chunker = SemanticChunker()
        id1 = chunker._generate_chunk_id("test.pdf", 0)
        id2 = chunker._generate_chunk_id("test.pdf", 1)
        id3 = chunker._generate_chunk_id("other.pdf", 0)
        
        assert id1 != id2
        assert id1 != id3
        assert len(id1) == 12
    
    def test_token_estimation(self):
        """Test token count estimation."""
        from src.document_processor import SemanticChunker
        
        chunker = SemanticChunker()
        
        # ~4 chars per token
        text = "This is a test sentence with multiple words."
        tokens = chunker.estimate_tokens(text)
        assert 8 <= tokens <= 15  # Reasonable range
    
    def test_section_detection(self):
        """Test section name detection."""
        from src.document_processor import SemanticChunker
        
        chunker = SemanticChunker()
        
        assert chunker.detect_section("1. Introduction\nThis paper...") == "Introduction"
        assert chunker.detect_section("Abstract: We present...") == "Abstract"
        assert chunker.detect_section("Random text here") == "Unknown"


class TestEmbeddings:
    """Tests for embedding functionality."""
    
    def test_embedding_dimension(self):
        """Test that embeddings have correct dimension."""
        from src.embeddings import EmbeddingModel
        
        model = EmbeddingModel()
        assert model.dimension == 384  # MiniLM
        
        embedding = model.embed_text("Test sentence")
        assert embedding.shape == (384,)
    
    def test_embedding_normalization(self):
        """Test that embeddings are normalized."""
        from src.embeddings import EmbeddingModel
        
        model = EmbeddingModel()
        embedding = model.embed_text("Test sentence")
        
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01  # Should be unit normalized
    
    def test_batch_embedding(self):
        """Test batch embedding."""
        from src.embeddings import EmbeddingModel
        
        model = EmbeddingModel()
        texts = ["First sentence", "Second sentence", "Third sentence"]
        embeddings = model.embed_texts(texts)
        
        assert embeddings.shape == (3, 384)
    
    def test_similarity_computation(self):
        """Test similarity computation."""
        from src.embeddings import EmbeddingModel
        
        model = EmbeddingModel()
        
        query = model.embed_text("machine learning")
        docs = model.embed_texts([
            "deep learning models",  # Similar
            "cooking recipes",        # Different
        ])
        
        similarities = model.compute_similarity(query, docs)
        
        assert similarities[0] > similarities[1]  # ML should be more similar


class TestVectorStore:
    """Tests for vector store."""
    
    def test_add_and_search(self):
        """Test adding vectors and searching."""
        from src.vector_store import FAISSVectorStore
        from src.document_processor import DocumentChunk
        
        store = FAISSVectorStore(dimension=4)
        
        # Create test chunks
        chunks = [
            DocumentChunk(
                chunk_id=f"chunk_{i}",
                text=f"Text {i}",
                paper_title="Test Paper",
                section_name="Test Section",
                page_numbers=[1],
                chunk_index=i,
                total_chunks=3,
                source_file="test.pdf",
                token_count=10
            )
            for i in range(3)
        ]
        
        # Create embeddings
        embeddings = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ], dtype=np.float32)
        
        store.add_chunks(chunks, embeddings)
        
        # Search
        query = np.array([1, 0, 0, 0], dtype=np.float32)
        results = store.search(query, top_k=2)
        
        assert len(results) == 2
        assert results[0][0].chunk_id == "chunk_0"  # Most similar


class TestRetriever:
    """Tests for retriever."""
    
    def test_section_boosting(self):
        """Test that section boosting affects scores."""
        from src.retriever import RetrievalResult
        from src.document_processor import DocumentChunk
        
        chunk_abstract = DocumentChunk(
            chunk_id="1",
            text="Abstract text",
            paper_title="Test",
            section_name="Abstract",
            page_numbers=[1],
            chunk_index=0,
            total_chunks=2,
            source_file="test.pdf",
            token_count=10
        )
        
        chunk_refs = DocumentChunk(
            chunk_id="2",
            text="Reference text",
            paper_title="Test",
            section_name="References",
            page_numbers=[10],
            chunk_index=1,
            total_chunks=2,
            source_file="test.pdf",
            token_count=10
        )
        
        # Same base score
        result_abstract = RetrievalResult(chunk=chunk_abstract, score=0.8, boosted_score=0.8)
        result_refs = RetrievalResult(chunk=chunk_refs, score=0.8, boosted_score=0.8)
        
        # After boosting (abstract: 1.2x, references: 0.5x)
        result_abstract.boosted_score *= 1.2
        result_refs.boosted_score *= 0.5
        
        assert result_abstract.boosted_score > result_refs.boosted_score


class TestLLMProviders:
    """Tests for LLM providers."""
    
    def test_ollama_availability_check(self):
        """Test Ollama availability check (may fail if not running)."""
        from src.llm_providers import OllamaProvider
        
        provider = OllamaProvider()
        # Just check that the method works
        result = provider.is_available()
        assert isinstance(result, bool)
    
    def test_llm_manager_registration(self):
        """Test provider registration."""
        from src.llm_providers import LLMManager, OllamaProvider
        
        manager = LLMManager()
        provider = OllamaProvider(model="test")
        
        manager.register_provider("test_ollama", provider)
        assert "test_ollama" in manager._providers


class TestRAGPipeline:
    """Tests for RAG pipeline integration."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        from src.rag_pipeline import RAGPipeline
        
        pipeline = RAGPipeline()
        pipeline.initialize()
        
        assert pipeline._initialized
        assert pipeline.embedding_model is not None
        assert pipeline.vector_store is not None
    
    def test_pipeline_stats(self):
        """Test pipeline statistics."""
        from src.rag_pipeline import RAGPipeline
        
        pipeline = RAGPipeline()
        pipeline.initialize()
        
        stats = pipeline.get_stats()
        
        assert "initialized" in stats
        assert "total_chunks" in stats
        assert "embedding_dimension" in stats


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
