"""
Embedding Module

Handles text embedding using Sentence Transformers.
Implements efficient batch processing for large document sets.
"""

import logging
from typing import List, Union, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Embedding model wrapper using Sentence Transformers.
    
    Uses dense bi-encoder style embeddings as described in the RAG paper
    for efficient semantic similarity matching.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model name for embeddings
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for encoding
        """
        if SentenceTransformer is None:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
        
        self.model_name = model_name
        self.batch_size = batch_size
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True  # For cosine similarity
        )
        return embedding
    
    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed multiple texts efficiently in batches.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar
            
        Returns:
            Matrix of embeddings (num_texts x dimension)
        """
        logger.info(f"Embedding {len(texts)} texts...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a search query.
        
        Note: Some models have separate query encoders, but MiniLM uses
        the same encoder for both queries and documents.
        
        Args:
            query: Search query
            
        Returns:
            Query embedding vector
        """
        return self.embed_text(query)
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: Matrix of document embeddings
            
        Returns:
            Array of similarity scores
        """
        # Since embeddings are normalized, dot product = cosine similarity
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        similarities = np.dot(document_embeddings, query_embedding.T).flatten()
        return similarities


class EmbeddingCache:
    """
    Simple cache for embeddings to avoid recomputation.
    """
    
    def __init__(self, embedding_model: EmbeddingModel):
        self.model = embedding_model
        self._cache = {}
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from cache or compute it."""
        text_hash = hash(text)
        
        if text_hash not in self._cache:
            self._cache[text_hash] = self.model.embed_text(text)
        
        return self._cache[text_hash]
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()
