"""
Vector Store Module

Implements FAISS-based vector storage and retrieval.
Supports persistence and efficient similarity search.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from .document_processor import DocumentChunk

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS-based vector store for efficient similarity search.
    
    Implements dense vector indexing as used in the RAG paper's
    retrieval component (similar to DPR).
    """
    
    def __init__(self, dimension: int = 384):
        """
        Initialize vector store.
        
        Args:
            dimension: Embedding dimension (384 for MiniLM)
        """
        if faiss is None:
            raise ImportError("Please install faiss: pip install faiss-cpu")
        
        self.dimension = dimension
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.chunk_id_to_idx: Dict[str, int] = {}
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index with cosine similarity (via inner product on normalized vectors)."""
        # Using Inner Product index since our embeddings are normalized
        # This is equivalent to cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        logger.info(f"Initialized FAISS index with dimension {self.dimension}")
    
    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: np.ndarray
    ):
        """
        Add document chunks and their embeddings to the index.
        
        Args:
            chunks: List of DocumentChunk objects
            embeddings: Corresponding embedding matrix
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings")
        
        # Ensure embeddings are float32 and normalized
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # Add to index
        start_idx = len(self.chunks)
        self.index.add(embeddings)
        
        # Store chunks and mapping
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.chunk_id_to_idx[chunk.chunk_id] = start_idx + i
        
        logger.info(f"Added {len(chunks)} chunks to index. Total: {len(self.chunks)}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (chunk, score) tuples
        """
        if len(self.chunks) == 0:
            logger.warning("Empty index, no results")
            return []
        
        # Prepare query
        query = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        
        # Search
        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query, k)
        
        # Gather results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= threshold:
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def search_with_mmr(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        threshold: float = 0.0
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search with Maximal Marginal Relevance for diversity.
        
        MMR balances relevance and diversity to avoid returning
        near-duplicate chunks.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Final number of results
            fetch_k: Initial candidates to consider
            lambda_mult: Trade-off (1=pure relevance, 0=pure diversity)
            threshold: Minimum similarity threshold
            
        Returns:
            List of (chunk, score) tuples
        """
        if len(self.chunks) == 0:
            return []
        
        # Prepare query
        query = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        
        # Get initial candidates
        fetch_k = min(fetch_k, len(self.chunks))
        scores, indices = self.index.search(query, fetch_k)
        
        # Filter by threshold
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= threshold:
                candidates.append((idx, score))
        
        if not candidates:
            return []
        
        # MMR selection
        selected = []
        selected_embeddings = []
        
        while len(selected) < top_k and candidates:
            if not selected:
                # First selection: highest relevance
                best_idx = max(range(len(candidates)), key=lambda i: candidates[i][1])
            else:
                # MMR: balance relevance and diversity
                best_score = -float('inf')
                best_idx = 0
                
                for i, (idx, relevance) in enumerate(candidates):
                    # Get embedding for this candidate
                    candidate_emb = self._get_embedding_by_idx(idx)
                    
                    # Compute max similarity to already selected
                    if selected_embeddings:
                        selected_matrix = np.array(selected_embeddings)
                        similarities = np.dot(selected_matrix, candidate_emb)
                        max_sim = float(np.max(similarities))
                    else:
                        max_sim = 0.0
                    
                    # MMR score
                    mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = i
            
            # Add best candidate
            idx, score = candidates.pop(best_idx)
            selected.append((self.chunks[idx], score))
            selected_embeddings.append(self._get_embedding_by_idx(idx))
        
        return selected
    
    def _get_embedding_by_idx(self, idx: int) -> np.ndarray:
        """Reconstruct embedding from index."""
        return self.index.reconstruct(int(idx))
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get chunk by its ID."""
        idx = self.chunk_id_to_idx.get(chunk_id)
        if idx is not None:
            return self.chunks[idx]
        return None
    
    def save(self, directory: Path):
        """
        Save index and metadata to disk.
        
        Args:
            directory: Directory to save to
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = directory / "faiss.index"
        faiss.write_index(self.index, str(index_path))
        
        # Save chunks metadata
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        chunks_path = directory / "chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        # Save config
        config = {
            "dimension": self.dimension,
            "num_chunks": len(self.chunks)
        }
        config_path = directory / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved index to {directory}")
    
    @classmethod
    def load(cls, directory: Path) -> "FAISSVectorStore":
        """
        Load index and metadata from disk.
        
        Args:
            directory: Directory to load from
            
        Returns:
            Loaded FAISSVectorStore instance
        """
        directory = Path(directory)
        
        # Load config
        config_path = directory / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create instance
        store = cls(dimension=config["dimension"])
        
        # Load FAISS index
        index_path = directory / "faiss.index"
        store.index = faiss.read_index(str(index_path))
        
        # Load chunks
        chunks_path = directory / "chunks.json"
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        store.chunks = [DocumentChunk.from_dict(c) for c in chunks_data]
        store.chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(store.chunks)}
        
        logger.info(f"Loaded index with {len(store.chunks)} chunks from {directory}")
        return store
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        papers = set(c.paper_title for c in self.chunks)
        sections = set(c.section_name for c in self.chunks)
        
        return {
            "total_chunks": len(self.chunks),
            "num_papers": len(papers),
            "papers": list(papers),
            "sections": list(sections),
            "dimension": self.dimension,
            "index_type": type(self.index).__name__
        }
