"""
Retrieval Module

Implements dense semantic retrieval with section-aware boosting,
Maximal Marginal Relevance (MMR) for diversity, and dual retrieval
strategy for factual vs conceptual queries.
"""

import logging
import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np

from .document_processor import DocumentChunk
from .embeddings import EmbeddingModel
from .vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)

# Keywords that indicate factual/numeric queries requiring exact matching
FACTUAL_KEYWORDS = [
    "table", "ratio", "percentage", "%", "number", "value", "score",
    "accuracy", "bleu", "rouge", "f1", "precision", "recall",
    "how many", "how much", "what is the", "exact", "specific"
]

# Keywords indicating table-specific queries (need exact table data)
TABLE_KEYWORDS = ["table", "figure", "chart", "row", "column", "cell"]

# Keywords indicating section-specific queries
SECTION_KEYWORDS = ["section", "which section", "introduced in", "defined in", "where is"]


@dataclass
class RetrievalResult:
    """Result from retrieval with score and metadata."""
    chunk: DocumentChunk
    score: float
    boosted_score: float
    
    def format_context(self) -> str:
        """Format chunk as context for LLM with specific page."""
        # Use only the first page number for precise citation
        page = self.chunk.page_numbers[0] if self.chunk.page_numbers else "?"
        return f"""[Source: {self.chunk.paper_title}, {self.chunk.section_name}, Page {page}]
{self.chunk.text}"""
    
    def format_citation(self) -> str:
        """Format as precise citation with single page."""
        # FIXED: Only cite the specific page, not ranges
        page = self.chunk.page_numbers[0] if self.chunk.page_numbers else "?"
        return f"{self.chunk.paper_title}, {self.chunk.section_name}, Page {page}"


def is_factual_query(query: str) -> bool:
    """
    Detect if query requires factual/numeric precision vs conceptual understanding.
    
    Section queries (e.g., "which section introduces X") are NOT factual -
    they ask about document structure, not numeric data.
    
    Args:
        query: User's question
        
    Returns:
        True if query needs exact facts (tables, numbers, ratios)
    """
    query_lower = query.lower()
    
    # Section queries are NOT factual (they're about structure)
    if is_section_query(query):
        return False
    
    return any(kw in query_lower for kw in FACTUAL_KEYWORDS)


def is_table_query(query: str) -> bool:
    """Detect if query specifically asks about table data."""
    query_lower = query.lower()
    # Only table if asking for data, not just "which section"
    if is_section_query(query):
        return False
    return any(kw in query_lower for kw in TABLE_KEYWORDS)


def is_section_query(query: str) -> bool:
    """Detect if query asks about a specific section."""
    query_lower = query.lower()
    return any(kw in query_lower for kw in SECTION_KEYWORDS)


def extract_target_paper(query: str) -> Optional[str]:
    """
    Extract the target paper from query for citation filtering.
    
    Returns partial paper title if mentioned, else None.
    """
    query_lower = query.lower()
    
    # Known paper patterns
    paper_patterns = [
        (r"transformer\s*paper|attention\s*is\s*all", "Attention Is All You Need"),
        (r"rag\s*paper|retrieval[- ]augmented", "Retrieval-Augmented Generation"),
        (r"gpt[- ]?3\s*paper|few[- ]shot", "Language Models are Few-Shot Learners"),
    ]
    
    for pattern, paper_name in paper_patterns:
        if re.search(pattern, query_lower):
            return paper_name
    
    return None


class Retriever:
    """
    Dense semantic retriever for RAG system.
    
    Implements bi-encoder style retrieval (similar to DPR) with:
    - Dense vector similarity search
    - Maximal Marginal Relevance for diversity
    - Section-aware score boosting
    - Dual retrieval strategy for factual queries
    """
    
    # Higher confidence threshold for reliable answers
    CONFIDENCE_THRESHOLD = 0.50  # Below this, answer is unreliable
    FACTUAL_CONFIDENCE_THRESHOLD = 0.75  # Much higher bar for table/numeric queries (strict)
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: FAISSVectorStore,
        top_k: int = 6,
        use_mmr: bool = True,
        mmr_lambda: float = 0.7,
        similarity_threshold: float = 0.3,
        section_boost: Optional[Dict[str, float]] = None
    ):
        """
        Initialize retriever.
        
        Args:
            embedding_model: Model for query embedding
            vector_store: Vector store with indexed chunks
            top_k: Number of results to return
            use_mmr: Whether to use MMR for diversity
            mmr_lambda: MMR trade-off (higher = more relevance)
            similarity_threshold: Minimum similarity score
            section_boost: Section name -> boost multiplier mapping
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        self.similarity_threshold = similarity_threshold
        self.section_boost = section_boost or {
            "abstract": 1.2,
            "introduction": 1.1,
            "architecture": 1.3,
            "method": 1.25,
            "methodology": 1.25,
            "model": 1.25,
            "attention": 1.3,
            "training": 1.1,
            "experiments": 1.15,
            "results": 1.2,
            "tables": 1.4,  # Higher boost for tables
            "analysis": 1.1,
            "conclusion": 1.0,
            "related work": 0.8,
            "references": 0.5,
            "appendix": 0.7,
            "unknown": 0.9
        }
    
    def get_confidence(self, results: List['RetrievalResult'], query: str) -> float:
        """
        Calculate retrieval confidence score.
        
        Args:
            results: Retrieved chunks
            query: Original query
            
        Returns:
            Confidence score 0-1
        """
        if not results:
            return 0.0
        
        # Use top result's score as base confidence
        # For section queries, use boosted_score since section boost is meaningful
        if is_section_query(query):
            top_score = results[0].boosted_score if results else 0.0
        else:
            top_score = results[0].score if results else 0.0
        
        # For factual queries, check if content actually contains expected data
        if is_factual_query(query):
            # Look for numeric content in top results
            has_numbers = any(
                bool(re.search(r'\d+\.?\d*%?', r.chunk.text[:500]))
                for r in results[:3]
            )
            if not has_numbers:
                top_score *= 0.7  # Penalize if no numbers found for factual query
        
        # For table queries, apply extra penalty if no table content found
        if is_table_query(query):
            has_table_content = self._has_exact_table_chunk(query, results)
            if not has_table_content:
                top_score *= 0.5  # Heavy penalty - don't trust without exact table data
        
        return top_score
    
    def _has_exact_table_chunk(self, query: str, results: List['RetrievalResult']) -> bool:
        """
        Check if we have a chunk with EXACT table data for this query.
        
        Critical for table queries - prevents hallucination.
        """
        # Extract table number from query
        table_match = re.search(r'table\s*(\d+)', query.lower())
        if not table_match:
            return True  # Not a numbered table query
        
        table_num = table_match.group(1)
        
        # Check if any result contains exact table data
        for r in results[:4]:
            text_lower = r.chunk.text.lower()
            # Look for table N with actual data (percentages, numbers)
            if f"table {table_num}" in text_lower:
                # Must have numeric values nearby (not just a reference)
                if re.search(rf'table\s*{table_num}.*?(\d+\.?\d*%|\d+\.\d+)', text_lower, re.DOTALL):
                    return True
        
        return False
    
    def is_confident(self, results: List['RetrievalResult'], query: str) -> bool:
        """
        Check if retrieval confidence meets threshold.
        
        Args:
            results: Retrieved chunks
            query: Original query
            
        Returns:
            True if confident enough to answer
        """
        confidence = self.get_confidence(results, query)
        threshold = (self.FACTUAL_CONFIDENCE_THRESHOLD 
                    if is_factual_query(query) 
                    else self.CONFIDENCE_THRESHOLD)
        return confidence >= threshold
    
    def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.
        
        Uses dual strategy:
        - Dense retrieval for all queries
        - Keyword boosting for factual/numeric queries
        
        Args:
            query: Natural language query
            
        Returns:
            List of RetrievalResult objects, sorted by boosted score
        """
        logger.info(f"Retrieving for query: {query[:50]}...")
        
        # Detect query type for dual strategy
        factual_query = is_factual_query(query)
        section_query = is_section_query(query)
        
        if factual_query:
            logger.info("Factual query detected - using keyword boosting")
        if section_query:
            logger.info("Section query detected - using direct search with section boosting (no MMR)")
        
        # Embed query
        query_embedding = self.embedding_model.embed_query(query)
        
        # For section queries: skip MMR and fetch more candidates to ensure section-labeled chunks are included
        # MMR can filter out important section chunks if they're similar to other content
        if section_query:
            # Direct search without MMR for section queries
            raw_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=self.top_k * 6,  # Fetch many more to find section-labeled chunks
                threshold=self.similarity_threshold
            )
        elif self.use_mmr:
            fetch_multiplier = 4 if factual_query else 3
            raw_results = self.vector_store.search_with_mmr(
                query_embedding=query_embedding,
                top_k=self.top_k,
                fetch_k=self.top_k * fetch_multiplier,
                lambda_mult=self.mmr_lambda,
                threshold=self.similarity_threshold
            )
        else:
            raw_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=self.top_k * 2,  # Fetch more for re-ranking
                threshold=self.similarity_threshold
            )
        
        # Apply section boosting
        results = []
        for chunk, score in raw_results:
            section_lower = chunk.section_name.lower()
            boost = self.section_boost.get(section_lower, 1.0)
            boosted_score = score * boost
            
            results.append(RetrievalResult(
                chunk=chunk,
                score=score,
                boosted_score=boosted_score
            ))
        
        # DUAL STRATEGY: Apply keyword boosting for factual queries
        if factual_query:
            results = self._apply_keyword_boost(query, results)
        
        # Boost chunks with explicit section numbers for section queries
        if section_query:
            results = self._apply_section_boost(query, results)
        
        # Re-sort by boosted score
        results.sort(key=lambda r: r.boosted_score, reverse=True)
        
        # Trim to top_k
        results = results[:self.top_k]
        
        logger.info(f"Retrieved {len(results)} chunks (confidence: {self.get_confidence(results, query):.3f})")
        return results
    
    def _apply_keyword_boost(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Apply keyword matching boost for factual queries.
        
        Args:
            query: Original query
            results: Retrieved results
            
        Returns:
            Results with keyword boost applied
        """
        # Extract important keywords from query
        query_lower = query.lower()
        
        # Look for table references
        table_match = re.search(r'table\s*(\d+)', query_lower)
        table_num = table_match.group(1) if table_match else None
        
        # Extract numeric terms and key phrases
        keywords = set()
        for word in query.split():
            word_clean = word.lower().strip('.,?!')
            if len(word_clean) > 2:
                keywords.add(word_clean)
        
        for result in results:
            text_lower = result.chunk.text.lower()
            boost = 0.0
            
            # Strong boost if specific table number is mentioned and found
            if table_num and f"table {table_num}" in text_lower:
                boost += 0.15
                logger.debug(f"Table {table_num} found in chunk")
            
            # Boost for keyword matches
            keyword_matches = sum(1 for kw in keywords if kw in text_lower)
            boost += 0.02 * keyword_matches
            
            # Boost for numeric content in factual queries
            if re.search(r'\d+\.?\d*%', text_lower):
                boost += 0.05
            
            result.boosted_score += boost
        
        return results
    
    def _apply_section_boost(self, query: str, results: List['RetrievalResult']) -> List['RetrievalResult']:
        """
        Boost chunks that have explicit section numbers for section queries.
        
        Args:
            query: Original query
            results: Retrieved results
            
        Returns:
            Results with section boost applied
        """
        query_lower = query.lower()
        
        for result in results:
            boost = 0.0
            text = result.chunk.text
            section_name = result.chunk.section_name
            
            # Strong boost for chunks with explicit section number in section_name
            if section_name.startswith("Section"):
                boost += 0.15
            
            # Extra boost if text has section number prefix
            if text.startswith("[Section"):
                boost += 0.10
            
            # Boost if looking for specific concept and chunk has it in section header
            # Extract key concept from query
            concepts = ["multi-head attention", "attention", "encoder", "decoder", "scaled dot-product"]
            for concept in concepts:
                if concept in query_lower and concept in section_name.lower():
                    boost += 0.20
                    break
            
            result.boosted_score += boost
        
        return results
    
    def retrieve_with_filter(
        self,
        query: str,
        paper_filter: Optional[List[str]] = None,
        section_filter: Optional[List[str]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve with optional paper/section filters.
        
        Args:
            query: Natural language query
            paper_filter: List of paper titles to include
            section_filter: List of section names to include
            
        Returns:
            Filtered list of RetrievalResult objects
        """
        results = self.retrieve(query)
        
        if paper_filter:
            paper_filter_lower = [p.lower() for p in paper_filter]
            results = [
                r for r in results
                if r.chunk.paper_title.lower() in paper_filter_lower
            ]
        
        if section_filter:
            section_filter_lower = [s.lower() for s in section_filter]
            results = [
                r for r in results
                if r.chunk.section_name.lower() in section_filter_lower
            ]
        
        return results
    
    def format_context(self, results: List[RetrievalResult]) -> str:
        """
        Format retrieval results as context string for LLM.
        
        Args:
            results: List of RetrievalResult objects
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant context found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"--- Context {i} ---")
            context_parts.append(result.format_context())
        
        return "\n\n".join(context_parts)
    
    def format_citations(self, results: List[RetrievalResult], query: str = "", max_citations: int = 2) -> List[str]:
        """
        Extract filtered citations from results.
        
        CITATION DISCIPLINE:
        - Max 2 citations for clarity
        - Filter to relevant paper if query mentions a specific paper
        - For "No" answers, only cite the authoritative source
        
        Args:
            results: List of RetrievalResult objects
            query: Original query (for paper filtering)
            max_citations: Maximum citations to return (default 2)
            
        Returns:
            List of unique, filtered citation strings
        """
        if not results:
            return []
        
        # Check if query targets a specific paper
        target_paper = extract_target_paper(query)
        
        seen = set()
        citations = []
        
        for result in results:
            # If query targets a specific paper, only cite from that paper
            if target_paper:
                if target_paper.lower() not in result.chunk.paper_title.lower():
                    continue
            
            citation = result.format_citation()
            if citation not in seen:
                seen.add(citation)
                citations.append(citation)
            
            # Enforce max citations
            if len(citations) >= max_citations:
                break
        
        # If no matching citations found for targeted query, return top 1-2 anyway
        if not citations and results:
            for result in results[:max_citations]:
                citation = result.format_citation()
                if citation not in seen:
                    seen.add(citation)
                    citations.append(citation)
        
        return citations


class HybridRetriever(Retriever):
    """
    Extended retriever with keyword-based re-ranking.
    
    Combines dense retrieval with optional keyword matching
    for better precision on technical terms.
    """
    
    def __init__(self, *args, keyword_boost: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.keyword_boost = keyword_boost
    
    def retrieve(self, query: str) -> List[RetrievalResult]:
        """Retrieve with keyword boosting."""
        # Get base results
        results = super().retrieve(query)
        
        # Extract query keywords (simple tokenization)
        query_keywords = set(
            word.lower() for word in query.split()
            if len(word) > 3
        )
        
        # Boost results containing query keywords
        for result in results:
            text_lower = result.chunk.text.lower()
            keyword_matches = sum(1 for kw in query_keywords if kw in text_lower)
            
            if keyword_matches > 0:
                result.boosted_score += self.keyword_boost * keyword_matches
        
        # Re-sort
        results.sort(key=lambda r: r.boosted_score, reverse=True)
        
        return results
