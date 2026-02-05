"""
Configuration settings for the RAG system.
Supports multiple LLM providers: Gemini, Groq, Ollama
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
INDEX_DIR = DATA_DIR / "index"
TEMP_DIR = DATA_DIR / "temp"

# Create directories if they don't exist
for dir_path in [DATA_DIR, PAPERS_DIR, INDEX_DIR, TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    chunk_size: int = 400  # tokens
    chunk_overlap: int = 75  # tokens
    min_chunk_size: int = 100  # minimum tokens per chunk


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32


@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""
    top_k: int = 6
    similarity_threshold: float = 0.3
    use_mmr: bool = True  # Max Marginal Relevance
    mmr_diversity: float = 0.3  # Lambda for MMR (0 = max diversity, 1 = max relevance)
    section_boost: dict = field(default_factory=lambda: {
        "abstract": 1.2,
        "introduction": 1.1,
        "architecture": 1.3,
        "method": 1.25,
        "methodology": 1.25,
        "model": 1.25,
        "experiment": 1.0,
        "results": 1.1,
        "conclusion": 1.0,
        "related work": 0.8,
        "references": 0.5,
        "appendix": 0.7
    })


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: Literal["gemini", "groq", "ollama"] = "ollama"
    
    # Gemini settings
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    gemini_model: str = "gemini-1.5-flash"
    
    # Groq settings
    groq_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    groq_model: str = "llama-3.1-8b-instant"  # Fast and capable
    
    # Ollama settings (local)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:latest"  # Better reasoning for tables and complex content
    
    # Generation settings
    max_tokens: int = 1024
    temperature: float = 0.1  # Low for factual responses


@dataclass
class RAGConfig:
    """Main configuration combining all settings."""
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    # System prompts
    system_prompt: str = """You are a precise research assistant that answers questions based ONLY on the provided context from AI research papers.

RULES:
1. Answer ONLY using information from the provided context
2. If the answer is not in the context, say "I cannot find this information in the provided documents"
3. Be concise but thorough
4. Use technical terminology accurately
5. Never make up information or hallucinate facts"""

    answer_prompt_template: str = """Answer the question using ONLY the provided context from research papers.
If the information is not present in the context, explicitly state that.

CONTEXT:
{context}

QUESTION:
{question}

Provide a clear, accurate answer with proper technical details:"""


def get_config() -> RAGConfig:
    """Get the default configuration."""
    return RAGConfig()
