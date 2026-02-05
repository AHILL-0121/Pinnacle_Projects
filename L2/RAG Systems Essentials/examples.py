"""
Example Usage Script

Demonstrates programmatic usage of the RAG system.
Run this after installing dependencies and adding papers.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline, create_rag_pipeline
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingModel
from src.vector_store import FAISSVectorStore
from config import PAPERS_DIR, INDEX_DIR


def example_basic_usage():
    """Basic usage example with Ollama."""
    print("=" * 60)
    print("RAG System - Basic Usage Example")
    print("=" * 60)
    
    # Create pipeline with Ollama (local)
    pipeline = create_rag_pipeline(
        llm_provider="ollama",
        llm_config={
            "model": "mistral",  # or llama2, phi3
            "base_url": "http://localhost:11434"
        }
    )
    
    # Check if we have an existing index
    if (INDEX_DIR / "faiss.index").exists():
        print("Loading existing index...")
        pipeline.load_index(INDEX_DIR)
    else:
        # Ingest papers
        print(f"Ingesting papers from {PAPERS_DIR}...")
        pdf_files = list(PAPERS_DIR.glob("*.pdf"))
        
        if pdf_files:
            pipeline.ingest_documents(pdf_files)
            pipeline.save_index(INDEX_DIR)
            print(f"Indexed {len(pdf_files)} papers")
        else:
            print("No PDFs found. Add papers to data/papers/")
            return
    
    # Example questions
    questions = [
        "What is the Transformer architecture?",
        "How does multi-head attention work?",
        "What is positional encoding?",
        "What are the main components of RAG?",
    ]
    
    print("\n" + "=" * 60)
    print("Sample Questions")
    print("=" * 60)
    
    for question in questions:
        print(f"\n❓ {question}")
        
        try:
            response = pipeline.query(question)
            print(f"\n📝 Answer:\n{response.answer[:500]}...")
            print(f"\n📚 Sources:")
            for source in response.sources[:3]:
                print(f"   • {source}")
            print(f"\n⏱️ Total time: {response.total_time_ms:.0f}ms")
        except Exception as e:
            print(f"   Error: {e}")
        
        print("-" * 40)


def example_with_groq():
    """Example using Groq API for fast inference."""
    print("=" * 60)
    print("RAG System - Groq Example")
    print("=" * 60)
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set. Skipping Groq example.")
        return
    
    pipeline = create_rag_pipeline(
        llm_provider="groq",
        llm_config={
            "api_key": api_key,
            "model": "llama-3.1-8b-instant"  # Fast and capable
        },
        index_path=INDEX_DIR if (INDEX_DIR / "faiss.index").exists() else None
    )
    
    response = pipeline.query("Explain the attention mechanism in transformers")
    print(f"\n📝 Answer:\n{response.answer}")
    print(f"\n⏱️ Generation time: {response.generation_time_ms:.0f}ms")


def example_with_gemini():
    """Example using Gemini API."""
    print("=" * 60)
    print("RAG System - Gemini Example")
    print("=" * 60)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set. Skipping Gemini example.")
        return
    
    pipeline = create_rag_pipeline(
        llm_provider="gemini",
        llm_config={
            "api_key": api_key,
            "model": "gemini-1.5-flash"
        },
        index_path=INDEX_DIR if (INDEX_DIR / "faiss.index").exists() else None
    )
    
    response = pipeline.query("What is few-shot learning according to GPT-3?")
    print(f"\n📝 Answer:\n{response.answer}")
    print(f"\n⏱️ Generation time: {response.generation_time_ms:.0f}ms")


def example_document_processing():
    """Example of document processing only."""
    print("=" * 60)
    print("Document Processing Example")
    print("=" * 60)
    
    processor = DocumentProcessor(chunk_size=400, overlap=75)
    
    pdf_files = list(PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found. Add papers to data/papers/")
        return
    
    # Process first PDF
    pdf_path = pdf_files[0]
    print(f"Processing: {pdf_path.name}")
    
    chunks = processor.process_pdf(pdf_path)
    
    print(f"\nCreated {len(chunks)} chunks")
    print("\nFirst chunk:")
    print("-" * 40)
    print(f"Paper: {chunks[0].paper_title}")
    print(f"Section: {chunks[0].section_name}")
    print(f"Pages: {chunks[0].page_numbers}")
    print(f"Text preview: {chunks[0].text[:200]}...")


def example_retrieval_only():
    """Example of retrieval without LLM generation."""
    print("=" * 60)
    print("Retrieval-Only Example")
    print("=" * 60)
    
    # Load embedding model
    embedding_model = EmbeddingModel()
    
    # Load index
    if not (INDEX_DIR / "faiss.index").exists():
        print("No index found. Run example_basic_usage() first.")
        return
    
    vector_store = FAISSVectorStore.load(INDEX_DIR)
    
    # Create retriever
    from src.retriever import Retriever
    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=5,
        use_mmr=True
    )
    
    # Retrieve for a query
    query = "What is self-attention?"
    results = retriever.retrieve(query)
    
    print(f"\nQuery: {query}")
    print(f"\nTop {len(results)} results:")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result.score:.3f} (boosted: {result.boosted_score:.3f})")
        print(f"   Source: {result.chunk.paper_title}, {result.chunk.section_name}")
        print(f"   Text: {result.chunk.text[:150]}...")


if __name__ == "__main__":
    # Run basic example
    example_basic_usage()
    
    # Uncomment to run other examples:
    # example_document_processing()
    # example_retrieval_only()
    # example_with_groq()
    # example_with_gemini()
