"""
RAG Research Assistant - Main Entry Point

A portfolio-grade Retrieval-Augmented Generation system for
question answering over AI research papers.

Usage:
    python main.py                          # Interactive mode
    python main.py --provider ollama        # Use specific provider
    python main.py --ingest                 # Ingest papers on startup
    python main.py -q "What is attention?"  # Single question mode
"""

from cli import main

if __name__ == "__main__":
    main()
