"""
Command Line Interface for RAG System

Provides interactive Q&A over research papers with:
- Multiple LLM provider selection (Gemini, Groq, Ollama)
- Document ingestion
- Source-attributed answers
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import PAPERS_DIR, INDEX_DIR, get_config
from src.rag_pipeline import RAGPipeline, create_rag_pipeline
from src.llm_providers import LLMManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGCli:
    """Interactive CLI for the RAG system."""
    
    BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║                    RAG Research Assistant                        ║
║          Question Answering over AI Research Papers              ║
╠══════════════════════════════════════════════════════════════════╣
║  Commands:                                                       ║
║    /help      - Show this help                                   ║
║    /ingest    - Ingest PDFs from papers directory                ║
║    /stats     - Show index statistics                            ║
║    /provider  - Change LLM provider (gemini/groq/ollama)         ║
║    /clear     - Clear screen                                     ║
║    /quit      - Exit the application                             ║
╚══════════════════════════════════════════════════════════════════╝
"""
    
    def __init__(self):
        self.pipeline: Optional[RAGPipeline] = None
        self.config = get_config()
    
    def setup_llm_provider(self, provider: str) -> bool:
        """Setup and validate LLM provider."""
        if provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("\n⚠️  GEMINI_API_KEY not found in environment variables.")
                api_key = input("Enter your Gemini API key (or press Enter to skip): ").strip()
                if not api_key:
                    return False
            
            success = self.pipeline.llm_manager.setup_gemini(
                api_key=api_key,
                model=self.config.llm.gemini_model
            )
            if success:
                self.pipeline.llm_manager.set_active_provider("gemini")
                print(f"✓ Gemini configured with model: {self.config.llm.gemini_model}")
            return success
        
        elif provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                print("\n⚠️  GROQ_API_KEY not found in environment variables.")
                api_key = input("Enter your Groq API key (or press Enter to skip): ").strip()
                if not api_key:
                    return False
            
            success = self.pipeline.llm_manager.setup_groq(
                api_key=api_key,
                model=self.config.llm.groq_model
            )
            if success:
                self.pipeline.llm_manager.set_active_provider("groq")
                print(f"✓ Groq configured with model: {self.config.llm.groq_model}")
            return success
        
        elif provider == "ollama":
            success = self.pipeline.llm_manager.setup_ollama(
                base_url=self.config.llm.ollama_base_url,
                model=self.config.llm.ollama_model
            )
            if success:
                self.pipeline.llm_manager.set_active_provider("ollama")
                print(f"✓ Ollama configured with model: {self.config.llm.ollama_model}")
                
                # List available models
                available = self.pipeline.llm_manager._providers["ollama"].list_models()
                if available:
                    print(f"  Available models: {', '.join(available[:5])}")
            else:
                print(f"\n⚠️  Ollama not available at {self.config.llm.ollama_base_url}")
                print("    Make sure Ollama is running: ollama serve")
            return success
        
        return False
    
    def select_llm_provider(self) -> bool:
        """Interactive LLM provider selection."""
        print("\n" + "="*60)
        print("Select LLM Provider:")
        print("="*60)
        print("  1. Ollama   (Local, free, requires Ollama installed)")
        print("  2. Groq     (Cloud, fast, requires API key)")
        print("  3. Gemini   (Cloud, capable, requires API key)")
        print("="*60)
        
        while True:
            choice = input("\nEnter choice [1-3]: ").strip()
            
            if choice == "1":
                return self.setup_llm_provider("ollama")
            elif choice == "2":
                return self.setup_llm_provider("groq")
            elif choice == "3":
                return self.setup_llm_provider("gemini")
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
    
    def initialize(self, provider: Optional[str] = None):
        """Initialize the RAG pipeline."""
        print("\n🔄 Initializing RAG pipeline...")
        
        # Create pipeline
        self.pipeline = RAGPipeline()
        
        # Check for existing index
        index_exists = INDEX_DIR.exists() and (INDEX_DIR / "faiss.index").exists()
        
        if index_exists:
            print(f"📂 Loading existing index from {INDEX_DIR}")
            self.pipeline.initialize(index_path=INDEX_DIR)
        else:
            print("📂 No existing index found. Creating new index...")
            self.pipeline.initialize()
        
        # Setup LLM provider
        if provider:
            success = self.setup_llm_provider(provider)
            if not success:
                print("\n⚠️  Failed to setup requested provider. Selecting interactively...")
                self.select_llm_provider()
        else:
            self.select_llm_provider()
        
        print("\n✓ RAG pipeline ready!")
    
    def ingest_papers(self):
        """Ingest PDFs from papers directory."""
        print(f"\n📄 Looking for PDFs in: {PAPERS_DIR}")
        
        if not PAPERS_DIR.exists():
            PAPERS_DIR.mkdir(parents=True, exist_ok=True)
            print(f"   Created directory: {PAPERS_DIR}")
        
        pdf_files = list(PAPERS_DIR.glob("*.pdf"))
        
        if not pdf_files:
            print("\n⚠️  No PDF files found!")
            print(f"   Please add research paper PDFs to: {PAPERS_DIR}")
            print("\n   Suggested papers:")
            print("   - attention_is_all_you_need.pdf (Transformer)")
            print("   - rag_paper.pdf (RAG)")
            print("   - gpt3_paper.pdf (GPT-3)")
            return
        
        print(f"\n   Found {len(pdf_files)} PDF(s):")
        for pdf in pdf_files:
            print(f"   • {pdf.name}")
        
        confirm = input("\n   Proceed with ingestion? [Y/n]: ").strip().lower()
        if confirm == 'n':
            return
        
        print("\n🔄 Processing documents...")
        num_chunks = self.pipeline.ingest_documents(pdf_files)
        
        # Save index
        self.pipeline.save_index(INDEX_DIR)
        print(f"\n✓ Indexed {num_chunks} chunks")
        print(f"✓ Index saved to {INDEX_DIR}")
    
    def show_stats(self):
        """Display index statistics."""
        stats = self.pipeline.get_stats()
        
        print("\n" + "="*50)
        print("Index Statistics")
        print("="*50)
        print(f"  Total chunks:     {stats.get('total_chunks', 0)}")
        print(f"  Number of papers: {stats.get('num_papers', 0)}")
        print(f"  Embedding model:  {stats.get('embedding_model', 'N/A')}")
        print(f"  Embedding dim:    {stats.get('embedding_dimension', 'N/A')}")
        print(f"  LLM providers:    {', '.join(stats.get('llm_providers', []))}")
        
        if stats.get('papers'):
            print("\n  Papers indexed:")
            for paper in stats['papers']:
                print(f"    • {paper}")
        
        print("="*50)
    
    def change_provider(self):
        """Change the active LLM provider."""
        self.select_llm_provider()
    
    def ask_question(self, question: str):
        """Process a user question."""
        if not self.pipeline.vector_store or len(self.pipeline.vector_store.chunks) == 0:
            print("\n⚠️  No documents indexed. Use /ingest to add papers first.")
            return
        
        try:
            available = self.pipeline.llm_manager.list_available_providers()
            if not available:
                print("\n⚠️  No LLM provider available. Use /provider to configure one.")
                return
        except Exception:
            print("\n⚠️  No LLM provider configured. Use /provider to configure one.")
            return
        
        print("\n🔍 Searching documents...")
        
        try:
            response = self.pipeline.query(question)
            
            print("\n" + "="*60)
            print("📝 ANSWER")
            print("="*60)
            print(response.answer)
            
            print("\n" + "-"*60)
            print("📚 SOURCES")
            print("-"*60)
            for source in response.sources:
                print(f"  • {source}")
            
            print("\n" + "-"*60)
            print(f"⏱️  Retrieval: {response.retrieval_time_ms:.0f}ms | "
                  f"Generation: {response.generation_time_ms:.0f}ms | "
                  f"Total: {response.total_time_ms:.0f}ms")
            print(f"🤖 Provider: {response.llm_provider}/{response.llm_model}")
            print("="*60)
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            print(f"\n❌ Error: {e}")
    
    def run(self, provider: Optional[str] = None):
        """Run the interactive CLI."""
        print(self.BANNER)
        
        self.initialize(provider)
        
        print("\n💡 Ask any question about the indexed research papers.")
        print("   Type /help for available commands.\n")
        
        while True:
            try:
                user_input = input("\n❓ You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    cmd = user_input.lower().split()[0]
                    
                    if cmd == '/quit' or cmd == '/exit':
                        print("\n👋 Goodbye!")
                        break
                    elif cmd == '/help':
                        print(self.BANNER)
                    elif cmd == '/ingest':
                        self.ingest_papers()
                    elif cmd == '/stats':
                        self.show_stats()
                    elif cmd == '/provider':
                        self.change_provider()
                    elif cmd == '/clear':
                        os.system('cls' if os.name == 'nt' else 'clear')
                    else:
                        print(f"Unknown command: {cmd}. Type /help for available commands.")
                else:
                    self.ask_question(user_input)
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\n❌ Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Research Assistant - Question Answering over AI Papers"
    )
    parser.add_argument(
        '--provider', '-p',
        choices=['gemini', 'groq', 'ollama'],
        help='LLM provider to use'
    )
    parser.add_argument(
        '--ingest', '-i',
        action='store_true',
        help='Ingest papers from data/papers directory'
    )
    parser.add_argument(
        '--question', '-q',
        type=str,
        help='Ask a single question and exit'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    cli = RAGCli()
    
    if args.question:
        # Single question mode
        cli.initialize(args.provider)
        if args.ingest:
            cli.ingest_papers()
        cli.ask_question(args.question)
    else:
        # Interactive mode
        cli.run(args.provider)


if __name__ == "__main__":
    main()
