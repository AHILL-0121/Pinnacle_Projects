"""
AI Web Research Agent - ReAct Pattern Implementation
=====================================================

This module implements an AI-powered web research agent using the ReAct
(Reason + Act) design pattern. It orchestrates LLM reasoning with web
search tools to produce comprehensive research reports.

Architecture:
    User Topic → Planner (LLM) → Actor (Web Search) → Reasoner (LLM) → Report

Author: AHILL S
Version: 1.0.0
"""

import os
import json
import logging
import webbrowser
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Rich console for beautiful output
console = Console()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SearchResult:
    """Represents a single search result from web search."""
    title: str
    url: str
    content: str
    score: float = 0.0


@dataclass
class QuestionResults:
    """Holds search results for a single research question."""
    question: str
    results: list[SearchResult] = field(default_factory=list)
    synthesis: str = ""


@dataclass
class ResearchReport:
    """Final research report structure."""
    topic: str
    timestamp: str
    questions: list[str] = field(default_factory=list)
    sections: list[QuestionResults] = field(default_factory=list)
    introduction: str = ""
    conclusion: str = ""


# =============================================================================
# LLM Provider Abstraction
# =============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate text from prompt."""
        pass


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider using new google.genai SDK with retry logic."""
    
    def __init__(self):
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        self.client = genai.Client(api_key=api_key)
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.max_retries = 3
        self.base_delay = 10  # seconds
        logger.info(f"Initialized Gemini provider with model: {self.model_name}")
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        import time
        from google.genai.errors import ClientError
        
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                    config={"max_output_tokens": 800}
                )
                return response.text
            except ClientError as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait_time = self.base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}")
                    console.print(f"[yellow]⏳ Rate limited. Waiting {wait_time}s...[/yellow]")
                    time.sleep(wait_time)
                    if attempt == self.max_retries - 1:
                        raise ValueError(
                            f"Gemini rate limit exceeded after {self.max_retries} retries. "
                            "Options:\n"
                            "  1. Wait a few minutes and try again\n"
                            "  2. Use Ollama: --provider ollama\n"
                            "  3. Use Groq: --provider groq (requires GROQ_API_KEY)"
                        ) from e
                else:
                    raise


class GroqProvider(LLMProvider):
    """Groq LLM provider for fast inference."""
    
    def __init__(self):
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.client = Groq(api_key=api_key)
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        logger.info(f"Initialized Groq provider with model: {self.model}")
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        return response.choices[0].message.content


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""
    
    def __init__(self):
        import ollama
        self.client = ollama
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2")
        logger.info(f"Initialized Ollama provider with model: {self.model}")
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat(model=self.model, messages=messages)
        return response['message']['content']


class OpenRouterProvider(LLMProvider):
    """
    OpenRouter LLM provider - Access multiple models through one API.
    
    Popular models available:
    - anthropic/claude-3.5-sonnet (recommended)
    - openai/gpt-4o
    - meta-llama/llama-3.1-70b-instruct
    - google/gemini-pro-1.5
    - mistralai/mistral-large
    - deepseek/deepseek-chat
    
    Get API key at: https://openrouter.ai/keys
    """
    
    def __init__(self):
        import requests
        self.requests = requests
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        self.api_key = api_key
        self.model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        logger.info(f"Initialized OpenRouter provider with model: {self.model}")
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ai-research-agent",
            "X-Title": "AI Research Agent"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 800
        }
        
        response = self.requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=120
        )
        
        if response.status_code != 200:
            error_msg = response.json().get("error", {}).get("message", response.text)
            raise ValueError(f"OpenRouter API error: {error_msg}")
        
        return response.json()["choices"][0]["message"]["content"]


def get_llm_provider() -> LLMProvider:
    """Factory function to get configured LLM provider."""
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    providers = {
        "gemini": GeminiProvider,
        "groq": GroqProvider,
        "ollama": OllamaProvider,
        "openrouter": OpenRouterProvider
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown LLM provider: {provider}. Choose from: {list(providers.keys())}")
    
    return providers[provider]()


# =============================================================================
# Web Search Tool
# =============================================================================

class WebSearchTool:
    """Tavily-based web search tool for the Acting phase."""
    
    def __init__(self):
        from tavily import TavilyClient
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")
        self.client = TavilyClient(api_key=api_key)
        self.max_results = int(os.getenv("MAX_SEARCH_RESULTS", "3"))  # Reduced for token efficiency
        logger.info("Initialized Tavily search client")
    
    def search(self, query: str) -> list[SearchResult]:
        """Execute web search and return structured results."""
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=self.max_results
            )
            
            results = []
            for item in response.get("results", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    score=item.get("score", 0.0)
                ))
            
            logger.info(f"Search for '{query[:50]}...' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []


# =============================================================================
# ReAct Agent Prompts (Optimized for free tier - minimal tokens)
# =============================================================================

PLANNING_SYSTEM_PROMPT = """Generate 4 focused research questions for the topic. Cover: causes, effects, trends, solutions. Return ONLY a JSON array of strings.
Example: ["Q1?", "Q2?", "Q3?", "Q4?"]"""

SYNTHESIS_SYSTEM_PROMPT = """Synthesize the search results into a brief, factual summary (150-200 words).
Format as markdown bullet points using this EXACT format:
- First key finding
- Second key finding
- Third key finding

Use dash (-) for bullets, one finding per line. Only include facts from sources."""

INTRODUCTION_SYSTEM_PROMPT = """Write a brief 1-paragraph introduction for this research topic. Be concise."""

CONCLUSION_SYSTEM_PROMPT = """Write a brief 1-paragraph conclusion summarizing key findings. Be concise."""


# =============================================================================
# ReAct Research Agent
# =============================================================================

class ResearchAgent:
    """
    AI Web Research Agent implementing the ReAct pattern.
    
    The ReAct (Reason + Act) pattern alternates between:
    1. REASONING: Using LLM to think, plan, and synthesize
    2. ACTING: Using tools (web search) to gather information
    
    Flow:
        Topic → Plan (Reason) → Search (Act) → Synthesize (Reason) → Report
    """
    
    def __init__(self):
        console.print(Panel("🔬 Initializing AI Research Agent", style="bold blue"))
        self.llm = get_llm_provider()
        self.search_tool = WebSearchTool()
        self.max_questions = int(os.getenv("MAX_QUESTIONS", "4"))  # Reduced for token efficiency
        console.print("✅ Agent initialized successfully\n", style="green")
    
    def plan(self, topic: str) -> list[str]:
        """
        REASONING PHASE: Generate research questions for the topic.
        
        This is the planning step where the LLM reasons about what
        information is needed to comprehensively cover the topic.
        """
        console.print(Panel(f"🧠 PLANNING: Generating research questions for '{topic}'", style="bold yellow"))
        
        prompt = f"""Topic: {topic}

Generate {self.max_questions} research questions covering causes, effects, trends, solutions.
Return ONLY a JSON array: ["Q1?", "Q2?", ...]"""

        try:
            response = self.llm.generate(prompt, PLANNING_SYSTEM_PROMPT)
            
            # Parse JSON from response
            # Handle potential markdown code blocks
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            response = response.strip()
            
            questions = json.loads(response)
            
            # Ensure we have the right number of questions
            questions = questions[:self.max_questions]
            
            console.print(f"✅ Generated {len(questions)} research questions:\n", style="green")
            for i, q in enumerate(questions, 1):
                console.print(f"  {i}. {q}")
            console.print()
            
            return questions
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse questions JSON: {e}")
            # Fallback: generate simple questions
            return [
                f"What are the main causes of {topic}?",
                f"What are the current trends in {topic}?",
                f"What is the impact of {topic}?",
                f"What solutions exist for {topic}?",
                f"What is the future outlook for {topic}?"
            ]
    
    def act(self, questions: list[str]) -> list[QuestionResults]:
        """
        ACTING PHASE: Execute web searches for each question.
        
        This is where the agent uses tools to gather real-world
        information from the web.
        """
        console.print(Panel("🔍 ACTING: Searching the web for information", style="bold cyan"))
        
        all_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for question in questions:
                task = progress.add_task(f"Searching: {question[:50]}...", total=None)
                
                results = self.search_tool.search(question)
                
                question_results = QuestionResults(
                    question=question,
                    results=results
                )
                all_results.append(question_results)
                
                progress.remove_task(task)
        
        console.print(f"✅ Completed {len(all_results)} web searches\n", style="green")
        return all_results
    
    def reason(self, question_results: QuestionResults) -> str:
        """
        REASONING PHASE: Synthesize search results into coherent insights.
        
        The LLM analyzes the raw search results and creates a
        factual, well-organized summary.
        """
        if not question_results.results:
            return "No search results found for this question."
        
        # Format search results for the LLM (truncate content for token efficiency)
        results_text = ""
        for i, result in enumerate(question_results.results, 1):
            # Truncate content to ~200 chars to save tokens
            content = result.content[:300] + "..." if len(result.content) > 300 else result.content
            results_text += f"\n[{i}] {result.title}\n{content}\n"
        
        prompt = f"""Q: {question_results.question}\n\nSources:{results_text}\nSummarize key facts (100-150 words, bullet points)."""

        synthesis = self.llm.generate(prompt, SYNTHESIS_SYSTEM_PROMPT)
        return synthesis
    
    def synthesize_all(self, results: list[QuestionResults]) -> list[QuestionResults]:
        """Synthesize results for all questions."""
        console.print(Panel("📊 REASONING: Synthesizing search results", style="bold magenta"))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for qr in results:
                task = progress.add_task(f"Synthesizing: {qr.question[:40]}...", total=None)
                qr.synthesis = self.reason(qr)
                progress.remove_task(task)
        
        console.print("✅ Synthesis complete\n", style="green")
        return results
    
    def generate_introduction(self, topic: str, questions: list[str]) -> str:
        """Generate report introduction."""
        questions_text = "\n".join(f"- {q}" for q in questions)
        prompt = f"""Write an introduction for a research report on the following topic:

TOPIC: {topic}

Aspects covered: {questions_text}

Write a brief intro (50-80 words)."""

        return self.llm.generate(prompt, INTRODUCTION_SYSTEM_PROMPT)
    
    def generate_conclusion(self, topic: str, sections: list[QuestionResults]) -> str:
        """Generate report conclusion."""
        findings_text = ""
        for section in sections:
            # Truncate synthesis for token efficiency
            findings_text += f"- {section.question}: {section.synthesis[:150]}...\\n"
        
        prompt = f"""Topic: {topic}

Findings:
{findings_text}

Write a brief conclusion (50-80 words)."""

        return self.llm.generate(prompt, CONCLUSION_SYSTEM_PROMPT)
    
    def generate_report(self, report: ResearchReport) -> str:
        """Generate the final markdown report."""
        console.print(Panel("📝 GENERATING: Creating final research report", style="bold green"))
        
        md = f"""# Research Report: {report.topic}

**Generated:** {report.timestamp}
**Research Method:** AI-Powered Web Research (ReAct Pattern)

---

## Executive Summary

{report.introduction}

---

## Table of Contents

"""
        # Add TOC
        for i, section in enumerate(report.sections, 1):
            md += f"{i}. {section.question}\n"
        
        md += f"{len(report.sections) + 1}. Conclusion\n"
        md += f"{len(report.sections) + 2}. Sources\n\n---\n\n"
        
        # Add sections
        all_sources = []
        for i, section in enumerate(report.sections, 1):
            md += f"""## Section {i}: {section.question}

{section.synthesis}

"""
            # Collect sources
            for result in section.results:
                if result.url not in [s['url'] for s in all_sources]:
                    all_sources.append({
                        'title': result.title,
                        'url': result.url
                    })
        
        # Add conclusion
        md += f"""---

## Conclusion

{report.conclusion}

---

## Sources

"""
        for i, source in enumerate(all_sources, 1):
            md += f"{i}. [{source['title']}]({source['url']})\n"
        
        md += f"""
---

*This report was generated using an AI Research Agent implementing the ReAct (Reason + Act) pattern. 
The agent autonomously generated research questions, searched the web, and synthesized findings.*
"""
        
        return md
    
    def _generate_filename(self, topic: str) -> str:
        """Generate a meaningful filename using LLM based on the topic."""
        prompt = f"""Generate a short, descriptive filename for a research report about: "{topic}"

Rules:
- Use lowercase letters and underscores only
- Maximum 4 words
- No file extension
- Be descriptive but concise

Examples:
- "Climate Change Effects" -> climate_change_effects
- "AI in Healthcare" -> ai_healthcare_report
- "Trump vs Greenland" -> trump_greenland_analysis

Output ONLY the filename, nothing else."""
        
        try:
            filename = self.llm.generate(prompt).strip()
            # Clean up filename
            filename = filename.lower().replace(" ", "_").replace("-", "_")
            filename = ''.join(c for c in filename if c.isalnum() or c == '_')
            filename = filename[:50]  # Limit length
            if not filename:
                filename = "research_report"
        except Exception:
            # Fallback to timestamp-based name
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return f"{filename}.md"
    
    def _save_and_open_html(self, md_file: str, markdown_content: str, topic: str) -> str:
        """Convert markdown to HTML and open in browser."""
        import markdown
        
        # Convert markdown to HTML
        html_content = markdown.markdown(
            markdown_content,
            extensions=['tables', 'fenced_code', 'toc']
        )
        
        # Create styled HTML document
        html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{topic} - Research Report</title>
    <style>
        :root {{
            --primary: #2563eb;
            --bg: #f8fafc;
            --text: #1e293b;
            --border: #e2e8f0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            background: var(--bg);
            color: var(--text);
            line-height: 1.7;
        }}
        h1 {{
            color: var(--primary);
            border-bottom: 3px solid var(--primary);
            padding-bottom: 0.5rem;
        }}
        h2 {{
            color: var(--primary);
            margin-top: 2rem;
            border-left: 4px solid var(--primary);
            padding-left: 1rem;
        }}
        h3 {{
            color: #475569;
        }}
        a {{
            color: var(--primary);
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        blockquote {{
            border-left: 4px solid var(--primary);
            margin: 1rem 0;
            padding: 0.5rem 1rem;
            background: white;
            border-radius: 0 8px 8px 0;
        }}
        code {{
            background: #e2e8f0;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        pre {{
            background: #1e293b;
            color: #e2e8f0;
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
        }}
        pre code {{
            background: none;
            padding: 0;
        }}
        ul, ol {{
            padding-left: 1.5rem;
        }}
        li {{
            margin: 0.5rem 0;
            line-height: 1.6;
        }}
        hr {{
            border: none;
            border-top: 2px solid var(--border);
            margin: 2rem 0;
        }}
        .metadata {{
            background: white;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid var(--border);
            margin-bottom: 2rem;
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>"""
        
        # Save HTML file
        html_file = md_file.replace('.md', '.html')
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_doc)
        
        # Open in browser
        webbrowser.open(f'file://{Path(html_file).absolute()}')
        
        return html_file
    
    def research(self, topic: str, output_file: Optional[str] = None) -> ResearchReport:
        """
        Execute the full ReAct research workflow.
        
        This is the main orchestration method that coordinates:
        1. Planning (Reason) - Generate questions
        2. Acting (Act) - Web search
        3. Synthesizing (Reason) - Process results
        4. Reporting - Generate output
        """
        console.print(Panel(
            f"🚀 Starting Research on: [bold]{topic}[/bold]",
            style="bold white on blue"
        ))
        
        # Generate filename if not provided
        if output_file is None:
            output_file = self._generate_filename(topic)
        
        # Ensure reports directory exists
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        # If output_file doesn't include path, put in reports/
        output_path = Path(output_file)
        if not output_path.parent.exists() or output_path.parent == Path("."):
            output_file = str(reports_dir / output_path.name)
        
        # Initialize report
        report = ResearchReport(
            topic=topic,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # PHASE 1: Planning (Reason)
        questions = self.plan(topic)
        report.questions = questions
        
        # PHASE 2: Acting (Act)
        results = self.act(questions)
        
        # PHASE 3: Synthesizing (Reason)
        synthesized = self.synthesize_all(results)
        report.sections = synthesized
        
        # PHASE 4: Generate intro and conclusion
        console.print("📖 Generating introduction and conclusion...", style="yellow")
        report.introduction = self.generate_introduction(topic, questions)
        report.conclusion = self.generate_conclusion(topic, synthesized)
        
        # PHASE 5: Generate final report
        markdown_report = self.generate_report(report)
        
        # Save markdown report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        
        # Convert to HTML and open in browser
        html_file = self._save_and_open_html(output_file, markdown_report, topic)
        
        console.print(Panel(
            f"✅ Research Complete!\n\n📄 Markdown: [bold]{output_file}[/bold]\n🌐 HTML: [bold]{html_file}[/bold]",
            style="bold green"
        ))
        
        # Display report preview
        console.print("\n📄 Report Preview:\n", style="bold")
        console.print(Markdown(markdown_report[:2000] + "\n\n...[truncated for preview]..."))
        
        return report


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AI Web Research Agent - ReAct Pattern",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py "Climate Change"
  python agent.py "Artificial Intelligence in Healthcare" -o ai_health_report.md
  python agent.py "Renewable Energy Trends" --provider groq
        """
    )
    
    parser.add_argument(
        "topic",
        type=str,
        help="Research topic to investigate"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file name (default: auto-generated by LLM)"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        choices=["gemini", "groq", "ollama", "openrouter"],
        help="Override LLM provider (default: from .env)"
    )
    
    args = parser.parse_args()
    
    # Override provider if specified
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    
    try:
        # Create and run agent
        agent = ResearchAgent()
        agent.research(args.topic, args.output)
        
    except ValueError as e:
        console.print(f"[red]Configuration Error:[/red] {e}")
        console.print("\nPlease check your .env file and ensure API keys are set.")
        return 1
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.exception("Research failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
