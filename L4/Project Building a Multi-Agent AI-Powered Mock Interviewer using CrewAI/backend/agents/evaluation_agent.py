"""
Evaluation Agent — analyzes the quality of a candidate's answer.

Uses direct LLM calls (Ollama or Groq) — no CrewAI, no OpenAI.

Returns a structured evaluation covering:
  - Score (1–10)
  - Correctness, Depth, Example, Clarity
  - Strength summary
  - Improvement suggestion
"""

from backend.utils.llm_config import call_llm, LLMProvider
from backend.utils.scoring import parse_evaluation_output

SYSTEM_PROMPT = (
    "You are a strict but fair technical interview evaluator. "
    "You assess candidate answers on correctness, depth, use of examples, and clarity. "
    "Always respond in the exact structured format requested — no deviations."
)


EVALUATION_OUTPUT_FORMAT = """
Your response MUST follow this exact format (use the labels verbatim):

SCORE: <integer 1-10>
CORRECTNESS: <High | Medium | Low>
DEPTH: <High | Medium | Low>
EXAMPLE: <Present | Missing | Partial>
CLARITY: <High | Medium | Low>
STRENGTH: <one sentence describing what the candidate did well>
IMPROVEMENT: <one sentence describing the most important area to improve>
""".strip()


class EvaluationAgent:
    """Evaluates a candidate's answer using direct LLM calls."""

    def __init__(self, provider: LLMProvider = LLMProvider.OLLAMA):
        self.provider = provider

    def evaluate(
        self,
        question: str,
        answer: str,
        role: str,
        keywords: list[str] | None = None,
    ) -> dict:
        """
        Evaluate a candidate's answer.

        Returns:
            Parsed evaluation dict (see scoring.parse_evaluation_output).
        """
        keyword_hint = ""
        if keywords:
            keyword_hint = f"\nExpected concepts to look for: {', '.join(keywords)}"

        user_content = (
            f"You are evaluating a candidate interviewing for the role of: {role}\n\n"
            f"Interview Question:\n{question}\n\n"
            f"Candidate's Answer:\n{answer}"
            f"{keyword_hint}\n\n"
            "Evaluate strictly and honestly. A vague or incomplete answer should score low.\n\n"
            f"{EVALUATION_OUTPUT_FORMAT}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        raw_result = call_llm(messages, self.provider)
        return parse_evaluation_output(raw_result)
