"""
Feedback Agent — converts a structured evaluation into natural, coaching feedback.

Uses direct LLM calls (Ollama or Groq) — no CrewAI, no OpenAI.
"""

from backend.utils.llm_config import call_llm, LLMProvider

SYSTEM_PROMPT = (
    "You are a supportive but honest interview coach. "
    "You convert evaluation data into concise, actionable coaching feedback. "
    "Be direct: acknowledge strengths briefly, then focus on the most important improvement."
)


class FeedbackAgent:
    """Generates coaching feedback using direct LLM calls."""

    def __init__(self, provider: LLMProvider = LLMProvider.OLLAMA):
        self.provider = provider

    def generate_feedback(self, evaluation: dict, question: str) -> str:
        """
        Generate natural language coaching feedback from an evaluation dict.
        Returns a 2–4 sentence coaching string.
        """
        score = evaluation.get("score", 5)
        correctness = evaluation.get("correctness", "Unknown")
        depth = evaluation.get("depth", "Unknown")
        example = evaluation.get("example", "Unknown")
        clarity = evaluation.get("clarity", "Unknown")
        strength = evaluation.get("strength", "")
        improvement = evaluation.get("improvement", "")

        user_content = (
            f"The candidate answered this interview question:\n'{question}'\n\n"
            f"Evaluation results:\n"
            f"  - Score: {score}/10\n"
            f"  - Correctness: {correctness}\n"
            f"  - Depth: {depth}\n"
            f"  - Example given: {example}\n"
            f"  - Clarity: {clarity}\n"
            f"  - Strength: {strength}\n"
            f"  - Improvement area: {improvement}\n\n"
            "Write 2–4 sentences of coaching feedback. "
            "Start with what they did well (if anything), then say clearly what to improve and HOW. "
            "Do NOT repeat the raw evaluation labels — synthesize into natural language."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return call_llm(messages, self.provider)

    def generate_final_summary(self, report: dict, role: str) -> str:
        """
        Generate a 3–5 sentence final interview debrief narrative.
        """
        overall = report.get("overall_score", 0)
        grade = report.get("grade", "N/A")
        strengths = "; ".join(report.get("strengths", [])) or "None identified"
        improvements = "; ".join(report.get("improvements", [])) or "None identified"

        user_content = (
            f"Write a final interview debrief for a {role} candidate.\n\n"
            f"Overall score: {overall}/10 ({grade})\n"
            f"Strengths across answers: {strengths}\n"
            f"Areas needing improvement: {improvements}\n\n"
            "Write 3–5 sentences that:\n"
            "1. State the overall performance honestly.\n"
            "2. Highlight 1–2 genuine strengths.\n"
            "3. Point out the most critical area to work on.\n"
            "4. End with one actionable study/practice recommendation.\n"
            "Be encouraging but honest."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return call_llm(messages, self.provider)
