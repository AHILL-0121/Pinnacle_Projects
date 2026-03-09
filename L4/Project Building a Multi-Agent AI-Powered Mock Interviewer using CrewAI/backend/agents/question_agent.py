"""
Question Generator Agent.

Uses direct LLM calls (Ollama or Groq) — no CrewAI, no OpenAI.
In the MVP, base questions come from the static question bank.
This agent rephrases questions and generates follow-ups dynamically.
"""

from backend.utils.llm_config import call_llm, LLMProvider

SYSTEM_PROMPT = (
    "You are a senior technical interviewer. "
    "You generate clear, role-specific interview questions that test genuine understanding. "
    "Return ONLY the question text — no preamble, no explanation."
)


class QuestionAgent:
    """Generates and rephrases interview questions using direct LLM calls."""

    def __init__(self, provider: LLMProvider = LLMProvider.OLLAMA):
        self.provider = provider

    def rephrase_question(self, base_question: str, role: str) -> str:
        """Use LLM to rephrase a question for variety."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Rephrase the following interview question for a {role} candidate. "
                    "Keep the core meaning but vary the phrasing slightly. "
                    "Return ONLY the rephrased question.\n\n"
                    f"Original: {base_question}"
                ),
            },
        ]
        return call_llm(messages, self.provider)

    def generate_followup(self, question: str, answer: str, role: str) -> str:
        """Generate a follow-up question based on the candidate's answer."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"You asked a {role} candidate:\n{question}\n\n"
                    f"Their answer:\n{answer}\n\n"
                    "Generate ONE short follow-up question that probes deeper into "
                    "a weakness or interesting point in their answer. "
                    "Return ONLY the follow-up question."
                ),
            },
        ]
        return call_llm(messages, self.provider)
