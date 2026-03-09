"""
Interview Controller — orchestrates the full interview session.

Manages:
  - Question sequencing
  - Per-answer evaluation + feedback
  - Session state
  - Final report generation
"""

from dataclasses import dataclass, field
from backend.interview.question_bank import (
    get_questions,
    get_question_by_index,
    total_questions,
    AVAILABLE_ROLES,
)
from backend.agents.evaluation_agent import EvaluationAgent
from backend.agents.feedback_agent import FeedbackAgent
from backend.utils.scoring import compute_final_report
from backend.utils.llm_config import LLMProvider


@dataclass
class AnswerRecord:
    question_index: int
    question: str
    answer: str
    evaluation: dict
    feedback: str


@dataclass
class InterviewSession:
    role: str
    provider: LLMProvider
    questions: list[dict] = field(default_factory=list)
    current_index: int = 0
    records: list[AnswerRecord] = field(default_factory=list)
    is_complete: bool = False
    final_report: dict = field(default_factory=dict)
    final_summary: str = ""


class InterviewController:
    """
    Main controller for one interview session.
    Instantiate once per session; call methods in order.
    """

    def __init__(self, role: str, provider: LLMProvider = LLMProvider.OLLAMA):
        if role not in AVAILABLE_ROLES:
            raise ValueError(
                f"Role '{role}' not supported. Choose from: {AVAILABLE_ROLES}"
            )
        self.session = InterviewSession(
            role=role,
            provider=provider,
            questions=get_questions(role),
        )
        self._eval_agent = EvaluationAgent(provider=provider)
        self._feedback_agent = FeedbackAgent(provider=provider)

    # ------------------------------------------------------------------
    # Read-only helpers
    # ------------------------------------------------------------------

    @property
    def role(self) -> str:
        return self.session.role

    @property
    def current_index(self) -> int:
        return self.session.current_index

    @property
    def total(self) -> int:
        return len(self.session.questions)

    @property
    def is_complete(self) -> bool:
        return self.session.is_complete

    @property
    def records(self) -> list[AnswerRecord]:
        return self.session.records

    @property
    def final_report(self) -> dict:
        return self.session.final_report

    @property
    def final_summary(self) -> str:
        return self.session.final_summary

    # ------------------------------------------------------------------
    # Core interview flow
    # ------------------------------------------------------------------

    def get_current_question(self) -> dict | None:
        """Return the current unanswered question dict, or None if done."""
        if self.session.is_complete:
            return None
        return get_question_by_index(self.session.role, self.session.current_index)

    def submit_answer(self, answer: str) -> tuple[dict, str]:
        """
        Process the candidate's answer for the current question.

        Returns:
            (evaluation_dict, feedback_text)
        Advances the question index automatically.
        Triggers final report generation when all questions are answered.
        """
        if self.session.is_complete:
            raise RuntimeError("Interview is already complete.")

        q = self.get_current_question()
        if q is None:
            raise RuntimeError("No current question to answer.")

        # 1. Evaluate
        evaluation = self._eval_agent.evaluate(
            question=q["question"],
            answer=answer,
            role=self.session.role,
            keywords=q.get("keywords", []),
        )

        # 2. Generate feedback
        feedback = self._feedback_agent.generate_feedback(
            evaluation=evaluation,
            question=q["question"],
        )

        # 3. Store record
        record = AnswerRecord(
            question_index=self.session.current_index,
            question=q["question"],
            answer=answer,
            evaluation=evaluation,
            feedback=feedback,
        )
        evaluation["question"] = q["question"]
        evaluation["feedback"] = feedback
        self.session.records.append(record)

        # 4. Advance index
        self.session.current_index += 1

        # 5. Check completion
        if self.session.current_index >= self.total:
            self._finalize()

        return evaluation, feedback

    def _finalize(self):
        """Compute final report and summary."""
        eval_list = [r.evaluation for r in self.session.records]
        self.session.final_report = compute_final_report(eval_list)
        self.session.final_summary = self._feedback_agent.generate_final_summary(
            report=self.session.final_report,
            role=self.session.role,
        )
        self.session.is_complete = True

    def switch_provider(self, new_provider: LLMProvider):
        """
        Hot-swap the LLM provider mid-session.
        Existing records are preserved; only future calls use the new provider.
        """
        self.session.provider = new_provider
        self._eval_agent = EvaluationAgent(provider=new_provider)
        self._feedback_agent = FeedbackAgent(provider=new_provider)
