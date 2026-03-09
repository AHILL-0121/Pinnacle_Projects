"""
Scoring Utility — parses structured LLM evaluation output into a clean dict.
"""

import re


def parse_evaluation_output(raw_text: str) -> dict:
    """
    Parse the structured output from the evaluation agent.

    Expected format in raw_text:
        SCORE: 7
        CORRECTNESS: High
        DEPTH: Medium
        EXAMPLE: Missing
        CLARITY: High
        STRENGTH: ...
        IMPROVEMENT: ...

    Returns a dict with parsed values.
    Falls back gracefully if parsing fails.
    """
    result = {
        "score": None,
        "correctness": None,
        "depth": None,
        "example": None,
        "clarity": None,
        "strength": "",
        "improvement": "",
        "raw": raw_text,
    }

    patterns = {
        "score": r"SCORE\s*:\s*(\d+(?:\.\d+)?)",
        "correctness": r"CORRECTNESS\s*:\s*(.+)",
        "depth": r"DEPTH\s*:\s*(.+)",
        "example": r"EXAMPLE\s*:\s*(.+)",
        "clarity": r"CLARITY\s*:\s*(.+)",
        "strength": r"STRENGTH\s*:\s*(.+)",
        "improvement": r"IMPROVEMENT\s*:\s*(.+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if key == "score":
                try:
                    result["score"] = float(value)
                except ValueError:
                    result["score"] = 0.0
            else:
                result[key] = value

    # If no score was parsed, try to extract any standalone number 1-10
    if result["score"] is None:
        numbers = re.findall(r"\b(\d+(?:\.\d+)?)\s*/\s*10\b", raw_text)
        if numbers:
            try:
                result["score"] = float(numbers[0])
            except ValueError:
                result["score"] = 5.0
        else:
            result["score"] = 5.0

    return result


def compute_final_report(evaluations: list[dict]) -> dict:
    """
    Aggregate per-question evaluations into a final interview report.

    Args:
        evaluations: list of dicts returned by parse_evaluation_output,
                     each enriched with 'feedback' and 'question' keys.

    Returns:
        dict with overall_score, strengths, improvements, per_question summary.
    """
    if not evaluations:
        return {
            "overall_score": 0,
            "grade": "N/A",
            "strengths": [],
            "improvements": [],
            "per_question": [],
        }

    scores = [e.get("score", 5) or 5 for e in evaluations]
    overall_score = round(sum(scores) / len(scores), 1)

    strengths = [
        e["strength"] for e in evaluations if e.get("strength")
    ]
    improvements = [
        e["improvement"] for e in evaluations if e.get("improvement")
    ]

    grade = _score_to_grade(overall_score)

    per_question = [
        {
            "question": e.get("question", f"Q{i+1}"),
            "score": e.get("score", 5),
            "feedback": e.get("feedback", ""),
        }
        for i, e in enumerate(evaluations)
    ]

    return {
        "overall_score": overall_score,
        "grade": grade,
        "strengths": strengths,
        "improvements": improvements,
        "per_question": per_question,
    }


def _score_to_grade(score: float) -> str:
    if score >= 9:
        return "Excellent"
    elif score >= 7:
        return "Good"
    elif score >= 5:
        return "Average"
    elif score >= 3:
        return "Needs Improvement"
    else:
        return "Poor"
