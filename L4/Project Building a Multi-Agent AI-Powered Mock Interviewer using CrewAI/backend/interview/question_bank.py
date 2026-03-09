"""
Question Bank — role-specific interview questions.
Each role maps to a list of question dicts with metadata.
"""

QUESTION_BANK: dict[str, list[dict]] = {
    "Data Scientist": [
        {
            "id": 1,
            "question": "What is overfitting in machine learning, and what techniques can you use to prevent it?",
            "topic": "Machine Learning",
            "keywords": ["overfitting", "regularization", "cross-validation", "dropout", "pruning", "early stopping"],
        },
        {
            "id": 2,
            "question": "Explain the bias-variance tradeoff. How does it affect model selection?",
            "topic": "Machine Learning",
            "keywords": ["bias", "variance", "tradeoff", "underfitting", "overfitting", "model complexity"],
        },
        {
            "id": 3,
            "question": "What is cross-validation and why is it important? Describe k-fold cross-validation.",
            "topic": "Model Evaluation",
            "keywords": ["cross-validation", "k-fold", "generalization", "test set", "train set", "validation"],
        },
        {
            "id": 4,
            "question": "How do you handle missing data in a dataset? Describe at least three strategies.",
            "topic": "Data Preprocessing",
            "keywords": ["imputation", "mean", "median", "mode", "drop", "KNN", "forward fill", "missing"],
        },
        {
            "id": 5,
            "question": "Explain the difference between regression and classification problems with examples.",
            "topic": "Machine Learning",
            "keywords": ["regression", "classification", "continuous", "discrete", "label", "prediction"],
        },
        {
            "id": 6,
            "question": "How would you evaluate and handle an imbalanced dataset in a classification task?",
            "topic": "Model Evaluation",
            "keywords": ["imbalanced", "SMOTE", "oversampling", "undersampling", "precision", "recall", "F1", "AUC", "ROC"],
        },
    ],
    "Web Developer": [
        {
            "id": 1,
            "question": "Explain the difference between CSS Flexbox and CSS Grid. When would you use each?",
            "topic": "CSS",
            "keywords": ["flexbox", "grid", "layout", "responsive", "one-dimensional", "two-dimensional"],
        },
        {
            "id": 2,
            "question": "What is the event loop in JavaScript and how does it handle asynchronous operations?",
            "topic": "JavaScript",
            "keywords": ["event loop", "async", "callback", "promise", "microtask", "macrotask", "non-blocking"],
        },
        {
            "id": 3,
            "question": "Explain RESTful API design principles. What makes an API RESTful?",
            "topic": "Backend",
            "keywords": ["REST", "stateless", "HTTP", "GET", "POST", "PUT", "DELETE", "resource", "endpoint"],
        },
        {
            "id": 4,
            "question": "What is the Virtual DOM in React and why is it beneficial?",
            "topic": "React",
            "keywords": ["virtual DOM", "reconciliation", "diffing", "render", "performance", "DOM"],
        },
        {
            "id": 5,
            "question": "Describe the difference between SQL and NoSQL databases. Give an example use case for each.",
            "topic": "Databases",
            "keywords": ["SQL", "NoSQL", "relational", "document", "schema", "MongoDB", "PostgreSQL"],
        },
    ],
    "Product Manager": [
        {
            "id": 1,
            "question": "How do you prioritize features in a product backlog? Describe a framework you have used.",
            "topic": "Prioritization",
            "keywords": ["RICE", "MoSCoW", "Kano", "impact", "effort", "stakeholder", "priority"],
        },
        {
            "id": 2,
            "question": "Explain how you would define and track success metrics for a new product feature.",
            "topic": "Metrics",
            "keywords": ["KPI", "metric", "OKR", "success", "north star", "DAU", "retention", "conversion"],
        },
        {
            "id": 3,
            "question": "How do you gather and incorporate user feedback into product decisions?",
            "topic": "User Research",
            "keywords": ["user research", "interview", "survey", "NPS", "feedback", "usability", "persona"],
        },
        {
            "id": 4,
            "question": "Describe a time you had to say no to a stakeholder request. How did you handle it?",
            "topic": "Stakeholder Management",
            "keywords": ["stakeholder", "prioritization", "tradeoff", "communication", "alignment", "roadmap"],
        },
        {
            "id": 5,
            "question": "Walk me through how you would launch a new feature from idea to release.",
            "topic": "Execution",
            "keywords": ["discovery", "requirements", "design", "development", "testing", "launch", "GTM"],
        },
    ],
}

AVAILABLE_ROLES = list(QUESTION_BANK.keys())


def get_questions(role: str) -> list[dict]:
    """Return the list of questions for a given role."""
    if role not in QUESTION_BANK:
        raise ValueError(f"Role '{role}' not found. Available: {AVAILABLE_ROLES}")
    return QUESTION_BANK[role]


def get_question_by_index(role: str, index: int) -> dict | None:
    """Return a single question by index, or None if out of range."""
    questions = get_questions(role)
    if 0 <= index < len(questions):
        return questions[index]
    return None


def total_questions(role: str) -> int:
    return len(get_questions(role))
