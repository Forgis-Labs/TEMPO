"""Evaluation, sensitivity testing, and paper-reproducible scoring."""

from .sensitivity import sensitivity_test
from .scorer import score_one, score_predictions, extract_answer
from .evaluate import evaluate, EvalResult

__all__ = [
    "evaluate",
    "EvalResult",
    "sensitivity_test",
    "score_one",
    "score_predictions",
    "extract_answer",
]
