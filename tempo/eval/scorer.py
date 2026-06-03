"""Answer extraction and scoring for all supported task types.

Single module for ALL answer extraction — no more duplicated regexes
across 3 files. Each task type has one extraction function.

Supported tasks:
    classification: extract "Answer: X" label (HAR, bearing, sleep, ECG)
    anomaly:        extract yes/no or True/False
    mcq:            extract A/B/C/D letter
    true_false:     extract True/False
    free_text:      no extraction, return raw text
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

# Answer extraction — ONE function per format

_LABEL_RE = re.compile(
    r"(?:activity|answer|class|label|diagnosis|result|stage)\s+(?:is|:)\s*"
    r"([a-zA-Z0-9_\-\s]+?)[\.\,\n]",
    re.IGNORECASE,
)


def extract_answer(text: str, task: str = "classification") -> str:
    """Extract the answer from a model-generated text.

    This is THE answer extraction function. No other file should
    define its own regex for this purpose.

    Args:
        text: Raw model output.
        task: Task type (determines extraction strategy).

    Returns:
        Extracted answer string (normalized).
    """
    if task in ("classification", "anomaly_detection"):
        return _extract_label(text)
    elif task == "mcq":
        return _extract_mc_letter(text) or ""
    elif task == "true_false":
        return _extract_true_false(text) or ""
    else:
        return text.strip()


def _extract_label(text: str) -> str:
    """Extract classification label from CoT answer.

    Handles both single-word ("Answer: walking") and multi-word
    ("Answer: Non-REM stage 2") labels.
    """
    # Try "Answer: X" pattern first (handles multi-word)
    m = re.search(r"Answer:\s*(.+?)(?:\.|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")

    # Try "the activity/class is X" pattern
    m = _LABEL_RE.search(text.lower())
    if m:
        return m.group(1).strip()

    # Fallback: first sentence
    return re.split(r"[\.\n]", text.strip(), maxsplit=1)[0].strip()[:80]


def _extract_mc_letter(text: str) -> str | None:
    """Extract MC choice letter (A-D) from answer.

    Strict: only matches unambiguous patterns to avoid false positives.
    """
    if not text:
        return None
    s = text.strip()
    # Leading letter: "B) ...", "(B) ...", "**B** ..."
    m = re.match(r"^\s*\(?\*{0,2}([A-D])\*{0,2}\)?\s*[\)\.\:]", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # "Option B"
    m = re.search(r"\boption\s+([A-D])\b", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # "Answer: B", "answer is (C)", "correct answer is (C)"
    m = re.search(r"\banswer\s+is\s+\(?([A-D])\)?\b", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\banswer\s*[:=]\s*\(?([A-D])\)?\b", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def _extract_true_false(text: str) -> str | None:
    """Extract True/False from answer."""
    if not text:
        return None
    m = re.search(r"\b(true|false)\b", text, re.IGNORECASE)
    return m.group(1).lower() if m else None


# Scoring

def score_one(
    task: str,
    question: str,
    gold: str,
    prediction: str,
) -> dict[str, Any]:
    """Score a single prediction against gold. Returns dict with 'correct' key."""
    gold_ans = extract_answer(gold, task)
    pred_ans = extract_answer(prediction, task)

    if task in ("classification", "anomaly_detection"):
        g, p = gold_ans.lower(), pred_ans.lower()
        correct = float(g and (g == p or g in p or p in g))
        return {"correct": correct, "subtask": task, "gold_ans": g, "pred_ans": p}

    elif task == "mcq":
        g = _extract_mc_letter(gold)
        p = _extract_mc_letter(prediction)
        correct = float(g is not None and g == p)
        return {"correct": correct, "subtask": "mcq", "gold_letter": g, "pred_letter": p}

    elif task == "true_false":
        g = _extract_true_false(gold)
        p = _extract_true_false(prediction)
        correct = float(g is not None and g == p)
        return {"correct": correct, "subtask": "true_false", "gold_tf": g, "pred_tf": p}

    return {"correct": None, "subtask": "free_text"}


def score_predictions(predictions: list[dict]) -> dict[str, Any]:
    """Aggregate scored predictions into per-task metrics."""
    per_sub: dict[str, list[float]] = defaultdict(list)
    for rec in predictions:
        c = rec.get("correct")
        if c is not None:
            per_sub[rec["subtask"]].append(c)

    result: dict[str, Any] = {"n_total": len(predictions), "per_subtask": {}}
    all_scores: list[float] = []
    for sub in sorted(per_sub):
        scores = per_sub[sub]
        acc = sum(scores) / len(scores) if scores else 0.0
        result["per_subtask"][sub] = {
            "accuracy": round(acc, 4),
            "n": len(scores),
            "correct": int(sum(scores)),
        }
        all_scores.extend(scores)

    result["overall_accuracy"] = round(
        sum(all_scores) / len(all_scores), 4
    ) if all_scores else 0.0
    return result
