"""Diverse question and answer templates for Stage 1 QA generation.

Each task type has 10+ question variants and 5+ answer styles to prevent
the model from memorizing a single phrasing. Templates use Python
format-string placeholders filled with computed ground-truth values.

Design rules:
  - No two templates should share the same sentence structure.
  - Mix formal and conversational tones.
  - Vary answer length (1 sentence to 3 sentences).
  - Include some templates that provide reasoning, not just labels.
  - Include templates that explicitly reference the signal domain.
"""

from __future__ import annotations

import random
from typing import Any


# Question templates per task

QUESTIONS = {
    "trend": [
        "What is the overall trend of this time series?",
        "Describe the direction this signal is heading.",
        "Is this series increasing, decreasing, or staying roughly flat?",
        "Looking at the general movement, what trend do you observe?",
        "Analyze the long-term direction of this data.",
        "Would you characterize this signal as rising, falling, or stable?",
        "What is the dominant directional pattern in this segment?",
        "If you drew a best-fit line through this data, what slope would it have?",
        "Summarize the overall trajectory of this time series in one sentence.",
        "Does this signal show any clear upward or downward tendency?",
        "How would you describe the general drift of these values over time?",
        "Examine this time series and classify its trend direction.",
    ],
    "forecast": [
        "Continue this time series.",
        "Forecast the continuation of this signal.",
        "Predict what comes next in this sequence.",
        "Based on the observed pattern, generate the next segment.",
        "Extrapolate this signal forward.",
        "What does the continuation of this time series look like?",
        "Generate the next part of this signal.",
        "If this pattern continues, produce the following segment.",
        "Extend this time series.",
        "Provide your prediction for the next portion of this signal.",
    ],
    "describe": [
        "Describe the key characteristics of this time series.",
        "What patterns do you notice in this signal?",
        "Give a concise summary of this data segment.",
        "Characterize the behavior of this time series.",
        "What stands out when you look at this signal?",
        "Provide an overview of the main features of this time series.",
        "How would you describe this data to a colleague?",
        "Summarize what this signal tells you.",
        "Walk me through the notable features of this time series.",
        "What is your high-level assessment of this data?",
        "If you had to write a one-paragraph summary of this signal, what would it say?",
        "Analyze the shape and behavior of this time series segment.",
    ],
    "period": [
        "Does this time series exhibit any repeating pattern?",
        "What is the dominant cycle length, if any?",
        "Is there periodicity in this signal?",
        "Can you detect any cyclical behavior in the data?",
        "Estimate the period of repetition, if one exists.",
        "Does this series show seasonal or periodic structure?",
        "Are there recurring patterns, and if so, how long is each cycle?",
        "Analyze whether this signal repeats at regular intervals.",
        "Is there evidence of a dominant frequency in this time series?",
        "Check for periodicity: does this pattern repeat?",
        "What is the approximate wavelength of any repeating pattern you see?",
        "Would you describe this signal as periodic, quasi-periodic, or aperiodic?",
    ],
    "anomaly": [
        "Are there any anomalies or outliers in this data?",
        "Does this time series contain any unusual values?",
        "Identify any data points that deviate significantly from the norm.",
        "Is there anything abnormal in this signal?",
        "Check this data for anomalous behavior.",
        "Do you see any unexpected spikes, dips, or deviations?",
        "Are all values in this series within the expected range?",
        "Perform an outlier analysis on this time series.",
        "Flag any data points that look suspicious or out of place.",
        "Does this signal contain any values that break the overall pattern?",
        "Examine this data for potential measurement errors or anomalies.",
        "Is this time series clean, or are there problematic data points?",
    ],
    # statistics removed: model sees TOTEM codes, not raw numbers.
    # It cannot compute min/max/mean/std from discrete codes.
    "volatility": [
        "How volatile is this time series?",
        "Assess the stability of this signal.",
        "How much fluctuation does this data show?",
        "Rate the variability: is this a calm or turbulent signal?",
        "Would you describe this time series as stable, moderately variable, or highly volatile?",
        "How much noise or variation is present in this data?",
        "Evaluate the consistency of this signal over time.",
        "Is this a smooth signal or a noisy one?",
        "Characterize the level of fluctuation in this time series.",
        "On a scale from smooth to chaotic, where does this signal fall?",
        "How predictable are the values in this time series?",
        "Assess whether this signal shows steady behavior or erratic swings.",
    ],
    "turning_points": [
        "How many turning points (peaks and valleys) are in this signal?",
        "Count the local maxima and minima.",
        "How often does this time series change direction?",
        "How many times does this signal reverse its trajectory?",
        "Identify the number of peaks and troughs in this data.",
        "Is this a smooth signal with few direction changes, or a choppy one?",
        "How many local extrema can you detect?",
        "Count the number of times this signal switches between rising and falling.",
        "Estimate the frequency of direction reversals in this time series.",
        "How jagged or smooth is this signal based on its turning points?",
    ],
}


# Answer templates per task

# Each entry is a callable: (ground_truth_dict, domain) -> str
# The ground_truth_dict keys vary by task.

def _trend_answers() -> list:
    """Templates for trend task. GT keys: shape, trend_direction, strength, details."""
    return [
        lambda gt, d: f"The signal shows a pattern of {gt['shape']}. {gt['details'].capitalize()}.",
        lambda gt, d: f"Overall, this {d} data exhibits {gt['shape']}.",
        lambda gt, d: f"I observe {gt['shape']} in the values — {gt['details']}.",
        lambda gt, d: f"The dominant pattern is {gt['shape']}. The overall direction is {gt['trend_direction']} with {gt['strength']} intensity.",
        lambda gt, d: f"Looking at this {d} signal: {gt['details']}. I would classify the shape as {gt['shape']}.",
        lambda gt, d: f"This time series shows {gt['shape']}. In terms of directionality, the signal is {gt['trend_direction']}.",
        lambda gt, d: f"Shape analysis: {gt['shape']} ({gt['strength']}). {gt['details'].capitalize()}.",
        lambda gt, d: f"The data follows a {gt['shape']} pattern — {gt['details']}. The overall movement is {gt['trend_direction']}.",
    ]


def _forecast_answers() -> list:
    """Templates for forecast task. GT keys: horizon_codes (str of <ts_*> tokens).

    The target is discrete TOTEM code tokens wrapped in <ts_start>/<ts_end>,
    NOT numerical values. At inference the generated codes are decoded by
    TOTEM's decoder back to continuous values (Chameleon-style).
    """
    return [
        lambda gt, d: f"<ts_start> {gt['horizon_codes']} <ts_end>",
        lambda gt, d: f"Forecast: <ts_start> {gt['horizon_codes']} <ts_end>",
        lambda gt, d: f"Continuation: <ts_start> {gt['horizon_codes']} <ts_end>",
    ]


def _describe_answers() -> list:
    """Templates for describe task. GT keys: shape, trend_direction, details, period, cv_label."""
    return [
        lambda gt, d: f"This {d} signal shows {gt['shape']}{_period_clause(gt)}. {_variability_sentence(gt)}",
        lambda gt, d: f"I see {gt['shape']} with {gt['cv_label']} variability.{_period_clause2(gt)}",
        lambda gt, d: f"The data exhibits {gt['shape']}, {'with a repeating cycle of about ' + str(gt['period']) + ' steps' if gt['period'] > 0 else 'without clear periodicity'}. Variability is {gt['cv_label']}.",
        lambda gt, d: f"Key observations: {gt['details']}. Variability is {gt['cv_label']}{', and there is a periodic component with period ~' + str(gt['period']) if gt['period'] > 0 else ''}.",
        lambda gt, d: f"This {d} signal exhibits {gt['shape']}. {'It repeats roughly every ' + str(gt['period']) + ' time steps. ' if gt['period'] > 0 else ''}{_variability_sentence(gt)}",
        lambda gt, d: f"Shape: {gt['shape']}. {'Periodicity detected at ~' + str(gt['period']) + ' steps. ' if gt['period'] > 0 else 'No obvious cycle. '}The fluctuations are {gt['cv_label']}.",
        lambda gt, d: f"A {gt['cv_label']}-variability {d} segment showing {gt['shape']}.{_period_clause2(gt)}",
    ]


def _period_answers() -> list:
    """Templates for period task. GT keys: period (int), is_periodic (bool), confidence (float)."""
    return [
        lambda gt, d: f"The dominant period is approximately {gt['period']} time steps." if gt.get('is_periodic') else "No clear periodicity is detected in this signal.",
        lambda gt, d: f"Yes, there is a repeating cycle of about {gt['period']} steps." if gt.get('is_periodic') else "No, this signal does not show a repeating pattern.",
        lambda gt, d: f"The signal has a dominant cycle of {gt['period']} time steps." if gt.get('is_periodic') else "The signal is aperiodic — no repeating structure.",
        lambda gt, d: f"Periodic — the pattern repeats roughly every {gt['period']} observations." if gt.get('is_periodic') else "Aperiodic — no repeating structure detected.",
        lambda gt, d: (f"I detect a cycle with period ~{gt['period']} in this {d} data." if gt.get('is_periodic') else f"This {d} signal appears aperiodic. No consistent repetition is evident."),
        lambda gt, d: f"This signal is quasi-periodic with an estimated period of {gt['period']} steps." if gt.get('is_periodic') else "The signal does not repeat at regular intervals.",
    ]


def _anomaly_answers() -> list:
    """Templates for anomaly task. GT keys: has_anomaly (bool), explanation (str), n_outliers, pct, positions."""
    return [
        lambda gt, d: gt['explanation'],
        lambda gt, d: _anomaly_yn(gt, d),
        lambda gt, d: f"Anomalies detected. {gt.get('n_outliers', 0)} data points deviate significantly from the bulk of the data." if gt['has_anomaly'] else "The data looks clean — no significant outliers found.",
        lambda gt, d: f"This signal contains unusual values that break the overall pattern. Specifically, {gt.get('n_outliers', 0)} points lie outside 1.5x the IQR." if gt['has_anomaly'] else "No anomalies. The values are consistent throughout the segment.",
        lambda gt, d: f"Warning: anomalous behavior detected in this {d} data. Some values are unexpectedly extreme." if gt['has_anomaly'] else f"This {d} segment is well-behaved with no anomalous readings.",
    ]


    # _statistics_answers removed: model cannot compute numerical
    # statistics from discrete TOTEM codes.


def _volatility_answers() -> list:
    """Templates for volatility task. GT keys: label (low/moderate/high)."""
    return [
        lambda gt, d: f"The time series has {gt['label']} volatility.",
        lambda gt, d: f"{'This is a relatively calm, stable signal.' if gt['label'] == 'low' else 'The signal shows moderate fluctuations — neither very smooth nor very noisy.' if gt['label'] == 'moderate' else 'This is a highly volatile signal with large swings between consecutive values.'}",
        lambda gt, d: f"Volatility assessment: {gt['label']}. {'The values change slowly and predictably.' if gt['label'] == 'low' else 'There is noticeable variation but it stays within bounds.' if gt['label'] == 'moderate' else 'The data swings rapidly, making short-term prediction difficult.'}",
        lambda gt, d: f"{'Smooth and steady' if gt['label'] == 'low' else 'Moderately variable' if gt['label'] == 'moderate' else 'Highly erratic'} — this {d} signal has {gt['label']} volatility.",
        lambda gt, d: f"I would rate this signal's volatility as {gt['label']}. {'You could predict the next value with reasonable confidence.' if gt['label'] == 'low' else 'The fluctuations add meaningful uncertainty.' if gt['label'] == 'moderate' else 'The amplitude of variation is substantial relative to the signal level.'}",
        lambda gt, d: f"The fluctuation level is {gt['label']}. {'Consecutive values rarely differ much.' if gt['label'] == 'low' else 'Some ups and downs are visible but not extreme.' if gt['label'] == 'moderate' else 'Sharp jumps and drops are frequent.'}",
    ]


def _turning_points_answers() -> list:
    """Templates for turning_points task. GT keys: count (int)."""
    return [
        lambda gt, d: f"The signal contains {gt['count']} turning points (local peaks and valleys).",
        lambda gt, d: f"I count {gt['count']} direction reversals — {'this is a smooth signal with few changes' if gt['count'] < 5 else 'a moderately oscillating signal' if gt['count'] < 15 else 'a choppy signal that changes direction frequently'}.",
        lambda gt, d: f"There are {gt['count']} local extrema. {'The signal is quite smooth.' if gt['count'] < 5 else 'The signal oscillates regularly.' if gt['count'] < 15 else 'The signal is highly jagged.'}",
        lambda gt, d: f"{gt['count']} peaks and valleys. {'Few direction changes — a smooth trajectory.' if gt['count'] < 5 else 'A moderate number of reversals.' if gt['count'] < 15 else 'Very frequent reversals — high-frequency oscillation.'}",
        lambda gt, d: f"This segment has {gt['count']} turning points, meaning the signal changes direction {gt['count']} times within the observation window.",
        lambda gt, d: f"Direction changes: {gt['count']}. For a segment of this length, that is {'unusually few' if gt['count'] < 3 else 'typical' if gt['count'] < 20 else 'quite high'}.",
    ]


# Answer template registry

ANSWER_TEMPLATES = {
    "trend": _trend_answers(),
    "forecast": _forecast_answers(),
    "describe": _describe_answers(),
    "period": _period_answers(),
    "anomaly": _anomaly_answers(),
    "volatility": _volatility_answers(),
    "turning_points": _turning_points_answers(),
}


# Pre-prompt (question context) templates

PRE_PROMPTS = [
    "You are analyzing a segment of {domain} data.",
    "The following is a time series from the {domain} domain.",
    "Consider this {domain} signal segment.",
    "A {domain} sensor recorded the following time series.",
    "Here is a data segment from a {domain} source.",
    "You are given {domain} time series data to analyze.",
    "Examine the following {domain} signal.",
    "This data was recorded from a {domain} system.",
    "Below is a segment of {domain} measurements.",
    "Analyze this {domain} time series.",
    "A time series segment from the {domain} domain is provided.",
    "The following signal comes from {domain} monitoring.",
]


# Public API

def make_qa_pair(
    task: str,
    ground_truth: dict[str, Any],
    domain: str,
    rng: random.Random | None = None,
) -> dict[str, str]:
    """Generate a diverse (question, answer, pre_prompt) triple.

    Args:
        task: one of the 8 QA task types.
        ground_truth: dict with task-specific keys (see per-task templates).
        domain: human-readable domain description.
        rng: optional RNG for deterministic selection.

    Returns:
        Dict with keys: pre_prompt, question, answer, task.
    """
    if rng is None:
        rng = random.Random()

    questions = QUESTIONS.get(task, QUESTIONS["describe"])
    answer_fns = ANSWER_TEMPLATES.get(task, ANSWER_TEMPLATES["describe"])

    question = rng.choice(questions)
    # Fill format placeholders if any (e.g., {horizon})
    try:
        question = question.format(**ground_truth)
    except (KeyError, IndexError):
        pass

    answer_fn = rng.choice(answer_fns)
    try:
        answer = answer_fn(ground_truth, domain)
    except (KeyError, TypeError) as e:
        # Fallback to first template if ground_truth keys don't match
        answer = answer_fns[0](ground_truth, domain)

    pre_prompt = rng.choice(PRE_PROMPTS).format(domain=domain)

    return {
        "pre_prompt": pre_prompt,
        "question": question,
        "answer": answer,
        "task": task,
    }


# Helpers

def _anomaly_yn(gt: dict, d: str) -> str:
    if gt['has_anomaly']:
        n = gt.get('n_outliers', 0)
        pct = gt.get('pct', 0)
        noun = "outlier" if n == 1 else "outliers"
        return f"Yes — there {'is' if n == 1 else 'are'} {n} {noun} ({pct:.1f}% of values) that fall outside the expected range."
    return "No — all values are within the normal interquartile range."


def _article(word: str) -> str:
    return "an" if word and word[0] in "aeiouAEIOU" else "a"

def _trending(trend: str) -> str:
    return {"upward": "trending upward", "downward": "trending downward", "flat": "relatively flat"}.get(trend, trend)

def _period_clause(gt: dict) -> str:
    p = gt.get("period", 0)
    return f" with a repeating cycle of approximately {p} time steps" if p > 0 else ""

def _period_clause2(gt: dict) -> str:
    p = gt.get("period", 0)
    return f" A periodic component with period ~{p} is present." if p > 0 else " No clear periodicity."

def _variability_sentence(gt: dict) -> str:
    cv = gt.get("cv_label", "moderate")
    return {
        "high": "The series shows considerable variability relative to its mean.",
        "moderate": "Variability is moderate.",
        "low": "The values are relatively stable with little fluctuation.",
    }.get(cv, "Variability is moderate.")
