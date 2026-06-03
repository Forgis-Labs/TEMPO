"""GPT-4o one-shot evaluation on the 5-rig bearing fault dataset.

Sends raw time series values as text to GPT-4o and asks it to classify
the bearing condition into one of 4 classes: normal, inner_race, outer_race, ball_fault.

This mirrors the TEMPO bearing eval but using GPT-4o as a text-only baseline.

Usage:
    uv run python -m tempo.eval.gpt_bearing_eval
    uv run python -m tempo.eval.gpt_bearing_eval --max-samples 50  # quick test
    uv run python -m tempo.eval.gpt_bearing_eval --model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# Load .env
_env = Path(__file__).resolve().parent.parent.parent / ".env"
if _env.exists():
    for line in _env.read_text().strip().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


CLASSES = ["normal", "inner_race", "outer_race", "ball_fault"]

SYSTEM_PROMPT = (
    "You are a vibration analysis expert specializing in rolling element bearing "
    "fault diagnosis. You will be given a raw vibration signal from a bearing "
    "and contextual information (sampling rate, RPM, computed features). "
    "Classify the bearing condition into exactly one of these categories:\n"
    "  - normal\n"
    "  - inner_race\n"
    "  - outer_race\n"
    "  - ball_fault\n\n"
    "Reason briefly about the signal features, then end your response with "
    "'Answer: <class>' on the last line, where <class> is one of the four categories above."
)

ONE_SHOT_EXAMPLE = (
    "Example:\n"
    "Signal (first 20 values): [0.12, -0.05, 0.18, -0.11, 0.42, -0.38, 0.51, -0.45, "
    "0.15, -0.08, 0.55, -0.50, 0.22, -0.15, 0.48, -0.42, 0.13, -0.07, 0.50, -0.44]\n"
    "RPM: 1797, Sampling rate: 12000 Hz\n"
    "Features: RMS=0.35g, Peak=0.55g, Kurtosis=4.8, Crest factor=5.2\n"
    "Bearing frequencies: BPFI=159.9Hz, BPFO=105.9Hz, BSF=69.6Hz\n"
    "Envelope peaks: 160Hz, 320Hz, 30Hz\n\n"
    "The envelope spectrum peaks at 160 Hz and 320 Hz closely match BPFI (159.9 Hz) "
    "and its 2x harmonic. The high kurtosis (4.8) and crest factor (5.2) indicate "
    "impulsive events consistent with a localized defect on the inner race.\n"
    "Answer: inner_race\n"
)


def ts_to_text(signal: list[float], max_points: int = 512) -> str:
    """Format time series as comma-separated values, downsampled if needed."""
    arr = np.array(signal, dtype=np.float64)
    if len(arr) > max_points:
        indices = np.linspace(0, len(arr) - 1, max_points).astype(int)
        arr = arr[indices]
    return ", ".join(f"{v:.4f}" for v in arr)


def build_bearing_prompt(sample: dict, max_points: int = 512) -> tuple[str, str]:
    """Build (system, user) prompt pair for a bearing sample."""
    system = SYSTEM_PROMPT + "\n\n" + ONE_SHOT_EXAMPLE

    signal = sample["signal"]
    features = sample.get("features", {})
    rpm = sample.get("rpm", 1797)

    # Build user message with signal + context
    parts = []
    parts.append(f"Bearing vibration signal sampled at 12,000 Hz, motor at {rpm} RPM.")
    parts.append(f"\nSignal ({len(signal)} points):")
    parts.append(f"[{ts_to_text(signal, max_points)}]")

    # Add computed features if available
    if features:
        feat_lines = []
        if "rms" in features:
            feat_lines.append(f"RMS={features['rms']:.4f}g")
        if "peak" in features:
            feat_lines.append(f"Peak={features['peak']:.4f}g")
        if "kurtosis" in features:
            feat_lines.append(f"Kurtosis={features['kurtosis']:.2f}")
        if "crest_factor" in features:
            feat_lines.append(f"Crest factor={features['crest_factor']:.2f}")
        if "peak_to_peak" in features:
            feat_lines.append(f"Peak-to-peak={features['peak_to_peak']:.4f}g")
        if feat_lines:
            parts.append(f"\nFeatures: {', '.join(feat_lines)}")

        if "bearing_freqs" in features:
            bf = features["bearing_freqs"]
            parts.append(
                f"Bearing characteristic frequencies: "
                f"BPFI={bf.get('BPFI', '?')}Hz, "
                f"BPFO={bf.get('BPFO', '?')}Hz, "
                f"BSF={bf.get('BSF', '?')}Hz, "
                f"FTF={bf.get('FTF', '?')}Hz"
            )

        if "envelope_peak_freqs" in features:
            epf = features["envelope_peak_freqs"]
            parts.append(f"Envelope spectrum peaks: {', '.join(f'{f:.1f}Hz' for f in epf)}")

        if "dominant_freq" in features:
            parts.append(f"Dominant FFT frequency: {features['dominant_freq']:.1f}Hz")

    parts.append("\nClassify this bearing condition as: normal, inner_race, outer_race, or ball_fault.")

    return system, "\n".join(parts)


def call_gpt(system: str, user: str, model: str = "gpt-4o",
             max_tokens: int = 500, temperature: float = 0.0) -> str:
    """Call OpenAI API."""
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def extract_answer(text: str) -> str:
    """Extract bearing class from GPT response."""
    import re

    # Look for "Answer: <class>"
    m = re.search(r"Answer:\s*(\S+)", text, re.IGNORECASE)
    if m:
        ans = m.group(1).strip().lower().rstrip(".")
        # Normalize
        for cls in CLASSES:
            if cls in ans or ans in cls:
                return cls
        # Partial matches
        if "inner" in ans:
            return "inner_race"
        if "outer" in ans:
            return "outer_race"
        if "ball" in ans:
            return "ball_fault"
        if "normal" in ans or "healthy" in ans:
            return "normal"
        return ans

    # Fallback: search for class names in the text
    text_lower = text.lower()
    # Check last few lines first (more likely to be the answer)
    last_lines = "\n".join(text.strip().split("\n")[-3:]).lower()
    for cls in CLASSES:
        if cls in last_lines:
            return cls

    for cls in CLASSES:
        if cls in text_lower:
            return cls

    if "inner" in text_lower:
        return "inner_race"
    if "outer" in text_lower:
        return "outer_race"
    if "ball" in text_lower:
        return "ball_fault"
    if "normal" in text_lower or "healthy" in text_lower:
        return "normal"

    return text.strip().split("\n")[-1][:100]


def load_bearing_test(data_path: str | None = None) -> list[dict]:
    """Load bearing test dataset."""
    search_paths = [
        data_path,
        "data/bearing_cot/bearing_features_test.json",
        "../data/bearing_cot/bearing_features_test.json",
        os.path.expanduser("~/data/bearing_cot/bearing_features_test.json"),
    ]
    for p in search_paths:
        if p and os.path.exists(p):
            with open(p) as f:
                data = json.load(f)
            print(f"Loaded {len(data)} bearing test samples from {p}")
            return data

    raise FileNotFoundError("Cannot find bearing_features_test.json")


def run_eval(
    model: str = "gpt-4o",
    max_samples: int | None = None,
    max_points: int = 512,
    output_dir: str = "results/gpt_bearing_baseline",
    data_path: str | None = None,
    seed: int = 42,
):
    """Run GPT bearing classification evaluation."""
    import random
    from sklearn.metrics import (
        accuracy_score, f1_score, classification_report, confusion_matrix,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model}_bearing_oneshot_{run_id}"
    run_dir = Path(output_dir) / run_name

    # Load data
    data = load_bearing_test(data_path)

    if max_samples and max_samples < len(data):
        rng = random.Random(seed)
        # Stratified sampling by fault_type
        by_class = defaultdict(list)
        for i, s in enumerate(data):
            by_class[s["fault_type"]].append(i)
        indices = []
        per_class = max_samples // len(by_class)
        for cls, cls_indices in by_class.items():
            n = min(per_class, len(cls_indices))
            indices.extend(rng.sample(cls_indices, n))
        # Fill remaining
        remaining = max_samples - len(indices)
        if remaining > 0:
            all_remaining = [i for i in range(len(data)) if i not in set(indices)]
            indices.extend(rng.sample(all_remaining, min(remaining, len(all_remaining))))
        rng.shuffle(indices)
        data = [data[i] for i in indices]

    fault_dist = Counter(s["fault_type"] for s in data)

    print("=" * 60)
    print(f"GPT BEARING BASELINE — ONE-SHOT EVAL")
    print(f"  Model:       {model}")
    print(f"  Samples:     {len(data)}")
    print(f"  Max points:  {max_points}")
    print(f"  Classes:     {CLASSES}")
    print(f"  Distribution: {dict(fault_dist)}")
    print(f"  Output:      {run_dir}")
    print("=" * 60)

    # Run inference
    results = []
    errors = 0
    t0 = time.time()

    # Stream predictions to file
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_file = open(run_dir / "predictions.jsonl", "w", encoding="utf-8")

    for i, sample in enumerate(data):
        system, user = build_bearing_prompt(sample, max_points)

        try:
            raw_output = call_gpt(system, user, model=model)
        except Exception as e:
            print(f"  [{i}] API error: {e}")
            errors += 1
            if errors > 20:
                print("  Too many errors, stopping.")
                break
            time.sleep(2)
            continue

        pred = extract_answer(raw_output)
        gold = sample["fault_type"]
        correct = pred == gold

        result = {
            "idx": i,
            "gold": gold,
            "pred": pred,
            "correct": correct,
            "source": sample.get("source", ""),
            "rpm": sample.get("rpm", 0),
            "raw_output": raw_output,
        }
        results.append(result)

        # Stream to file
        pred_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        pred_file.flush()

        # Progress
        if i < 5 or i % 25 == 0 or not correct:
            tag = "OK" if correct else "WRONG"
            print(f"  [{i+1}/{len(data)}] [{tag}] gold={gold}, pred={pred}")

    pred_file.close()
    elapsed = time.time() - t0

    if not results:
        print("No results collected!")
        return

    # Compute metrics
    golds = [r["gold"] for r in results]
    preds = [r["pred"] for r in results]

    acc = accuracy_score(golds, preds)
    f1_w = f1_score(golds, preds, average="weighted", zero_division=0, labels=CLASSES)
    f1_m = f1_score(golds, preds, average="macro", zero_division=0, labels=CLASSES)

    print(f"\n{'=' * 60}")
    print(f"RESULTS — {model} — Bearing One-Shot")
    print(f"{'=' * 60}")
    print(f"  Accuracy:    {acc:.1%}")
    print(f"  F1 weighted: {f1_w:.1%}")
    print(f"  F1 macro:    {f1_m:.1%}")
    print(f"  Elapsed:     {elapsed:.0f}s ({elapsed/len(results):.1f}s/sample)")
    print(f"  API errors:  {errors}")

    # Per-class metrics
    print(f"\nClassification Report:")
    report = classification_report(golds, preds, labels=CLASSES, zero_division=0)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(golds, preds, labels=CLASSES)
    print("Confusion Matrix (rows=gold, cols=pred):")
    print(f"{'':>14} {'normal':>10} {'inner_race':>12} {'outer_race':>12} {'ball_fault':>12}")
    for i, cls in enumerate(CLASSES):
        row = "  ".join(f"{cm[i][j]:>10}" for j in range(len(CLASSES)))
        print(f"  {cls:>12} {row}")

    # Per-fault-type accuracy
    print(f"\nPer-class accuracy:")
    for cls in CLASSES:
        cls_results = [r for r in results if r["gold"] == cls]
        if cls_results:
            cls_acc = sum(r["correct"] for r in cls_results) / len(cls_results)
            print(f"  {cls:>12}: {cls_acc:.1%} ({sum(r['correct'] for r in cls_results)}/{len(cls_results)})")

    # Save metrics
    metrics = {
        "model": model,
        "n_samples": len(results),
        "n_errors": errors,
        "elapsed_sec": round(elapsed, 1),
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1_w, 4),
        "f1_macro": round(f1_m, 4),
        "per_class": {},
        "confusion_matrix": cm.tolist(),
        "class_order": CLASSES,
    }
    for cls in CLASSES:
        cls_results = [r for r in results if r["gold"] == cls]
        if cls_results:
            cls_preds = [r["pred"] for r in cls_results]
            cls_golds = [r["gold"] for r in cls_results]
            metrics["per_class"][cls] = {
                "n": len(cls_results),
                "accuracy": round(sum(r["correct"] for r in cls_results) / len(cls_results), 4),
            }

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Save examples (5 correct + 5 wrong per class)
    examples = defaultdict(lambda: {"correct": [], "wrong": []})
    for r in results:
        bucket = "correct" if r["correct"] else "wrong"
        if len(examples[r["gold"]][bucket]) < 5:
            examples[r["gold"]][bucket].append({
                "gold": r["gold"],
                "pred": r["pred"],
                "output_snippet": r["raw_output"][:500],
            })

    with open(run_dir / "examples.json", "w", encoding="utf-8") as f:
        json.dump(dict(examples), f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {run_dir}/")
    print(f"  metrics.json       — accuracy, F1, confusion matrix")
    print(f"  predictions.jsonl  — all {len(results)} predictions")
    print(f"  examples.json      — 5 correct + 5 wrong per class")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="GPT bearing baseline evaluation")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples (stratified). None = all 1208.")
    parser.add_argument("--max-points", type=int, default=512,
                        help="Max signal points per sample (downsample if longer)")
    parser.add_argument("--output-dir", default="results/gpt_bearing_baseline")
    parser.add_argument("--data-path", default=None,
                        help="Path to bearing_features_test.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_eval(
        model=args.model,
        max_samples=args.max_samples,
        max_points=args.max_points,
        output_dir=args.output_dir,
        data_path=args.data_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
