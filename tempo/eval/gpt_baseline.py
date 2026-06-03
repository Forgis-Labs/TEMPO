"""GPT baseline evaluation — text-serialized time series.

Sends raw time series values as text to GPT and evaluates on the same
benchmarks as TEMPO. This establishes the "LLM-without-tokenizer" baseline:
can a frontier LLM reason about time series from numbers alone?

Supported benchmarks:
  - TSQA (TSAQA-Benchmark): classification, anomaly, mcq, true_false, free_text
  - EngineMT-QA: understanding, perception, reasoning, decision-making
  - TS-Instruction Gold: instruction-following on time series

Metrics:
  - Classification/Anomaly: weighted F1, accuracy
  - MCQ: accuracy (exact match on letter)
  - True/False: accuracy
  - Free text: ROUGE-L, CoT quality (answer length, reasoning presence)

Usage:
    # Set your API key in tempo/.env:
    #   OPENAI_API_KEY=sk-...
    #   OPENAI_MODEL=gpt-4.1

    # Run on TSQA
    uv run python -m tempo.eval.gpt_baseline --dataset tsqa --max-samples 200

    # Run on all benchmarks
    uv run python -m tempo.eval.gpt_baseline --dataset all --max-samples 100

    # Custom model
    uv run python -m tempo.eval.gpt_baseline --model gpt-4.1-mini --max-samples 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Load .env
_env = Path(__file__).resolve().parent.parent.parent / ".env"
if _env.exists():
    for line in _env.read_text().strip().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


# 1. Time series formatting strategies

def ts_to_text_values(ts: list, max_points: int = 512) -> str:
    """Format time series as comma-separated values (truncated/downsampled)."""
    # Flatten if nested
    flat = np.array(ts, dtype=np.float64).ravel().tolist()
    if len(flat) > max_points:
        indices = np.linspace(0, len(flat) - 1, max_points).astype(int)
        flat = [flat[i] for i in indices]
    return ", ".join(f"{v:.4f}" for v in flat)


def ts_to_text_summary(ts: list[float]) -> str:
    """Statistical summary: mean, std, min, max, trend, length."""
    arr = np.array(ts, dtype=np.float64)
    n = len(arr)
    trend = "increasing" if arr[-1] > arr[0] + arr.std() * 0.5 else \
            "decreasing" if arr[-1] < arr[0] - arr.std() * 0.5 else "flat"
    return (f"Length: {n} points. "
            f"Mean: {arr.mean():.4f}, Std: {arr.std():.4f}, "
            f"Min: {arr.min():.4f}, Max: {arr.max():.4f}. "
            f"Overall trend: {trend}.")


def format_signal_for_gpt(
    ts_channels: list[list[float]],
    ts_labels: list[str],
    strategy: str = "values",
    max_points: int = 512,
) -> str:
    """Format time series for GPT prompt."""
    parts = []
    for i, (ch, label) in enumerate(zip(ts_channels, ts_labels)):
        if strategy == "values":
            parts.append(f"{label}\n[{ts_to_text_values(ch, max_points)}]")
        elif strategy == "summary":
            parts.append(f"{label}\n{ts_to_text_summary(ch)}")
        elif strategy == "both":
            parts.append(
                f"{label}\n{ts_to_text_summary(ch)}\nValues: [{ts_to_text_values(ch, max_points)}]"
            )
    return "\n".join(parts)


# 2. Task-specific prompts

FEW_SHOT_EXAMPLES = """
Example 1:
Signal: [0.12, 0.15, 0.13, 0.14, 0.82, 0.91, 0.85, 0.13, 0.14, 0.12]
Question: Does this signal contain an anomaly?
Answer: T
Reasoning: The values around indices 4-6 (0.82, 0.91, 0.85) are significantly higher than the baseline (~0.13), indicating a transient anomaly.

Example 2:
Signal: [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
Question: What is the trend of this signal?
Answer: The signal shows a clear linear increasing trend.

Example 3:
Signal: [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
Question: Is this signal periodic?
Answer: Yes, the signal oscillates with period 2 between values 1.0 and -1.0.
"""

SYSTEM_PROMPTS = {
    "classification": (
        "You are a time series analysis expert. You will be given numerical "
        "time series data and a classification question. Analyze the signal "
        "patterns carefully and provide your answer. End your response with "
        "'Answer: <your_answer>' on the last line.\n\n"
        "Here are some examples of how to analyze time series:\n" + FEW_SHOT_EXAMPLES
    ),
    "anomaly_detection": (
        "You are a time series anomaly detection expert. You will be given "
        "numerical time series data. Identify whether anomalies are present "
        "and describe them. End your response with 'Answer: <letter>' "
        "matching the correct option.\n\n"
        "Here are some examples:\n" + FEW_SHOT_EXAMPLES
    ),
    "mcq": (
        "You are a time series analysis expert. You will be given a time series "
        "and a multiple-choice question. Analyze the signal and select the best "
        "answer. End your response with 'Answer: X' where X is the letter (A/B/C/D).\n\n"
        "Here are some examples:\n" + FEW_SHOT_EXAMPLES
    ),
    "true_false": (
        "You are a time series analysis expert. Determine if the statement about "
        "the time series is true or false. End with 'Answer: True' or 'Answer: False'.\n\n"
        "Here are some examples:\n" + FEW_SHOT_EXAMPLES
    ),
    "free_text": (
        "You are a time series analysis expert. Analyze the given time series "
        "data and answer the question thoroughly with reasoning.\n\n"
        "Here are some examples:\n" + FEW_SHOT_EXAMPLES
    ),
    "engine_qa": (
        "You are an aero-engine diagnostics expert analyzing sensor data from "
        "a turbofan engine (N-CMAPSS simulation). The data contains normalized "
        "sensor readings across multiple channels. Analyze the patterns and "
        "answer the question. End with 'Answer: <your_answer>'."
    ),
}


SYSTEM_PROMPTS_ZERO_SHOT = {
    "classification": (
        "You are a time series analysis expert. You will be given numerical "
        "time series data and a classification question. Analyze the signal "
        "patterns carefully and provide your answer. End your response with "
        "'Answer: <your_answer>' on the last line."
    ),
    "anomaly_detection": (
        "You are a time series anomaly detection expert. You will be given "
        "numerical time series data. Identify whether anomalies are present "
        "and describe them. End your response with 'Answer: <letter>' "
        "matching the correct option."
    ),
    "mcq": (
        "You are a time series analysis expert. You will be given a time series "
        "and a multiple-choice question. Analyze the signal and select the best "
        "answer. End your response with 'Answer: X' where X is the letter (A/B/C/D)."
    ),
    "true_false": (
        "You are a time series analysis expert. Determine if the statement about "
        "the time series is true or false. End with 'Answer: True' or 'Answer: False'."
    ),
    "free_text": (
        "You are a time series analysis expert. Analyze the given time series "
        "data and answer the question thoroughly with reasoning."
    ),
    "engine_qa": (
        "You are an aero-engine diagnostics expert analyzing sensor data from "
        "a turbofan engine (N-CMAPSS simulation). The data contains normalized "
        "sensor readings across multiple channels. Analyze the patterns and "
        "answer the question. End with 'Answer: <your_answer>'."
    ),
}


def build_prompt(sample: dict, strategy: str = "values", max_points: int = 512,
                 few_shot: bool = True) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for a sample.

    Returns (system, user) message pair.
    """
    task = sample.get("task", "free_text")
    source = sample.get("source", "")
    prompts = SYSTEM_PROMPTS if few_shot else SYSTEM_PROMPTS_ZERO_SHOT

    # Pick system prompt — map TSQA task names to prompt categories
    task_to_prompt = {
        "classification": "classification",
        "anomaly_detection": "anomaly_detection",
        "characterization": "mcq",
        "comparison": "mcq",
        "data_transformation": "mcq",
        "temporal_relationship": "mcq",
    }
    if "engine" in source:
        system = prompts["engine_qa"]
    else:
        prompt_key = task_to_prompt.get(task, task)
        system = prompts.get(prompt_key, prompts["free_text"])

    # Build user message
    parts = []
    if sample.get("pre_prompt"):
        parts.append(sample["pre_prompt"])

    ts = sample.get("time_series", [])
    ts_text = sample.get("time_series_text", [])
    if ts and isinstance(ts[0], list) and len(ts[0]) > 0:
        if not ts_text:
            ts_text = [f"Channel {i}:" for i in range(len(ts))]
        parts.append(format_signal_for_gpt(ts, ts_text, strategy, max_points))

    if sample.get("post_prompt"):
        parts.append(sample["post_prompt"])

    return system, "\n\n".join(parts)


# 3. OpenAI API call

def call_gpt(
    system: str,
    user: str,
    model: str = "gpt-4.1",
    max_tokens: int = 800,
    temperature: float = 0.0,
) -> str:
    """Call OpenAI API. Returns response text."""
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


# 3b. Tool-use agent mode

TS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "compute_statistics",
            "description": "Compute summary statistics of a time series: mean, std, min, max, median, skewness, kurtosis, trend slope.",
            "parameters": {
                "type": "object",
                "properties": {
                    "values": {"type": "array", "items": {"type": "number"}, "description": "Time series values"}
                },
                "required": ["values"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_peaks",
            "description": "Find peaks and valleys in the time series. Returns indices, heights, and prominence of significant peaks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "values": {"type": "array", "items": {"type": "number"}, "description": "Time series values"},
                    "prominence": {"type": "number", "description": "Minimum peak prominence (default 0.5)", "default": 0.5},
                },
                "required": ["values"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_fft",
            "description": "Compute FFT to find dominant frequencies. Returns top-5 frequencies and their magnitudes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "values": {"type": "array", "items": {"type": "number"}, "description": "Time series values"},
                    "sampling_rate": {"type": "number", "description": "Sampling rate in Hz (default 1.0)", "default": 1.0},
                },
                "required": ["values"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_changepoints",
            "description": "Detect abrupt changes in signal level or variance. Returns indices where significant changes occur.",
            "parameters": {
                "type": "object",
                "properties": {
                    "values": {"type": "array", "items": {"type": "number"}, "description": "Time series values"},
                },
                "required": ["values"],
            },
        },
    },
]


def _exec_tool(name: str, args: dict) -> str:
    """Execute a tool call and return result as string."""
    vals = np.array(args.get("values", []), dtype=np.float64)
    if len(vals) == 0:
        return json.dumps({"error": "empty values"})

    if name == "compute_statistics":
        from scipy import stats as sp_stats
        slope = np.polyfit(np.arange(len(vals)), vals, 1)[0] if len(vals) > 1 else 0
        return json.dumps({
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
            "min": round(float(vals.min()), 4),
            "max": round(float(vals.max()), 4),
            "median": round(float(np.median(vals)), 4),
            "skewness": round(float(sp_stats.skew(vals)), 4),
            "kurtosis": round(float(sp_stats.kurtosis(vals)), 4),
            "trend_slope": round(float(slope), 6),
            "length": len(vals),
        })

    elif name == "detect_peaks":
        from scipy.signal import find_peaks as _find_peaks
        prom = args.get("prominence", 0.5)
        peak_idx, props = _find_peaks(vals, prominence=prom)
        valley_idx, _ = _find_peaks(-vals, prominence=prom)
        return json.dumps({
            "n_peaks": len(peak_idx),
            "peak_indices": peak_idx.tolist()[:10],
            "peak_values": [round(float(vals[i]), 4) for i in peak_idx[:10]],
            "n_valleys": len(valley_idx),
            "valley_indices": valley_idx.tolist()[:10],
        })

    elif name == "compute_fft":
        sr = args.get("sampling_rate", 1.0)
        fft_vals = np.abs(np.fft.rfft(vals - vals.mean()))
        freqs = np.fft.rfftfreq(len(vals), d=1.0/sr)
        top_idx = np.argsort(fft_vals)[-5:][::-1]
        return json.dumps({
            "dominant_frequencies": [round(float(freqs[i]), 4) for i in top_idx],
            "magnitudes": [round(float(fft_vals[i]), 4) for i in top_idx],
            "is_periodic": bool(fft_vals[1:].max() > fft_vals[1:].mean() * 5),
        })

    elif name == "detect_changepoints":
        # Simple: sliding window variance change
        w = max(10, len(vals) // 20)
        changes = []
        for i in range(w, len(vals) - w):
            v_before = vals[i-w:i].var()
            v_after = vals[i:i+w].var()
            if abs(v_after - v_before) > vals.var() * 0.5:
                if not changes or i - changes[-1] > w:
                    changes.append(i)
        return json.dumps({
            "n_changepoints": len(changes),
            "changepoint_indices": changes[:10],
        })

    return json.dumps({"error": f"unknown tool: {name}"})


def call_gpt_with_tools(
    system: str,
    user: str,
    model: str = "gpt-4.1",
    max_tokens: int = 800,
    temperature: float = 0.0,
) -> str:
    """Call OpenAI API with tool use. Model can call analysis tools, then answer."""
    from openai import OpenAI
    client = OpenAI()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    # Allow up to 3 rounds of tool calls
    for _ in range(3):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TS_TOOLS,
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                result = _exec_tool(tc.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            return msg.content.strip() if msg.content else ""

    # Final call without tools to force an answer
    response = client.chat.completions.create(
        model=model, messages=messages,
        max_completion_tokens=max_tokens, temperature=temperature,
    )
    return response.choices[0].message.content.strip()


# 4. Dataset loaders (test splits only)

def load_tsqa_test(max_samples: int = 500, seed: int = 42) -> list[dict]:
    """Load Time-MQA/TSQA benchmark (the held-out evaluation set).

    CSV files with QA_list column containing JSON question/answer pairs.
    Time series is embedded in the question text.
    Tasks: classification, anomaly_detection, open_ended_QA.
    """
    import random
    import re
    from huggingface_hub import hf_hub_download
    import pandas as pd

    rng = random.Random(seed)
    samples = []

    task_files = {
        "classification": "Classification/classification.csv",
        "anomaly_detection": "Anomaly_Detection/anomaly_detection.csv",
        "open_ended_qa": "Open_Ended_QA/open_ended_QA.csv",
    }

    for task_name, csv_path in task_files.items():
        print(f"  Loading Time-MQA/TSQA {task_name}...")
        try:
            local_path = hf_hub_download("Time-MQA/TSQA", csv_path, repo_type="dataset")
            df = pd.read_csv(local_path)

            # Sample
            n = min(max_samples // len(task_files), len(df))
            indices = rng.sample(range(len(df)), n)

            for idx in indices:
                row = df.iloc[idx]
                qa_raw = row.get("QA_list", "")
                domain = row.get("application_domain", "general")

                # Parse question and answer from the QA_list string
                q_match = re.search(r'"question"\s*:\s*"(.+?)"(?:\s*,\s*"answer")', qa_raw, re.DOTALL)
                a_match = re.search(r'"answer"\s*:\s*"(.+?)"', qa_raw, re.DOTALL)

                if not q_match or not a_match:
                    continue

                question = q_match.group(1).replace('\\"', '"')
                answer = a_match.group(1).replace('\\"', '"')

                # Extract time series from question text
                ts_match = re.search(r'\[([0-9eE+\-., \n]+)\]', question)
                ts = None
                if ts_match:
                    try:
                        ts = [float(x.strip()) for x in ts_match.group(1).split(",") if x.strip()]
                    except ValueError:
                        ts = None

                # Determine task type for open_ended_qa (has subtypes)
                actual_task = task_name
                if task_name == "open_ended_qa":
                    sub = row.get("task_type", "")
                    if sub:
                        actual_task = sub

                samples.append({
                    "time_series": [ts] if ts else [],
                    "time_series_text": ["Signal:"] if ts else [],
                    "pre_prompt": question,  # question already contains context + TS
                    "post_prompt": "",  # question is self-contained
                    "answer": answer,
                    "task": actual_task,
                    "domain": domain,
                    "source": "time-mqa",
                })
        except Exception as e:
            print(f"    {task_name}: SKIP ({e})")

    rng.shuffle(samples)
    print(f"    Loaded {len(samples)} Time-MQA samples")
    task_counts = Counter(s["task"] for s in samples)
    for t, c in task_counts.most_common():
        print(f"      {t}: {c}")
    return samples


def load_engine_qa_test(max_samples: int = 200, seed: int = 42) -> list[dict]:
    """Load EngineMT-QA test samples with actual sensor data from HDF5."""
    import random
    from datasets import load_dataset

    rng = random.Random(seed)
    samples = []

    print("  Loading pandalin98/EngineMT-QA...")
    try:
        ds = load_dataset("pandalin98/EngineMT-QA", split="test")
    except Exception:
        ds = load_dataset("pandalin98/EngineMT-QA", split="train")

    # Load HDF5 sensor data
    h5_data = None
    h5_id_to_idx = {}
    h5_paths = [
        Path(__file__).resolve().parent.parent.parent / "data" / "engine_qa" / "time_series_data.h5",
        Path("data/engine_qa/time_series_data.h5"),
        Path("tempo/data/engine_qa/time_series_data.h5"),
        Path.home() / "data" / "engine_qa" / "time_series_data.h5",
    ]
    for hp in h5_paths:
        if hp.exists():
            try:
                import h5py
                h5_data = h5py.File(str(hp), "r")
                data_ids = h5_data["data_ID"][:]
                for i, did in enumerate(data_ids):
                    h5_id_to_idx[str(int(did))] = i
                print(f"    HDF5: {len(h5_id_to_idx)} sequences from {hp}")
                break
            except Exception as e:
                print(f"    HDF5 error: {e}")

    if h5_data is None:
        print("    WARNING: No HDF5 found — engine QA will have no time series data")

    indices = rng.sample(range(len(ds)), min(max_samples, len(ds)))
    for idx in indices:
        row = ds[idx]
        convs = row.get("conversations", [])
        if len(convs) < 2:
            continue

        human_msg = convs[0].get("value", "")
        gpt_msg = convs[1].get("value", "")
        stage = convs[0].get("stage", "1")
        task_map = {"1": "understanding", "2": "perception",
                    "3": "reasoning", "4": "decision-making"}

        # Load sensor data from HDF5
        ts_list = []
        ts_labels = []
        ts_ids = row.get("id", [])
        if isinstance(ts_ids, (int, float)):
            ts_ids = [str(int(ts_ids))]
        elif isinstance(ts_ids, str):
            ts_ids = [ts_ids]
        elif isinstance(ts_ids, list):
            ts_ids = [str(int(x)) if isinstance(x, (int, float)) else str(x) for x in ts_ids]

        if h5_data is not None and ts_ids:
            use_ids = ts_ids[:1] if stage in ("1", "2") else ts_ids[:3]
            for tid in use_ids:
                h5_idx = h5_id_to_idx.get(str(tid))
                if h5_idx is not None:
                    seq = h5_data["seq_data"][h5_idx]  # (600, 33)
                    for ch_idx in [0, 5, 10]:
                        if ch_idx < seq.shape[1]:
                            ts_list.append(seq[:, ch_idx].astype(np.float32).tolist())
                            ts_labels.append(f"Sensor ch{ch_idx}:")

        human_msg = human_msg.replace("<ts>", "[engine sensor data provided above]")

        samples.append({
            "time_series": ts_list,
            "time_series_text": ts_labels,
            "pre_prompt": "Aero-engine sensor data from N-CMAPSS turbofan simulation. Units: normalized sensor readings.",
            "post_prompt": human_msg,
            "answer": gpt_msg,
            "task": task_map.get(stage, "understanding"),
            "domain": "aero-engine",
            "source": "engine-qa",
        })

    if h5_data is not None:
        h5_data.close()

    has_ts = sum(1 for s in samples if s["time_series"])
    print(f"    Loaded {len(samples)} EngineMT-QA samples ({has_ts} with TS)")
    return samples


# 5. Scoring

def extract_gpt_answer(text: str, task: str) -> str:
    """Extract answer from GPT response (mirrors scorer.py logic)."""
    import re

    # "Answer: X" pattern
    m = re.search(r"Answer:\s*(.+?)(?:\.|$|\n)", text, re.IGNORECASE)
    if m:
        ans = m.group(1).strip().rstrip(".")
        if task == "mcq":
            letter = re.search(r"\b([A-D])\b", ans)
            return letter.group(1) if letter else ans
        return ans

    # MCQ fallback: find standalone letter
    if task == "mcq":
        m = re.search(r"\b([A-D])\b", text)
        return m.group(1) if m else ""

    # Binary fallback
    if task in ("anomaly_detection", "true_false"):
        text_lower = text.lower()
        if "true" in text_lower:
            return "True"
        if "false" in text_lower:
            return "False"
        if "yes" in text_lower:
            return "yes"
        if "no" in text_lower:
            return "no"

    return text.strip().split("\n")[0][:200]


def compute_metrics(results: list[dict]) -> dict:
    """Compute per-task and overall metrics."""
    from sklearn.metrics import f1_score, accuracy_score

    task_results = defaultdict(list)
    for r in results:
        task_results[r["task"]].append(r)

    metrics = {}
    all_correct = []

    # Tasks with extractable answers (exact match scoring)
    MCQ_TASKS = {"classification", "anomaly_detection", "mcq", "true_false",
                 "characterization", "comparison", "data_transformation",
                 "temporal_relationship", "peak_identification"}

    for task, task_items in sorted(task_results.items()):
        golds = [r["gold_answer"] for r in task_items]
        preds = [r["pred_answer"] for r in task_items]

        if task in MCQ_TASKS:
            golds_n = [g.lower().strip() for g in golds]
            preds_n = [p.lower().strip() for p in preds]
            correct = [g == p for g, p in zip(golds_n, preds_n)]
            acc = sum(correct) / len(correct)

            try:
                f1_w = f1_score(golds_n, preds_n, average="weighted", zero_division=0)
                f1_m = f1_score(golds_n, preds_n, average="macro", zero_division=0)
            except Exception:
                f1_w = f1_m = 0.0

            metrics[task] = {
                "accuracy": round(acc, 4),
                "f1_weighted": round(f1_w, 4),
                "f1_macro": round(f1_m, 4),
                "n": len(task_items),
            }
        else:
            # Free text / engine QA — measure answer length and presence of reasoning
            avg_len = np.mean([len(r["raw_output"]) for r in task_items])
            has_reasoning = sum(1 for r in task_items
                               if any(w in r["raw_output"].lower()
                                      for w in ["because", "therefore", "since",
                                                 "indicates", "suggests", "shows"]))
            metrics[task] = {
                "avg_answer_length": round(float(avg_len), 0),
                "reasoning_rate": round(has_reasoning / len(task_items), 4),
                "n": len(task_items),
            }
            correct = [False] * len(task_items)  # can't auto-score free text

        all_correct.extend(correct)

    # Overall (all MCQ-scorable tasks)
    scorable = [r for r in results if r["task"] in MCQ_TASKS]
    if scorable:
        golds_all = [r["gold_answer"].lower().strip() for r in scorable]
        preds_all = [r["pred_answer"].lower().strip() for r in scorable]
        metrics["overall_scorable"] = {
            "accuracy": round(sum(g == p for g, p in zip(golds_all, preds_all)) / len(scorable), 4),
            "n": len(scorable),
        }

    return metrics


# 6. Main

def run_eval(
    dataset: str = "tsqa",
    model: str = "gpt-4.1",
    max_samples: int = 200,
    strategy: str = "values",
    max_points: int = 512,
    output_dir: str = "results/gpt_baseline",
    seed: int = 42,
    mode: str = "direct",
    few_shot: bool = False,
):
    """Run GPT baseline evaluation.

    Args:
        mode: "direct" = raw text, "agent" = GPT with analysis tools
        few_shot: Include few-shot examples in system prompt
    """
    # Create unique run folder
    from datetime import datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    shot_tag = "fewshot" if few_shot else "zeroshot"
    run_name = f"{model}_{mode}_{shot_tag}_{strategy}_{run_id}"
    run_dir = Path(output_dir) / run_name

    print("=" * 60)
    print(f"GPT BASELINE EVALUATION")
    print(f"  Model:    {model}")
    print(f"  Mode:     {mode}")
    print(f"  Few-shot: {few_shot}")
    print(f"  Dataset:  {dataset}")
    print(f"  Strategy: {strategy}")
    print(f"  Samples:  {max_samples}")
    print(f"  Output:   {run_dir}")
    print("=" * 60)

    # Load data
    all_samples = []
    if dataset in ("tsqa", "all"):
        all_samples.extend(load_tsqa_test(max_samples, seed))
    if dataset in ("engine", "all"):
        all_samples.extend(load_engine_qa_test(max_samples, seed))

    if not all_samples:
        print("ERROR: No samples loaded")
        return

    print(f"\nTotal: {len(all_samples)} samples")

    # Pick call function based on mode
    call_fn = call_gpt_with_tools if mode == "agent" else call_gpt

    # Run inference
    results = []
    errors = 0
    t0 = time.time()

    for i, sample in enumerate(all_samples):
        system, user = build_prompt(sample, strategy, max_points, few_shot=few_shot)

        try:
            raw_output = call_fn(system, user, model=model)
        except Exception as e:
            print(f"  [{i}] API error: {e}")
            errors += 1
            if errors > 10:
                print("  Too many errors, stopping.")
                break
            time.sleep(2)
            continue

        pred = extract_gpt_answer(raw_output, sample["task"])
        gold = sample["answer"]

        # For classification/mcq/tf, extract gold answer too
        gold_extracted = extract_gpt_answer(gold, sample["task"]) if sample["task"] in (
            "classification", "anomaly_detection", "mcq", "true_false") else gold

        correct = pred.lower().strip() == gold_extracted.lower().strip()

        results.append({
            "task": sample["task"],
            "domain": sample.get("domain", ""),
            "source": sample.get("source", ""),
            "gold_raw": gold[:500],
            "gold_answer": gold_extracted,
            "pred_answer": pred,
            "raw_output": raw_output,
            "correct": correct,
        })

        # Progress
        if i < 10 or i % 50 == 0:
            tag = "OK" if correct else "WRONG"
            print(f"  [{i}/{len(all_samples)}] [{tag}] {sample['task']}: "
                  f"gold={gold_extracted[:50]}, pred={pred[:50]}")

    elapsed = time.time() - t0
    print(f"\nInference: {elapsed:.0f}s ({elapsed/max(len(results),1):.1f}s/sample), "
          f"{errors} API errors")

    # Compute metrics
    metrics = compute_metrics(results)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"RESULTS — {model}")
    print(f"{'=' * 60}")
    for task, m in sorted(metrics.items()):
        if "accuracy" in m:
            f1_str = f", F1w={m['f1_weighted']:.1%}" if "f1_weighted" in m else ""
            print(f"  {task:<25} Acc={m['accuracy']:.1%}{f1_str}  (n={m['n']})")
        elif "reasoning_rate" in m:
            print(f"  {task:<25} AvgLen={m['avg_answer_length']:.0f}, "
                  f"Reasoning={m['reasoning_rate']:.0%}  (n={m['n']})")

    # Save to unique run folder
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": model,
            "mode": mode,
            "dataset": dataset,
            "strategy": strategy,
            "max_samples": max_samples,
            "n_evaluated": len(results),
            "elapsed_sec": round(elapsed, 1),
            "metrics": metrics,
        }, f, indent=2, ensure_ascii=False)

    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save detailed examples per task (5 correct + 5 wrong for each)
    examples = defaultdict(lambda: {"correct": [], "wrong": []})
    for r in results:
        bucket = "correct" if r["correct"] else "wrong"
        if len(examples[r["task"]][bucket]) < 5:
            examples[r["task"]][bucket].append({
                "gold": r["gold_answer"],
                "pred": r["pred_answer"],
                "output_snippet": r["raw_output"][:500],
            })

    with open(run_dir / "examples.json", "w", encoding="utf-8") as f:
        json.dump(dict(examples), f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {run_dir}/")
    print(f"  metrics.json      — per-task accuracy/F1")
    print(f"  predictions.jsonl — all {len(results)} predictions")
    print(f"  examples.json     — 5 correct + 5 wrong per task")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="GPT baseline evaluation")
    parser.add_argument("--dataset", default="tsqa", choices=["tsqa", "engine", "all"])
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1"))
    parser.add_argument("--mode", default="direct", choices=["direct", "agent"],
                        help="direct = raw text prompt, agent = GPT with analysis tools")
    parser.add_argument("--few-shot", action="store_true",
                        help="Include few-shot examples in system prompt")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--strategy", default="values",
                        choices=["values", "summary", "both"],
                        help="How to serialize TS: raw values, stats summary, or both")
    parser.add_argument("--max-points", type=int, default=512,
                        help="Max data points per channel (downsample if longer)")
    parser.add_argument("--output-dir", default="results/gpt_baseline")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_eval(
        dataset=args.dataset,
        model=args.model,
        max_samples=args.max_samples,
        strategy=args.strategy,
        max_points=args.max_points,
        output_dir=args.output_dir,
        seed=args.seed,
        mode=args.mode,
        few_shot=args.few_shot,
    )


if __name__ == "__main__":
    main()
