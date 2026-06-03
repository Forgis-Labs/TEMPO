"""Stage 1 dataset builder — uses ONLY public, pre-existing datasets.

No synthetic generation, no custom templates. Just:
  1. OpenTSLM TSQA (ChengsenWang/TSQA) → MCQ about TS patterns (stage1_mcq)
  2. M4 Caption Dataset → signal description/captioning (stage2_captioning)
  3. UCR/UEA archive → classification (via aeon)
  4. ETT/Weather/Electricity/Traffic → forecasting as code generation
  5. TSQA (Time-MQA) → pre-made QA pairs (classification, anomaly, open-ended)
  6. OpenOrca → instruction following (proportional to TS data)

All datasets are publicly available. Anyone can reproduce Stage 1
by running this script.

Requires: pip install aeon

Usage:
    from tempo.data.stage1 import build_stage1_dataset
    dataset = build_stage1_dataset(totem_ckpt="totem.pt")
    dataset.save_to_disk("data/stage1")
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import torch
from datasets import Dataset, DatasetDict

if TYPE_CHECKING:
    from tempo.model.totem import TOTEMTokenizer


# OpenTSLM TSQA — MCQ about time series patterns (stage1_mcq)

def load_opentslm_tsqa(
    max_samples: int = 10_000,
    seed: int = 42,
) -> list[dict]:
    """Load OpenTSLM's TSQA dataset (ChengsenWang/TSQA).

    Simple MCQs like "is this signal trending up or down?" that teach
    the model to read TOTEM codes. This was stage1_mcq in the original
    OpenTSLM curriculum and produced 93% accuracy on Llama-1B.

    Uses only the train portion (first 80%).
    """
    import json
    from datasets import load_dataset

    TRAIN_FRAC = 0.8

    ds = load_dataset("ChengsenWang/TSQA", split="train")
    # Deterministic split
    n_train = int(len(ds) * TRAIN_FRAC)
    ds = ds.shuffle(seed=seed).select(range(n_train))

    if len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    samples = []
    for row in ds:
        question = row.get("Question", "")
        answer = row.get("Answer", "")
        series = row.get("Series", "")

        if not question or not answer:
            continue

        # Parse time series
        try:
            ts_values = json.loads(series) if isinstance(series, str) else series
            if isinstance(ts_values, list) and len(ts_values) > 0:
                # Flatten if nested
                if isinstance(ts_values[0], list):
                    ts_values = ts_values[0]
                ts_list = [float(v) for v in ts_values]
            else:
                ts_list = []
        except (json.JSONDecodeError, ValueError, TypeError):
            ts_list = []

        task_type = row.get("Task", "mcq")

        samples.append({
            "time_series": [ts_list] if ts_list else [],
            "time_series_text": ["Signal:"] if ts_list else [],
            "pre_prompt": "You are analyzing a time series signal.",
            "post_prompt": question,
            "answer": answer,
            "task": f"mcq_{task_type}" if task_type else "mcq",
            "domain": "general",
            "source": "OpenTSLM-TSQA",
        })

    print(f"  OpenTSLM-TSQA: {len(samples)} train samples (MCQ)")
    return samples


# M4 Caption Dataset — signal description (stage2_captioning)

def load_m4_captions(
    max_samples: int = 10_000,
    seed: int = 42,
) -> list[dict]:
    """Load M4 time series caption dataset.

    M4 competition time series paired with LLM-generated captions.
    Teaches the model to describe signal patterns in natural language.
    This was stage2_captioning in the original OpenTSLM curriculum.

    Downloads from ETH Zurich Polybox if not cached.
    """
    import urllib.request
    import zipfile

    RELEASE_URL = "https://polybox.ethz.ch/index.php/s/MT3y9WdEebT8wfj/download/M4TimeSeriesCaptionDatasetV02.zip"
    CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "tempo", "m4_captions")
    ZIP_PATH = os.path.join(CACHE_DIR, "m4_captions.zip")

    # Download + extract if needed
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        print(f"  Downloading M4 captions from {RELEASE_URL}...")
        urllib.request.urlretrieve(RELEASE_URL, ZIP_PATH)
        print(f"  Extracting M4 captions...")
        import zipfile
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(CACHE_DIR)

    # M4 dataset has per-frequency files: m4_series_{freq}.csv + m4_captions_{freq}.csv
    # Each series CSV has columns [id, series] where series is a JSON list string.
    # Each caption CSV has columns [id, caption].
    # Load all frequencies, join series+captions by ID.
    import pandas as pd
    FREQUENCIES = ["Daily", "Hourly", "Monthly", "Quarterly", "Weekly", "Yearly"]

    # Find the extracted data directory
    data_dir = None
    for root, dirs, files in os.walk(CACHE_DIR):
        if any(f.startswith("m4_captions_") and f.endswith(".csv") for f in files):
            data_dir = root
            break

    if data_dir is None:
        print(f"  M4 captions: SKIP (no per-frequency CSVs found in {CACHE_DIR})")
        return []

    # Load and merge all frequencies
    series_dict = {}
    caption_dict = {}
    for freq in FREQUENCIES:
        series_file = os.path.join(data_dir, f"m4_series_{freq}.csv")
        caption_file = os.path.join(data_dir, f"m4_captions_{freq}.csv")
        if not os.path.exists(series_file) or not os.path.exists(caption_file):
            continue
        try:
            sdf = pd.read_csv(series_file)
            cdf = pd.read_csv(caption_file)
            for _, row in sdf.iterrows():
                sid = str(row["id"])
                raw = row["series"]
                values = json.loads(raw) if isinstance(raw, str) else []
                if values:
                    series_dict[sid] = [float(v) for v in values]
            for _, row in cdf.iterrows():
                sid = str(row["id"])
                caption_dict[sid] = str(row["caption"])
        except Exception as e:
            print(f"  M4 {freq}: SKIP ({e})")

    if not series_dict:
        print(f"  M4 captions: SKIP (no series data loaded)")
        return []

    # Match series with captions
    rng = random.Random(seed)
    common_ids = list(set(series_dict.keys()) & set(caption_dict.keys()))
    rng.shuffle(common_ids)
    if len(common_ids) > max_samples:
        common_ids = common_ids[:max_samples]

    questions = [
        "Describe what you see in this signal.",
        "Summarize the pattern of this time series.",
        "What are the key features of this data?",
        "Provide a caption for this time series.",
        "What does this signal look like?",
    ]

    samples = []
    for sid in common_ids:
        ts_list = series_dict[sid]
        caption = caption_dict[sid]
        if not caption or not ts_list:
            continue

        samples.append({
            "time_series": [ts_list],
            "time_series_text": ["Signal:"],
            "pre_prompt": "You are analyzing a time series signal.",
            "post_prompt": rng.choice(questions),
            "answer": caption,
            "task": "captioning",
            "domain": "general",
            "source": "M4-Captions",
        })

    print(f"  M4 Captions: {len(samples)} train samples (from {len(common_ids)} matched IDs)")
    return samples


# UCR classification (via aeon)

def load_ucr_classification(
    dataset_names: list[str] | None = None,
    max_per_dataset: int = 500,
    seed: int = 42,
) -> list[dict]:
    """Load UCR datasets as classification samples.

    Each sample has: time_series, pre_prompt, post_prompt, answer, task, label.
    Uses class names from UCR_SEMANTIC_METADATA when available.
    """
    from tempo.data.ucr import load_ucr_dataset, list_ucr_datasets, UCR_SEMANTIC_METADATA

    if dataset_names is None:
        # Use all datasets that aeon can load
        dataset_names = list_ucr_datasets()

    samples = []
    for name in dataset_names:
        try:
            ds = load_ucr_dataset(name, cot=True, seed=seed)
            # ONLY use train split — test is reserved for evaluation
            train = ds["train"]
            n = min(max_per_dataset, len(train))
            for i in range(n):
                row = train[i]
                samples.append({
                    "time_series": row["time_series"],
                    "time_series_text": row["time_series_text"],
                    "pre_prompt": row["pre_prompt"],
                    "post_prompt": row["post_prompt"],
                    "answer": row["answer"],
                    "task": "classification",
                    "domain": row.get("domain", "time series"),
                    "source": f"UCR/{name}",
                })
            print(f"  UCR/{name}: {n} train samples (test={len(ds['test'])} held out)")
        except Exception as e:
            print(f"  UCR/{name}: SKIP ({e})")

    return samples


# Forecasting as code generation (from raw time series)

def load_forecasting_samples(
    totem: "TOTEMTokenizer",
    max_samples: int = 50_000,
    window_size: int = 256,
    context_frac: float = 0.75,
    seed: int = 42,
) -> list[dict]:
    """Generate code→code forecasting samples from ETT/Weather/etc.

    Uses ONLY the train split from each HF dataset. The forecasting
    "labels" (horizon codes) are derived from the same signal, so
    there's no leakage — but we still only use the train portion to
    keep the temporal order consistent with evaluation.

    Context codes go in the input, horizon codes go in the answer
    wrapped in <ts_start>/<ts_end>. At inference the generated codes
    are decoded by TOTEM back to numerical values.
    """
    from datasets import load_dataset as hf_load

    rng = random.Random(seed)
    samples = []

    sources = {
        "ETTh1": ("thuml/Time-Series-Library", "ETTh1", "transformer temperature"),
        "ETTh2": ("thuml/Time-Series-Library", "ETTh2", "transformer temperature"),
        "weather": ("thuml/Time-Series-Library", "weather", "weather observation"),
        "electricity": ("thuml/Time-Series-Library", "electricity", "electricity consumption"),
        "traffic": ("thuml/Time-Series-Library", "traffic", "traffic flow"),
    }

    cf = totem.COMPRESSION_FACTOR

    for name, (repo, config, domain) in sources.items():
        if len(samples) >= max_samples:
            break
        try:
            ds = hf_load(repo, config, split="train")
        except Exception as e:
            print(f"  Forecast/{name}: SKIP ({e})")
            continue

        # Get numeric columns
        skip = {"date", "node_id"}
        columns = [c for c in ds.column_names if c not in skip]
        if len(columns) > 50:
            columns = rng.sample(columns, 50)

        n_from_source = 0
        for col in columns:
            if len(samples) >= max_samples:
                break
            try:
                values = np.array(ds[col], dtype=np.float64)
            except (ValueError, TypeError):
                continue
            if len(values) < window_size or np.std(values) < 1e-12:
                continue

            # Extract windows
            for start in range(0, len(values) - window_size, window_size // 2):
                if len(samples) >= max_samples:
                    break
                window = values[start:start + window_size]

                # Split into context and horizon
                split = int(len(window) * context_frac)
                split = (split // cf) * cf
                horizon_len = len(window) - split
                horizon_codes_n = horizon_len // cf
                if horizon_codes_n < 2 or split < cf * 4:
                    continue
                horizon_len = horizon_codes_n * cf

                context = window[:split]
                horizon = window[split:split + horizon_len]

                # Normalize with shared stats
                mean, std = window.mean(), window.std()
                if std < 1e-8:
                    continue
                context_norm = (context - mean) / std
                horizon_norm = (horizon - mean) / std

                # Tokenize horizon
                with torch.no_grad():
                    dev = next(totem.parameters()).device
                    h_tensor = torch.tensor(horizon_norm, dtype=torch.float32).unsqueeze(0).to(dev)
                    h_codes = totem.tokenize(h_tensor)
                horizon_text = " ".join(f"<ts_{c}>" for c in h_codes.squeeze(0).tolist())

                questions = [
                    "Forecast the continuation of this signal.",
                    "Predict what comes next.",
                    "Continue this time series.",
                    "Generate the next segment.",
                    "Extend this signal forward.",
                ]

                samples.append({
                    "time_series": [context_norm.tolist()],
                    "time_series_text": [f"{domain.capitalize()} signal:"],
                    "pre_prompt": f"You are analyzing {domain} data, sampled regularly.",
                    "post_prompt": rng.choice(questions),
                    "answer": f"<ts_start> {horizon_text} <ts_end>",
                    "task": "forecasting",
                    "domain": domain,
                    "source": f"forecast/{name}",
                })
                n_from_source += 1

        print(f"  Forecast/{name}: {n_from_source} samples")

    return samples


# TSQA (Time-MQA) — pre-made QA

def load_tsqa_samples(
    max_per_task: int = 10_000,
    seed: int = 42,
    hf_token: str | None = None,
) -> list[dict]:
    """Load TSQA benchmark samples (classification, anomaly, open-ended QA).

    Uses 80% of each TSQA file as training data with a fixed seed split.
    The remaining 20% is reserved for evaluation (same split used in
    our TSQA benchmark experiments).
    """
    import json, re
    from huggingface_hub import hf_hub_download
    import pandas as pd

    TRAIN_FRAC = 0.8  # 80% train, 20% held out for eval

    rng = random.Random(seed)
    samples = []

    task_files = {
        "classification": "Classification/classification.csv",
        "anomaly_detection": "Anomaly_Detection/anomaly_detection.csv",
        "open_ended_qa": "Open_Ended_QA/open_ended_QA.csv",
    }

    ts_pattern = re.compile(r'\[([0-9eE.,\s\-\+]+)\]')

    for task, path in task_files.items():
        try:
            csv_path = hf_hub_download("Time-MQA/TSQA", path, repo_type="dataset", token=hf_token)
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  TSQA/{task}: SKIP ({e})")
            continue

        # ONLY use train portion — hold out 20% for evaluation
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        train_end = int(len(df) * TRAIN_FRAC)
        df = df.iloc[:train_end]

        if len(df) > max_per_task:
            df = df.sample(n=max_per_task, random_state=seed)

        n = 0
        for _, row in df.iterrows():
            qa_raw = row.get("QA_list", "")
            if not isinstance(qa_raw, str) or not qa_raw:
                continue
            try:
                qa = json.loads(qa_raw if qa_raw.startswith("{") else "{" + qa_raw + "}")
            except (json.JSONDecodeError, ValueError):
                continue

            question = qa.get("question", "")
            answer = qa.get("answer", "")
            if not question or not answer:
                continue

            # Extract time series from question
            matches = list(ts_pattern.finditer(question))
            if matches:
                best = max(matches, key=lambda m: len(m.group(0)))
                try:
                    ts_values = [float(x.strip()) for x in best.group(1).split(",") if x.strip()]
                except ValueError:
                    continue
                text_before = question[:best.start()].strip()
                text_after = question[best.end():].strip()
            else:
                ts_values = []
                text_before = question
                text_after = ""

            domain = row.get("application_domain", "general")

            samples.append({
                "time_series": [ts_values] if ts_values else [],
                "time_series_text": ["Time series:"] if ts_values else [],
                "pre_prompt": text_before if text_before else f"Analyze this {domain} data.",
                "post_prompt": text_after if text_after else f"Answer the {task} question.",
                "answer": answer,
                "task": task,
                "domain": domain,
                "source": f"TSQA/{task}",
            })
            n += 1

        print(f"  TSQA/{task}: {n} samples")

    return samples


# Instruction data (OpenOrca)

def load_instruction_samples(
    n_samples: int = 50_000,
    hf_token: str | None = None,
) -> list[dict]:
    """Load OpenOrca text-only instruction samples."""
    from datasets import load_dataset

    ds = load_dataset("Open-Orca/OpenOrca", split=f"train[:{n_samples * 3}]", token=hf_token)

    samples = []
    for ex in ds:
        q = ex.get("question") or ex.get("prompt") or ""
        a = ex.get("response") or ex.get("answer") or ""
        if not q or not a:
            continue
        samples.append({
            "time_series": [],
            "time_series_text": [],
            "pre_prompt": q,
            "post_prompt": "",
            "answer": a,
            "task": "instruction",
            "domain": "general",
            "source": "OpenOrca",
        })
        if len(samples) >= n_samples:
            break

    print(f"  OpenOrca: {len(samples)} samples")
    return samples


# Build the full Stage 1 dataset

def build_stage1_dataset(
    totem_ckpt: str = "checkpoints/totem.pt",
    ucr_datasets: list[str] | None = None,
    max_ucr_per_dataset: int = 500,
    max_opentslm_tsqa: int = 10_000,
    max_m4_captions: int = 10_000,
    max_forecast: int = 50_000,
    max_tsqa_per_task: int = 10_000,
    instruction_ratio: float = 0.25,
    seed: int = 42,
    hf_token: str | None = None,
) -> DatasetDict:
    """Build the complete Stage 1 dataset from public sources.

    IMPORTANT: Only train splits from each source are used.
    - OpenTSLM TSQA: first 80% (train portion)
    - M4 Captions: train split
    - UCR: only TRAIN split (TEST held out for evaluate_ucr.py)
    - TSQA (Time-MQA): only first 80% (last 20% held out for TSQA eval)
    - Forecasting: only train split from HF datasets
    - OpenOrca: proportional to total TS data (default 15%)

    The returned DatasetDict has train/validation splits carved from
    the pooled training data. Per-dataset test evaluation should use
    the original test splits via tempo.eval.evaluate().

    Returns a HuggingFace DatasetDict ready to save_to_disk() or push_to_hub().
    """
    from tempo.model.totem import TOTEMTokenizer

    print("=" * 60)
    print("Building Stage 1 dataset (public sources only)")
    print("=" * 60)

    # Load TOTEM for forecasting code generation
    totem = TOTEMTokenizer.from_pretrained(totem_ckpt)

    all_ts_samples = []

    # 1. OpenTSLM TSQA — MCQ (stage1_mcq equivalent)
    print("\n[1/6] OpenTSLM TSQA (MCQ):")
    all_ts_samples.extend(load_opentslm_tsqa(max_opentslm_tsqa, seed))

    # 2. M4 Captions — signal description (stage2_captioning equivalent)
    print("\n[2/6] M4 Captions (signal description):")
    all_ts_samples.extend(load_m4_captions(max_m4_captions, seed))

    # 3. UCR classification
    print("\n[3/6] UCR Classification:")
    all_ts_samples.extend(load_ucr_classification(ucr_datasets, max_ucr_per_dataset, seed))

    # 4. Forecasting (code→code)
    print("\n[4/6] Forecasting (code generation):")
    all_ts_samples.extend(load_forecasting_samples(totem, max_forecast, seed=seed))

    # 5. TSQA (Time-MQA) QA
    print("\n[5/6] TSQA Time-MQA (pre-made QA):")
    all_ts_samples.extend(load_tsqa_samples(max_tsqa_per_task, seed, hf_token))

    # 6. Instruction — proportional to TS data
    n_instruction = int(len(all_ts_samples) * instruction_ratio)
    print(f"\n[6/6] Instruction (OpenOrca, {instruction_ratio:.0%} of {len(all_ts_samples)} TS samples = {n_instruction}):")
    instruction_samples = load_instruction_samples(n_instruction, hf_token)

    all_samples = all_ts_samples + instruction_samples

    # Shuffle and split into train/validation only.
    # Test data comes from original source test splits (UCR test, TSQA 20%, etc.)
    # and is loaded separately via tempo.eval.evaluate().
    rng = random.Random(seed)
    rng.shuffle(all_samples)

    n = len(all_samples)
    val_n = max(500, int(0.05 * n))
    val = all_samples[:val_n]
    train = all_samples[val_n:]

    print(f"\n{'=' * 60}")
    print(f"Total: {n} samples (train={len(train)}, val={val_n})")
    print(f"NOTE: Test data uses original source test splits (UCR test, TSQA 20%)")
    tasks = Counter(s["task"] for s in train)
    sources = Counter(s["source"].split("/")[0] for s in train)
    print(f"Tasks: {dict(tasks)}")
    print(f"Sources: {dict(sources)}")

    # Verify ts_tokens are sufficient
    ts_lengths = [len(s["time_series"][0]) if s["time_series"] else 0 for s in train]
    ts_with_data = [l for l in ts_lengths if l > 0]
    if ts_with_data:
        max_len = max(ts_with_data)
        median_len = sorted(ts_with_data)[len(ts_with_data) // 2]
        max_codes = max_len // 4
        print(f"TS lengths: median={median_len}, max={max_len} ({max_codes} codes at 4:1)")
        if max_codes > 375:
            print(f"  WARNING: some signals need {max_codes} codes > max_ts_tokens=375")

    # Convert to HF Dataset
    def to_hf_dict(samples):
        keys = ["time_series", "time_series_text", "pre_prompt", "post_prompt", "answer", "task", "domain", "source"]
        return {k: [s[k] for s in samples] for k in keys}

    return DatasetDict({
        "train": Dataset.from_dict(to_hf_dict(train)),
        "validation": Dataset.from_dict(to_hf_dict(val)),
    })


# Alignment dataset (stage0_align)

def _compute_description(signal: np.ndarray, task: str, domain: str, rng) -> dict | None:
    """Compute ground-truth description for a signal using analysis.py.

    Returns a make_qa_pair result dict, or None if analysis fails.
    """
    from tempo.data.analysis import analyze_trend, analyze_periodicity, analyze_anomalies
    from tempo.data.templates import make_qa_pair

    try:
        if task == "trend":
            gt = analyze_trend(signal)
        elif task == "describe":
            gt = analyze_trend(signal)
            period_info = analyze_periodicity(signal)
            gt["period"] = period_info.get("period", 0)
            cv = np.std(signal) / max(abs(np.mean(signal)), 1e-8)
            gt["cv_label"] = "low" if cv < 0.15 else "high" if cv > 0.5 else "moderate"
        elif task == "period":
            gt = analyze_periodicity(signal)
        elif task == "anomaly":
            gt = analyze_anomalies(signal)
        elif task == "volatility":
            cv = np.std(signal) / max(abs(np.mean(signal)), 1e-8)
            gt = {"label": "low" if cv < 0.15 else "high" if cv > 0.5 else "moderate"}
        elif task == "turning_points":
            from scipy.signal import find_peaks
            sig_range = float(signal.max() - signal.min())
            prom = 0.15 * sig_range if sig_range > 0 else 0.01
            p, _ = find_peaks(signal, prominence=prom)
            v, _ = find_peaks(-signal, prominence=prom)
            gt = {"count": len(p) + len(v)}
        else:
            return None
        return make_qa_pair(task, gt, domain, rng)
    except Exception:
        return None


_ALIGN_TASKS = ["trend", "describe", "period", "anomaly", "volatility", "turning_points"]


_CAPTION_QUESTIONS = [
    "Describe this signal.",
    "What does this data show?",
    "Summarize this time series.",
    "What is this signal?",
    "Caption this data.",
]


def _strip_answer_label(rationale: str) -> str:
    """Strip the final 'Answer: X' from a CoT rationale, keeping only the description."""
    import re
    # Remove "Answer: ..." at the end
    text = re.split(r"\bAnswer\s*:", rationale, flags=re.IGNORECASE)
    return text[0].strip().rstrip(".")


def _load_hf_sensor_data(
    max_samples: int = 5_000,
    seed: int = 42,
) -> list[dict]:
    """Load real sensor data from HuggingFace (ETT, Weather, Traffic, Electricity)."""
    rng = random.Random(seed)
    samples = []

    sources = [
        {
            "hf_repo": "thuml/Time-Series-Library", "hf_config": "ETTh1",
            "domain": "electrical engineering",
            "variants": [
                {"col": "HUFL", "text": "High useful load:", "context": "Transformer high useful load factor, recorded hourly."},
                {"col": "HULL", "text": "High useless load:", "context": "Transformer high useless load factor, sampled every hour."},
                {"col": "MUFL", "text": "Mid useful load:", "context": "Transformer mid-level useful load, recorded at 1-hour intervals."},
                {"col": "MULL", "text": "Mid useless load:", "context": "Transformer mid-level useless load factor, hourly measurements."},
                {"col": "LUFL", "text": "Low useful load:", "context": "Transformer low useful load, recorded hourly."},
                {"col": "LULL", "text": "Low useless load:", "context": "Transformer low useless load factor, sampled every hour."},
                {"col": "OT", "text": "Oil temperature:", "context": "Transformer oil temperature, recorded hourly. Units: °C."},
            ],
        },
        {
            "hf_repo": "thuml/Time-Series-Library", "hf_config": "ETTh2",
            "domain": "electrical engineering",
            "variants": [
                {"col": "HUFL", "text": "High useful load:", "context": "Transformer station 2 high useful load factor, hourly data."},
                {"col": "OT", "text": "Oil temperature:", "context": "Transformer oil temperature (station 2), recorded every hour. Units: °C."},
            ],
        },
        {
            "hf_repo": "thuml/Time-Series-Library", "hf_config": "weather",
            "domain": "meteorology",
            "variants": [
                {"col": "p (mbar)", "text": "Atmospheric pressure:", "context": "Barometric pressure from a weather station, sampled every 10 minutes. Units: mbar."},
                {"col": "T (degC)", "text": "Air temperature:", "context": "Air temperature, recorded at 10-minute intervals. Units: °C."},
                {"col": "rh (%)", "text": "Relative humidity:", "context": "Relative humidity from a weather station, sampled every 10 minutes. Units: %."},
                {"col": "wv (m/s)", "text": "Wind speed:", "context": "Wind velocity, recorded every 10 minutes at a meteorological station. Units: m/s."},
            ],
        },
        {
            "hf_repo": "thuml/Time-Series-Library", "hf_config": "electricity",
            "domain": "energy consumption",
            "variants": [
                {"col": None, "text": "Power consumption:", "context": "Household electricity consumption, recorded hourly. Units: kWh."},
            ],
        },
        {
            "hf_repo": "thuml/Time-Series-Library", "hf_config": "traffic",
            "domain": "transportation",
            "variants": [
                {"col": None, "text": "Road occupancy:", "context": "Road occupancy rate from a San Francisco freeway sensor, hourly measurements. Units: % (occupancy fraction)."},
            ],
        },
    ]

    window_size = 256

    for src in sources:
        if len(samples) >= max_samples:
            break
        try:
            from datasets import load_dataset as hf_load
            ds = hf_load(src["hf_repo"], src["hf_config"], split="train")
        except Exception as e:
            print(f"    {src['hf_config']}: SKIP ({e})")
            continue

        for variant in src["variants"]:
            if len(samples) >= max_samples:
                break

            col = variant["col"]
            if col and col in ds.column_names:
                all_values = ds[col]
            elif col is None:
                skip = {"date", "node_id"}
                cols = [c for c in ds.column_names if c not in skip]
                col = rng.choice(cols[:20]) if cols else None
                if not col:
                    continue
                all_values = ds[col]
            else:
                continue

            n_windows = min(300, max_samples // (len(sources) * 3))
            for _ in range(n_windows):
                if len(samples) >= max_samples:
                    break

                start = rng.randint(0, max(0, len(all_values) - window_size - 1))
                window = all_values[start:start + window_size]
                try:
                    signal = np.array([float(v) for v in window], dtype=np.float32)
                except (ValueError, TypeError):
                    continue
                if len(signal) < 64 or np.isnan(signal).any() or signal.std() < 1e-10:
                    continue

                task = rng.choice(_ALIGN_TASKS)
                qa = _compute_description(signal, task, src["domain"], rng)
                if qa is None:
                    continue

                samples.append({
                    "time_series": [signal.tolist()],
                    "time_series_text": [variant["text"]],
                    "pre_prompt": variant["context"],
                    "post_prompt": qa["question"],
                    "answer": qa["answer"],
                    "task": task,
                    "domain": src["domain"],
                    "source": f"align-hf/{src['hf_config']}",
                })

        n = sum(1 for s in samples if src['hf_config'] in s['source'])
        print(f"    {src['hf_config']}: {n} samples")

    print(f"  HF sensor data: {len(samples)} total")
    return samples


def _load_opentslm_signals(
    max_per_source: int = 2_000,
    seed: int = 42,
) -> list[dict]:
    """Extract real signals from OpenTSLM datasets (HAR, Sleep, ECG, Bearing).

    Uses the actual signals but replaces CoT answers with simple computed
    descriptions from analysis.py + templates.py.
    """
    rng = random.Random(seed)
    samples = []

    # --- HAR: 3-axis accelerometer, 50 Hz ---
    try:
        from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
        har = HARCoTQADataset("train", EOS_TOKEN="")
        n = min(max_per_source, len(har))
        indices = rng.sample(range(len(har)), n)
        har_labels = ["Accelerometer x-axis:", "Accelerometer y-axis:", "Accelerometer z-axis:"]
        for idx in indices:
            row = har[idx]
            ts = row.get("time_series", [])
            if not ts or len(ts) < 3:
                continue
            # Per-channel normalization
            channels = []
            valid = True
            for ch in ts[:3]:
                sig = np.array(ch, dtype=np.float32)
                if len(sig) < 20 or sig.std() < 1e-10:
                    valid = False
                    break
                sig = (sig - sig.mean()) / max(sig.std(), 1e-8)
                channels.append(sig.tolist())
            if not valid:
                continue

            rationale = row.get("answer", "")
            caption = _strip_answer_label(rationale)
            if not caption or len(caption) < 20:
                continue

            samples.append({
                "time_series": channels,
                "time_series_text": har_labels,
                "pre_prompt": "Wrist accelerometer data sampled at 50 Hz. Units: g (gravitational acceleration).",
                "post_prompt": rng.choice(_CAPTION_QUESTIONS),
                "answer": caption,
                "task": "captioning",
                "domain": "human activity recognition",
                "source": "align-har",
            })
        print(f"    HAR: {sum(1 for s in samples if s['source'] == 'align-har')} samples")
    except Exception as e:
        print(f"    HAR: SKIP ({e})")

    # --- SleepEDF: EEG at 50 Hz, 30s epochs (1500 samples) ---
    try:
        from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
        sleep = SleepEDFCoTQADataset("train", EOS_TOKEN="")
        n = min(max_per_source, len(sleep))
        indices = rng.sample(range(len(sleep)), n)
        for idx in indices:
            row = sleep[idx]
            ts = row.get("time_series", [])
            if not ts:
                continue
            signal = np.array(ts[0], dtype=np.float32)
            if len(signal) < 20 or signal.std() < 1e-10:
                continue

            rationale = row.get("answer", "")
            caption = _strip_answer_label(rationale)
            if not caption or len(caption) < 20:
                continue

            samples.append({
                "time_series": [signal.tolist()],
                "time_series_text": ["EEG signal:"],
                "pre_prompt": "A 30-second EEG segment recorded at 50 Hz. Units: µV (microvolts).",
                "post_prompt": rng.choice(_CAPTION_QUESTIONS),
                "answer": caption,
                "task": "captioning",
                "domain": "sleep EEG",
                "source": "align-sleep",
            })
        print(f"    Sleep: {sum(1 for s in samples if s['source'] == 'align-sleep')} samples")
    except Exception as e:
        print(f"    Sleep: SKIP ({e})")

    # --- ECG-QA: 12-lead ECG at 100 Hz ---
    try:
        from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
        ecg = ECGQACoTQADataset("train", EOS_TOKEN="")
        n = min(max_per_source, len(ecg))
        indices = rng.sample(range(len(ecg)), n)
        lead_names = ["Lead I", "Lead II", "Lead III", "aVR", "aVL", "aVF",
                       "V1", "V2", "V3", "V4", "V5", "V6"]
        for idx in indices:
            try:
                row = ecg[idx]
            except Exception:
                continue
            ts = row.get("time_series", [])
            if not ts:
                continue
            # Pick one lead (Lead II is most commonly analyzed)
            lead_idx = rng.randint(0, min(len(ts), 12) - 1)
            signal = np.array(ts[lead_idx], dtype=np.float32)
            if len(signal) < 20 or signal.std() < 1e-10:
                continue

            lead_name = lead_names[lead_idx] if lead_idx < len(lead_names) else f"Lead {lead_idx}"
            rationale = row.get("answer", "")
            caption = _strip_answer_label(rationale)
            if not caption or len(caption) < 20:
                continue

            samples.append({
                "time_series": [signal.tolist()],
                "time_series_text": [f"ECG {lead_name}:"],
                "pre_prompt": f"Electrocardiogram {lead_name} recording, sampled at 100 Hz. Units: mV (millivolts).",
                "post_prompt": rng.choice(_CAPTION_QUESTIONS),
                "answer": caption,
                "task": "captioning",
                "domain": "electrocardiography",
                "source": "align-ecg",
            })
        print(f"    ECG: {sum(1 for s in samples if s['source'] == 'align-ecg')} samples")
    except Exception as e:
        print(f"    ECG: SKIP ({e})")

    # --- PAMAP2: 3-axis accelerometer, IMU, 100 Hz ---
    try:
        from opentslm.time_series_datasets.pamap2.PAMAP2CoTQADataset import PAMAP2CoTQADataset
        pamap = PAMAP2CoTQADataset("train", EOS_TOKEN="")
        n = min(max_per_source, len(pamap))
        indices = rng.sample(range(len(pamap)), n)
        pamap_labels = ["IMU accelerometer x-axis:", "IMU accelerometer y-axis:", "IMU accelerometer z-axis:"]
        for idx in indices:
            row = pamap[idx]
            ts = row.get("time_series", [])
            if not ts:
                continue
            # Per-channel normalization — use up to 3 channels
            n_ch = min(len(ts), 3)
            channels = []
            valid = True
            for ch in ts[:n_ch]:
                sig = np.array(ch, dtype=np.float32)
                if len(sig) < 20 or sig.std() < 1e-10:
                    valid = False
                    break
                sig = (sig - sig.mean()) / max(sig.std(), 1e-8)
                channels.append(sig.tolist())
            if not valid or not channels:
                continue

            rationale = row.get("answer", "")
            caption = _strip_answer_label(rationale)
            if not caption or len(caption) < 20:
                continue

            samples.append({
                "time_series": channels,
                "time_series_text": pamap_labels[:n_ch],
                "pre_prompt": "Body-worn IMU accelerometer data at 100 Hz. Units: m/s² (meters per second squared).",
                "post_prompt": rng.choice(_CAPTION_QUESTIONS),
                "answer": caption,
                "task": "captioning",
                "domain": "physical activity monitoring",
                "source": "align-pamap2",
            })
        print(f"    PAMAP2: {sum(1 for s in samples if s['source'] == 'align-pamap2')} samples")
    except Exception as e:
        print(f"    PAMAP2: SKIP ({e})")

    # --- Bearing: CWRU vibration data at 12 kHz ---
    try:
        bearing_paths = [
            os.path.join(os.path.expanduser("~"), "data", "bearing_cot", "bearing_cot_train.json"),
            os.path.join(os.path.expanduser("~"), "data", "bearing_cot_large", "bearing_cot_train.json"),
            "data/bearing_cot/bearing_cot_train.json",
            "data/bearing_cot_large/bearing_cot_train.json",
        ]
        bearing_file = None
        for p in bearing_paths:
            if os.path.exists(p):
                bearing_file = p
                break

        if bearing_file:
            with open(bearing_file) as f:
                bearing_data = json.load(f)
            n = min(max_per_source, len(bearing_data))
            indices = rng.sample(range(len(bearing_data)), n)
            for idx in indices:
                row = bearing_data[idx]
                signal = np.array(row["signal"], dtype=np.float32)
                if len(signal) < 20 or signal.std() < 1e-10:
                    continue

                sensor = row.get("sensor", "DE")
                rpm = row.get("rpm", 1797)
                rationale = row.get("cot_gpt4o", row.get("cot", ""))
                caption = _strip_answer_label(rationale)
                if not caption or len(caption) < 20:
                    continue

                samples.append({
                    "time_series": [signal.tolist()],
                    "time_series_text": [f"Vibration signal from {sensor} accelerometer:"],
                    "pre_prompt": f"Bearing vibration data sampled at 12,000 Hz, motor at {rpm:.0f} RPM. Units: g (gravitational acceleration).",
                    "post_prompt": rng.choice(_CAPTION_QUESTIONS),
                    "answer": caption,
                    "task": "captioning",
                    "domain": "bearing vibration",
                    "source": "align-bearing",
                })
            print(f"    Bearing: {sum(1 for s in samples if s['source'] == 'align-bearing')} samples")
            del bearing_data  # Free memory
        else:
            print(f"    Bearing: SKIP (no data file found)")
    except Exception as e:
        print(f"    Bearing: SKIP ({e})")

    print(f"  OpenTSLM signals: {len(samples)} total")
    return samples


def _build_synthetic_signals(max_samples: int = 10_000, seed: int = 42) -> list[dict]:
    """Generate synthetic signals with precise, educational descriptions.

    Teaches the model the vocabulary of signal features: what a sine wave
    looks like in token space, what impulses are, what AM modulation is.
    These are the building blocks for describing real signals.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    samples = []

    # Realistic sampling rates to choose from
    SAMPLE_RATES = [50, 100, 250, 500, 1000, 5000, 12000]
    SIGNAL_LENGTHS = [128, 192, 256, 384, 512, 768, 1024]

    def _pick_length():
        return rng.choice(SIGNAL_LENGTHS)

    def _pick_sr():
        return rng.choice(SAMPLE_RATES)

    # Mutable container so generators see the updated length each call
    _NP = [256]
    def _set_length():
        _NP[0] = rng.choice(SIGNAL_LENGTHS)
    def _n():
        return _NP[0]

    # Each generator returns (signal, sampling_rate, description)
    # Generators use rng.choice() over multiple templates to vary descriptions.
    def gen_sine():
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        t = np.linspace(0, duration, n)
        max_freq = sr / 4
        f = rng.choice([f for f in [0.5, 1, 2, 5, 10, 20, 50, 100, 500, 1000] if f < max_freq])
        amp = round(np_rng.uniform(0.5, 3.0), 1)
        sig = amp * np.sin(2 * np.pi * f * t + np_rng.uniform(0, 2 * np.pi))
        n_cycles = round(f * duration, 1)
        period_s = round(1/f, 4) if f > 0 else 0
        desc = rng.choice([
            f"A pure sinusoidal signal at {f} Hz with amplitude {amp}, sampled at {sr} Hz over {duration:.3f}s. The waveform completes {n_cycles} cycles and is smooth and periodic with no harmonics.",
            f"This is a periodic signal. It oscillates at {f} Hz, meaning one full cycle every {period_s}s. The amplitude is {amp}. There are approximately {n_cycles} complete cycles visible.",
            f"A smooth, repetitive waveform characteristic of a sine wave. Frequency: {f} Hz. The signal repeats the same pattern {n_cycles} times. No sharp edges, no trend, no noise.",
            f"Periodic oscillation at {f} Hz. The signal goes up and down symmetrically with amplitude {amp}. Each cycle takes {period_s}s. This is the simplest periodic pattern possible.",
            f"The dominant feature is periodicity at {f} Hz ({period_s}s per cycle). The signal is perfectly smooth with constant amplitude {amp}. No trend, no anomalies, no frequency changes.",
        ])
        return sig, sr, desc

    def gen_dual_sine():
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        t = np.linspace(0, duration, n)
        max_freq = sr / 4
        low_freqs = [f for f in [1, 2, 3, 5, 8] if f < max_freq]
        high_freqs = [f for f in [10, 15, 20, 30, 50, 100] if f < max_freq]
        if not low_freqs or not high_freqs:
            low_freqs, high_freqs = [1], [5]
        f1, f2 = rng.choice(low_freqs), rng.choice(high_freqs)
        a1, a2 = round(np_rng.uniform(0.5, 1.5), 1), round(np_rng.uniform(0.3, 1.0), 1)
        sig = a1 * np.sin(2 * np.pi * f1 * t) + a2 * np.sin(2 * np.pi * f2 * t)
        desc = rng.choice([
            f"A signal composed of two sinusoids at {f1} Hz (amplitude {a1}) and {f2} Hz (amplitude {a2}), sampled at {sr} Hz over {duration:.3f}s.",
            f"Two frequencies superimposed: a slow {f1} Hz component (amplitude {a1}) and a fast {f2} Hz component (amplitude {a2}). The result is a complex periodic waveform.",
            f"Multi-frequency signal. The dominant oscillation is at {f1} Hz with a higher-frequency {f2} Hz modulation. Both components are sinusoidal.",
            f"This is not a simple sine wave — it contains two distinct frequencies: {f1} Hz and {f2} Hz. The slower component has amplitude {a1}, the faster has {a2}.",
        ])
        return sig, sr, desc

    def gen_step():
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        pos = np_rng.randint(n // 4, 3 * n // 4)
        height = round(np_rng.uniform(0.5, 3.0), 1)
        direction = rng.choice(["upward", "downward"])
        sig = np.zeros(n)
        sig[pos:] = height if direction == "upward" else -height
        sig += np_rng.normal(0, 0.05, n)
        t_step = round(pos / sr, 4)
        pct = round(100 * pos / n)
        desc = rng.choice([
            f"A step function with an abrupt {direction} level change of {height} at t={t_step}s. Sampled at {sr} Hz over {duration:.3f}s.",
            f"The signal is flat near zero, then suddenly jumps {'up' if direction == 'upward' else 'down'} by {height} at {pct}% of the way through. This is a classic step change or regime shift.",
            f"An abrupt transition occurs at t={t_step}s. Before: level ~0. After: level ~{height if direction == 'upward' else -height}. The change is instantaneous with no gradual ramp.",
            f"This looks like a switch being {'turned on' if direction == 'upward' else 'turned off'}. The signal holds at zero, then at time {t_step}s it shifts to {height if direction == 'upward' else -height} and stays there.",
            f"A single discontinuity at sample {pos} (t={t_step}s). The signal transitions from baseline to a new level {height} {'above' if direction == 'upward' else 'below'}. No oscillation, no gradual change.",
        ])
        return sig, sr, desc

    def gen_ramp():
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        t = np.linspace(0, duration, n)
        slope = round(np_rng.uniform(0.5, 3.0), 1)
        direction = rng.choice(["upward", "downward"])
        sig = (slope if direction == "upward" else -slope) * t / duration + np_rng.normal(0, 0.1, n)
        start_val = round(float(sig[0]), 2)
        end_val = round(float(sig[-1]), 2)
        desc = rng.choice([
            f"A linear ramp with a steady {direction} trend, sampled at {sr} Hz over {duration:.3f}s.",
            f"The signal increases {'steadily' if direction == 'upward' else 'never — it decreases steadily'} from {start_val} to {end_val}. This is a pure linear trend with no oscillation.",
            f"A monotonic {direction} signal. The value changes at a constant rate (slope {slope}). No periodicity, no steps, just a straight line with minor noise.",
            f"Linear trend going {'up' if direction == 'upward' else 'down'}. Start: {start_val}, end: {end_val}. The rate of change is constant throughout the signal.",
            f"This is not periodic. The signal moves in one direction only: {direction}. It's a ramp or linear drift from {start_val} to {end_val} over {duration:.3f}s.",
        ])
        return sig, sr, desc

    def gen_impulse():
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        sig = np_rng.normal(0, 0.1, n)
        n_spikes = np_rng.randint(1, 5)
        positions = sorted(np_rng.randint(10, n - 10, n_spikes))
        amp = round(np_rng.uniform(2, 5), 1)
        for pos in positions:
            sig[pos] += amp * rng.choice([-1, 1])
        times = [round(p / sr, 3) for p in positions]
        desc = rng.choice([
            f"A signal with {n_spikes} sharp impulse spike{'s' if n_spikes > 1 else ''} of amplitude {amp}, sampled at {sr} Hz over {duration:.3f}s.",
            f"Mostly quiet background noise with {n_spikes} sudden spike{'s' if n_spikes > 1 else ''} at time{'s' if n_spikes > 1 else ''} {times}s. Each spike reaches amplitude {amp}. These are anomalies or transient events.",
            f"The signal is flat except for {n_spikes} sharp peak{'s' if n_spikes > 1 else ''}. The spikes are very brief (single-sample) and much larger ({amp}x) than the background noise.",
            f"Impulsive events detected at {times}s. The baseline is near-zero noise, with {n_spikes} sudden outlier{'s' if n_spikes > 1 else ''} of magnitude {amp}.",
        ])
        return sig, sr, desc

    def gen_periodic_impulse():
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        sig = np_rng.normal(0, 0.1, n)
        period = np_rng.randint(15, 40)
        freq_hz = round(sr / period, 1)
        amp = round(np_rng.uniform(1.5, 4.0), 1)
        count = 0
        for i in range(0, n, period):
            if i < n:
                sig[i] += amp
                for j in range(1, min(8, n - i)):
                    sig[i + j] += amp * 0.6 ** j
                count += 1
        period_s = round(period / sr, 4)
        desc = rng.choice([
            f"Periodic impulses at {freq_hz} Hz (every {period} samples) with amplitude {amp} and exponential ring-down. Sampled at {sr} Hz over {duration:.3f}s. {count} impulses total.",
            f"A train of {count} equally-spaced sharp peaks, one every {period_s}s ({freq_hz} Hz). Each peak has amplitude {amp} followed by a rapid exponential decay. The background is quiet.",
            f"Repeating transient events at regular intervals of {period_s}s. Each event is a sharp spike (amplitude {amp}) that rings down exponentially. {count} events visible in {duration:.3f}s.",
            f"Periodic spikes with ring-down. The repetition rate is {freq_hz} Hz. Unlike a sine wave, the signal is mostly near zero with brief sharp peaks every {period} samples.",
        ])
        return sig, sr, desc

    def gen_am():
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        t = np.linspace(0, duration, n)
        max_freq = sr / 4
        f_carrier = round(min(np_rng.uniform(10, 30), max_freq), 0)
        f_mod = round(np_rng.uniform(0.5, 3), 1)
        depth = round(np_rng.uniform(0.3, 0.9), 1)
        sig = (1 + depth * np.sin(2 * np.pi * f_mod * t)) * np.sin(2 * np.pi * f_carrier * t)
        desc = rng.choice([
            f"An amplitude-modulated signal with {f_carrier:.0f} Hz carrier modulated at {f_mod} Hz (depth {depth}). Sampled at {sr} Hz over {duration:.3f}s.",
            f"The signal oscillates rapidly at {f_carrier:.0f} Hz, but the amplitude of those oscillations slowly varies at {f_mod} Hz. This is AM modulation with depth {depth}.",
            f"Two frequencies are present but in a multiplicative relationship: a fast {f_carrier:.0f} Hz carrier whose envelope pulsates at {f_mod} Hz. The modulation depth is {depth}.",
            f"Amplitude modulation: the signal's envelope goes up and down at {f_mod} Hz while the underlying oscillation runs at {f_carrier:.0f} Hz. Think of a beating or wobbling tone.",
        ])
        return sig, sr, desc

    def gen_chirp():
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        t = np.linspace(0, duration, n)
        max_freq = sr / 4
        f0 = round(np_rng.uniform(1, 5), 0)
        f1 = round(min(np_rng.uniform(20, 50), max_freq), 0)
        sig = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * duration) * t ** 2))
        desc = rng.choice([
            f"A linear chirp sweeping from {f0:.0f} Hz to {f1:.0f} Hz over {duration:.3f}s, sampled at {sr} Hz.",
            f"The frequency changes continuously: it starts slow at {f0:.0f} Hz and accelerates to {f1:.0f} Hz. The oscillations get closer together as time progresses.",
            f"A frequency sweep. At the beginning the signal oscillates slowly ({f0:.0f} Hz), by the end it oscillates rapidly ({f1:.0f} Hz). The transition is linear in frequency.",
            f"Unlike a fixed-frequency sine wave, this signal's frequency increases over time from {f0:.0f} to {f1:.0f} Hz. The waveform looks compressed toward the end.",
        ])
        return sig, sr, desc

    def gen_decay():
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        t = np.linspace(0, duration, n)
        tau = round(duration * np_rng.uniform(0.1, 0.5), 4)
        max_freq = sr / 4
        f = round(min(np_rng.uniform(5, 20), max_freq), 0)
        sig = np.exp(-t / tau) * np.sin(2 * np.pi * f * t)
        desc = rng.choice([
            f"A damped oscillation at {f:.0f} Hz with time constant {tau}s, sampled at {sr} Hz over {duration:.3f}s. The amplitude decays exponentially.",
            f"The signal oscillates at {f:.0f} Hz but the amplitude shrinks over time. This is exponential decay — the signal rings down and approaches zero. Time constant: {tau}s.",
            f"A decaying sinusoid. Starts with large oscillations that progressively get smaller. The frequency is {f:.0f} Hz, the decay rate is 1/{tau}s. Think of a struck bell or a damped spring.",
            f"Oscillation with decreasing envelope. The peaks get shorter each cycle. Frequency {f:.0f} Hz, decay constant {tau}s. After a few time constants, the signal is essentially zero.",
            f"Transient response: {f:.0f} Hz oscillation dying out exponentially (tau={tau}s). The signal is large at the start and decays to nothing.",
        ])
        return sig, sr, desc

    def gen_noise():
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        std = round(np_rng.uniform(0.3, 2.0), 1)
        sig = np_rng.normal(0, std, n)
        desc = rng.choice([
            f"Random white noise with std {std}, sampled at {sr} Hz over {duration:.3f}s. No discernible pattern or periodicity.",
            f"Pure noise. No trend, no periodicity, no structure. Each sample is independent. The amplitude varies randomly with standard deviation {std}.",
            f"An aperiodic, structureless signal. This is random noise — there is nothing to predict or classify. Std={std}.",
            f"This signal has no pattern. It is white noise with zero mean and std {std}. Unlike sine waves or steps, there is no repeating or systematic behavior.",
            f"Noise floor only. No signal content detected. The values are randomly distributed around zero with spread {std}.",
        ])
        return sig, sr, desc

    def gen_random_walk():
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        step_std = round(np_rng.uniform(0.05, 0.2), 2)
        sig = np.cumsum(np_rng.normal(0, step_std, n))
        final = round(float(sig[-1]), 1)
        sig_max = round(float(sig.max()), 1)
        sig_min = round(float(sig.min()), 1)
        desc = rng.choice([
            f"A random walk with step size {step_std}, sampled at {sr} Hz over {duration:.3f}s. The signal wanders unpredictably, ending at {final}.",
            f"Stochastic drift — each value is the previous plus a random increment (std={step_std}). The path wanders between {sig_min} and {sig_max}. Not periodic, not trending systematically.",
            f"This is a random walk, not a trend. Although it may appear to go up or down, the direction is random at each step. Final value: {final}. It could have ended anywhere.",
            f"Cumulative sum of random noise (step std={step_std}). Looks like a noisy trend but has no predictable direction. Range [{sig_min}, {sig_max}].",
        ])
        return sig, sr, desc

    def gen_square():
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        t = np.linspace(0, duration, n)
        max_freq = sr / 4
        f = round(min(np_rng.uniform(2, 10), max_freq), 0)
        n_cycles = round(f * duration, 1)
        sig = np.sign(np.sin(2 * np.pi * f * t)) + np_rng.normal(0, 0.05, n)
        desc = rng.choice([
            f"A square wave at {f:.0f} Hz alternating between +1 and -1, sampled at {sr} Hz over {duration:.3f}s.",
            f"Periodic signal that abruptly switches between two levels (+1 and -1) at {f:.0f} Hz. Unlike a sine wave, the transitions are instantaneous — the signal is always at one extreme or the other.",
            f"A {f:.0f} Hz square wave with {n_cycles} cycles. The waveform has sharp edges and flat tops/bottoms. Each half-cycle is a constant value.",
            f"Binary-like oscillation at {f:.0f} Hz. The signal spends equal time at +1 and -1 with abrupt transitions. No gradual curves like a sinusoid.",
        ])
        return sig, sr, desc

    def gen_sawtooth():
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        t = np.linspace(0, duration, n)
        max_freq = sr / 4
        f = round(min(np_rng.uniform(2, 8), max_freq), 0)
        n_cycles = round(f * duration, 1)
        sig = 2 * (t * f - np.floor(0.5 + t * f)) + np_rng.normal(0, 0.05, n)
        desc = rng.choice([
            f"A sawtooth wave at {f:.0f} Hz, sampled at {sr} Hz over {duration:.3f}s. Linear ramps followed by sharp drops.",
            f"Periodic ramp pattern at {f:.0f} Hz. The signal rises linearly, then drops sharply back down. {n_cycles} teeth visible. Each ramp takes {round(1/f, 4)}s.",
            f"A {f:.0f} Hz sawtooth: gradual linear increase followed by an instantaneous reset. The asymmetry between the slow rise and fast fall is the defining feature.",
            f"Repeating linear ramps with abrupt resets at {f:.0f} Hz. Unlike a sine wave (smooth) or square wave (flat tops), the sawtooth has a constant slope within each cycle.",
        ])
        return sig, sr, desc

    def gen_noisy_sine():
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        t = np.linspace(0, duration, n)
        max_freq = sr / 4
        f = round(min(np_rng.uniform(2, 15), max_freq), 0)
        noise_std = round(np_rng.uniform(0.05, 1.5), 2)
        snr = round(1.0 / max(noise_std, 0.01), 1)
        sig = np.sin(2 * np.pi * f * t) + np_rng.normal(0, noise_std, n)
        visibility = "clearly visible" if noise_std < 0.2 else "partially obscured" if noise_std < 0.7 else "barely distinguishable"
        desc = rng.choice([
            f"A {f:.0f} Hz sinusoid with additive noise (std={noise_std}, SNR~{snr}). The periodic pattern is {visibility} against the noise floor.",
            f"Noisy periodic signal. The underlying pattern is a {f:.0f} Hz sine wave, but noise (std={noise_std}) {'slightly' if noise_std < 0.2 else 'significantly'} obscures it. SNR is approximately {snr}.",
            f"A sine wave buried in noise. Frequency: {f:.0f} Hz. Noise level: {noise_std}. The periodicity is {visibility}. In a clean version, this would be a smooth oscillation.",
            f"Signal plus noise: the signal component oscillates at {f:.0f} Hz, the noise component has std={noise_std}. The SNR of {snr} means the signal is {'dominant' if snr > 5 else 'comparable to' if snr > 1 else 'weaker than'} the noise.",
        ])
        return sig, sr, desc

    def gen_burst():
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        sig = np_rng.normal(0, 0.05, n)
        start = np_rng.randint(n // 6, n // 2)
        dur_samples = np_rng.randint(n // 10, n // 4)
        max_freq = sr / 4
        f = round(min(np_rng.uniform(10, 30), max_freq), 0)
        amp = round(np_rng.uniform(1, 3), 1)
        t_burst = np.arange(dur_samples) / sr
        if start + dur_samples <= n:
            sig[start:start + dur_samples] += np.sin(2 * np.pi * f * t_burst) * amp
        t_start = round(start / sr, 4)
        t_dur = round(dur_samples / sr, 4)
        pct_start = round(100 * start / n)
        pct_dur = round(100 * dur_samples / n)
        desc = rng.choice([
            f"A transient burst of {f:.0f} Hz oscillations (amplitude {amp}) starting at t={t_start}s and lasting {t_dur}s. The rest is quiet background noise.",
            f"The signal is mostly silence, with a burst of activity from {pct_start}% to {pct_start + pct_dur}% of the recording. The burst oscillates at {f:.0f} Hz with amplitude {amp}.",
            f"A localized event: {f:.0f} Hz oscillations appear suddenly at t={t_start}s, last for {t_dur}s, then vanish. Before and after the burst, only low-level noise is present.",
            f"Transient oscillatory event embedded in noise. The burst at {f:.0f} Hz (amplitude {amp}) occupies only {pct_dur}% of the signal duration. The onset is at t={t_start}s.",
        ])
        return sig, sr, desc

    # --- Generators that teach mean/std interpretation ---

    def gen_offset_sine():
        """Sine with a DC offset — teaches that mean != 0 means a baseline shift."""
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        t = np.linspace(0, duration, n)
        max_freq = sr / 4
        f = rng.choice([f for f in [1, 2, 5, 10] if f < max_freq])
        offset = round(np_rng.uniform(-5, 5), 1)
        amp = round(np_rng.uniform(0.3, 1.5), 1)
        sig = offset + amp * np.sin(2 * np.pi * f * t) + np_rng.normal(0, 0.05, n)
        sig_mean = round(float(np.mean(sig)), 2)
        sig_std = round(float(np.std(sig)), 2)
        desc = rng.choice([
            f"A {f} Hz sinusoid with amplitude {amp} oscillating around a baseline of {offset}. The signal mean is {sig_mean}, indicating it is centered {'above' if offset > 0 else 'below'} zero. The std of {sig_std} reflects the oscillation amplitude.",
            f"Periodic oscillation at {f} Hz, but shifted {'upward' if offset > 0 else 'downward'} by {abs(offset)}. The mean is not zero ({sig_mean}) because of this DC offset. The oscillation amplitude is {amp}.",
            f"This looks like a sine wave at {f} Hz, but it doesn't oscillate around zero. Instead, the center line is at {offset}. The mean ({sig_mean}) tells you the offset, the std ({sig_std}) tells you the oscillation size.",
            f"A baseline-shifted sinusoid. Without the offset, this would oscillate symmetrically around zero at {f} Hz. The {offset} shift means all values are {'positive' if offset > 3 else 'negative' if offset < -3 else 'near ' + str(offset)}.",
        ])
        return sig, sr, desc

    def gen_varying_amplitude():
        """Signal whose std directly indicates intensity."""
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        n_half = n // 2
        std_low = round(np_rng.uniform(0.1, 0.5), 2)
        std_high = round(np_rng.uniform(1.5, 4.0), 2)
        sig = np.concatenate([
            np_rng.normal(0, std_low, n_half),
            np_rng.normal(0, std_high, n_half),
        ])
        overall_std = round(float(np.std(sig)), 2)
        ratio = round(std_high / std_low, 1)
        desc = rng.choice([
            f"A signal that transitions from low intensity (std~{std_low}) to high intensity (std~{std_high}) halfway through. The overall std is {overall_std}. The amplitude increase indicates a change in signal energy or source activity.",
            f"Two-segment signal: the first half is quiet (std={std_low}), the second half is {ratio}x louder (std={std_high}). This abrupt change in variability suggests a regime change or fault onset.",
            f"The signal's amplitude changes dramatically at the midpoint. Before: low-energy noise (std={std_low}). After: high-energy noise (std={std_high}). The energy ratio is {ratio}:1.",
            f"An intensity transition. The first half has small fluctuations (std={std_low}), the second half has large fluctuations (std={std_high}). This is not a trend — the mean stays near zero, but the spread changes.",
        ])
        return sig, sr, desc

    def gen_gravity_axis():
        """Simulates an accelerometer axis aligned with gravity — mean ~ +/-9.8."""
        sr = rng.choice([50, 100, 250])
        n = _n()
        duration = n / sr
        g_sign = rng.choice([-1, 1])
        noise = round(np_rng.uniform(0.05, 0.3), 2)
        sig = g_sign * 9.81 + np_rng.normal(0, noise, n)
        sig_mean = round(float(np.mean(sig)), 2)
        direction = "downward" if g_sign > 0 else "upward"
        desc = rng.choice([
            f"An accelerometer axis aligned with gravity. The mean of {sig_mean} g indicates the sensor measures the gravitational acceleration component along this axis ({direction}-pointing). The low std of {noise} g shows the sensor is nearly stationary.",
            f"Nearly constant signal at {sig_mean} with tiny fluctuations (std={noise}). This is an accelerometer measuring gravity — the large mean (~9.8) is the gravitational constant, the small noise is sensor jitter.",
            f"A flat signal near {sig_mean}. The value barely changes (std={noise}). This represents a stationary sensor aligned with gravity. The mean encodes orientation, the std encodes motion intensity.",
            f"Constant baseline at approximately {'+'if g_sign > 0 else '-'}9.8 g with noise std={noise}. The sensor is not moving — it only sees Earth's gravitational field along the {direction}-pointing axis.",
        ])
        return sig, sr, desc

    def gen_scaled_comparison():
        """Same shape, different scale — teaches that z-normalization removes scale."""
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        t = np.linspace(0, duration, n)
        f = rng.choice([2, 5, 10])
        base = np.sin(2 * np.pi * f * t)
        scale = round(np_rng.uniform(0.01, 100), 2)
        offset = round(np_rng.uniform(-50, 50), 1)
        sig = offset + scale * base + np_rng.normal(0, scale * 0.05, n)
        sig_mean = round(float(np.mean(sig)), 2)
        sig_std = round(float(np.std(sig)), 2)
        desc = rng.choice([
            f"A {f} Hz sinusoid scaled to amplitude {scale} with offset {offset}. The mean of {sig_mean} reflects the DC offset, and the std of {sig_std} reflects the signal amplitude. After z-normalization, the shape is identical to a unit-amplitude sinusoid.",
            f"Despite the unusual scale (mean={sig_mean}, std={sig_std}), this is fundamentally a {f} Hz sine wave. The large numbers come from scaling by {scale} and shifting by {offset}. The underlying pattern is periodic.",
            f"A sine wave at {f} Hz, but with extreme scaling: amplitude {scale}, offset {offset}. The raw values look nothing like a normal sine wave, but the normalized shape is identical. This teaches that scale doesn't change the pattern.",
            f"Rescaled periodic signal. Original: a simple {f} Hz sinusoid. After scaling by {scale}x and adding offset {offset}: mean={sig_mean}, std={sig_std}. The waveform shape is preserved despite the extreme values.",
        ])
        return sig, sr, desc

    # --- New generators: compound and real-world patterns ---

    def gen_trend_plus_oscillation():
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        t = np.linspace(0, duration, n)
        max_freq = sr / 4
        slope = round(np_rng.uniform(0.5, 3.0), 1)
        direction = rng.choice(["upward", "downward"])
        f = round(min(np_rng.uniform(2, 10), max_freq), 0)
        amp = round(np_rng.uniform(0.3, 1.0), 1)
        trend = (slope if direction == "upward" else -slope) * t / duration
        sig = trend + amp * np.sin(2 * np.pi * f * t) + np_rng.normal(0, 0.05, n)
        desc = rng.choice([
            f"A signal combining a {direction} linear trend (slope {slope}) with a {f:.0f} Hz oscillation (amplitude {amp}). The dominant feature is the {direction} trend with periodic fluctuations superimposed.",
            f"Two patterns at once: the signal is going {'up' if direction == 'upward' else 'down'} overall (trend), but also oscillating at {f:.0f} Hz. The trend has slope {slope}, the oscillation amplitude is {amp}.",
            f"This is not a pure trend and not a pure oscillation — it's both. The {direction} drift has a {f:.0f} Hz periodic component riding on top. Common in real-world sensor data.",
            f"Trend plus seasonality: a {direction} linear component (slope {slope}) combined with {f:.0f} Hz periodic fluctuations (amplitude {amp}). Removing the trend would reveal a clean sinusoid.",
        ])
        return sig, sr, desc

    def gen_multi_step():
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        n_steps = np_rng.randint(2, 5)
        positions = sorted(np_rng.randint(n // 8, 7 * n // 8, n_steps))
        levels = [0.0] + [round(np_rng.uniform(-3, 3), 1) for _ in range(n_steps)]
        sig = np.zeros(n)
        for i, pos in enumerate(positions):
            sig[pos:] = levels[i + 1]
        sig += np_rng.normal(0, 0.05, n)
        times = [round(p / sr, 3) for p in positions]
        desc = rng.choice([
            f"A signal with {n_steps} abrupt level changes at times {times}s, transitioning between levels {levels}. This pattern is typical of a multi-state system.",
            f"Multiple step changes: the signal holds steady at one level, then jumps to another, {n_steps} times total. The levels are {levels}. Each transition is instantaneous.",
            f"A {n_steps+1}-state signal. The system switches between discrete levels {levels} at times {times}s. Between transitions, the signal is flat with minor noise.",
            f"Piecewise constant signal with {n_steps} discontinuities. Unlike a ramp (gradual) or sine (oscillating), this signal only changes value at specific moments: {times}s.",
        ])
        return sig, sr, desc

    def gen_exponential_growth():
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        t = np.linspace(0, duration, n)
        direction = rng.choice(["growth", "decay"])
        rate = round(np_rng.uniform(1.0, 5.0), 1)
        if direction == "growth":
            sig = np.exp(rate * t / duration) - 1
        else:
            sig = np.exp(-rate * t / duration)
        sig += np_rng.normal(0, 0.02 * max(sig.std(), 0.01), n)
        start_val = round(float(sig[0]), 2)
        end_val = round(float(sig[-1]), 2)
        desc = rng.choice([
            f"An exponential {direction} signal with rate {rate}. The signal {'increases rapidly' if direction == 'growth' else 'decreases toward zero'} over time.",
            f"Non-linear {'increase' if direction == 'growth' else 'decrease'}. Unlike a linear ramp, the rate of change itself changes — {'accelerating' if direction == 'growth' else 'decelerating'}. Rate constant: {rate}.",
            f"Exponential curve from {start_val} to {end_val}. The {'growth' if direction == 'growth' else 'decay'} rate is {rate}. The signal {'doubles' if direction == 'growth' else 'halves'} repeatedly.",
            f"{'Explosive growth' if direction == 'growth' else 'Exponential decay'}: the signal {'gets bigger faster and faster' if direction == 'growth' else 'shrinks rapidly at first, then more slowly'}. Rate={rate}.",
        ])
        return sig, sr, desc

    def gen_plateau_ramp_plateau():
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        t1 = np_rng.randint(n // 6, n // 3)
        t2 = np_rng.randint(2 * n // 3, 5 * n // 6)
        level1 = round(np_rng.uniform(-1, 1), 1)
        level2 = round(np_rng.uniform(level1 + 0.5, level1 + 3), 1)
        sig = np.zeros(n)
        sig[:t1] = level1
        sig[t1:t2] = np.linspace(level1, level2, t2 - t1)
        sig[t2:] = level2
        sig += np_rng.normal(0, 0.05, n)
        t1_s, t2_s = round(t1 / sr, 3), round(t2 / sr, 3)
        ramp_dur = round(t2_s - t1_s, 3)
        desc = rng.choice([
            f"A three-phase signal: flat at {level1} until t={t1_s}s, then ramping {'up' if level2 > level1 else 'down'} to {level2} until t={t2_s}s, then flat again. Typical of an industrial process transition.",
            f"Plateau-ramp-plateau pattern. First steady at {level1}, then a gradual transition over {ramp_dur}s to level {level2}, then steady again. The ramp connects two stable operating points.",
            f"The signal has three distinct phases: constant ({level1}), linear transition, constant ({level2}). The transition starts at {t1_s}s and ends at {t2_s}s. No oscillation, just a smooth level change.",
            f"A controlled ramp between two plateaus. Starting level: {level1}. Ending level: {level2}. The ramp takes {ramp_dur}s. This is neither a step (instant) nor a trend (continuous) — it's a bounded transition.",
        ])
        return sig, sr, desc

    def gen_seasonal():
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        t = np.linspace(0, duration, n)
        n_cycles = np_rng.uniform(0.5, 2.5)
        f = n_cycles / duration
        amp = round(np_rng.uniform(1.0, 3.0), 1)
        baseline = round(np_rng.uniform(-1, 1), 1)
        sig = baseline + amp * np.sin(2 * np.pi * f * t) + np_rng.normal(0, amp * 0.15, n)
        period_s = round(1/f, 3) if f > 0 else 0
        desc = rng.choice([
            f"A seasonal pattern completing approximately {n_cycles:.1f} cycles with amplitude {amp} around baseline {baseline}. The slow oscillation with noise is characteristic of environmental or energy data.",
            f"Slow periodic variation with period ~{period_s}s. The signal rises and falls with amplitude {amp} around {baseline}. Only {n_cycles:.1f} full cycles are visible — this is a long-period oscillation.",
            f"Low-frequency oscillation ({n_cycles:.1f} cycles visible). Unlike a high-frequency sine wave, this pattern unfolds slowly over the entire recording. Amplitude {amp}, baseline {baseline}.",
            f"Seasonal or cyclical pattern. The signal completes about {n_cycles:.1f} slow oscillations. Each cycle takes approximately {period_s}s. The amplitude ({amp}) and noise make this look like weather or energy data.",
        ])
        return sig, sr, desc

    def gen_irregular_spikes():
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        sig = np_rng.normal(0, 0.1, n)
        n_spikes = np_rng.randint(3, 12)
        positions = sorted(np_rng.randint(5, n - 5, n_spikes))
        amps = np_rng.uniform(1.5, 5.0, n_spikes)
        for pos, amp in zip(positions, amps):
            sign = rng.choice([-1, 1])
            sig[pos] += sign * amp
            if pos + 1 < n:
                sig[pos + 1] += sign * amp * 0.3
        avg_amp = round(float(np.mean(np.abs(amps))), 1)
        times = [round(p / sr, 3) for p in positions]
        desc = rng.choice([
            f"An irregular spike train with {n_spikes} spikes of varying amplitude (average {avg_amp}) at irregular intervals. The background is low-amplitude noise.",
            f"Random-looking spikes at times {times}s. Unlike periodic impulses, these have no regular spacing. Each spike has different amplitude. Average spike height: {avg_amp}.",
            f"{n_spikes} sharp transient events at unpredictable times. The spacing between spikes is irregular — this is not periodic. Background is quiet noise.",
            f"Aperiodic spike activity. {n_spikes} events detected, irregularly spaced, with varying amplitudes (avg {avg_amp}). The signal between spikes is baseline noise.",
        ])
        return sig, sr, desc

    def gen_crescendo():
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        t = np.linspace(0, duration, n)
        max_freq = sr / 4
        f = round(min(np_rng.uniform(3, 15), max_freq), 0)
        sig = (t / duration) * np.sin(2 * np.pi * f * t)
        direction = rng.choice(["increasing", "decreasing"])
        if direction == "decreasing":
            sig = sig[::-1].copy()
        sig += np_rng.normal(0, 0.03, n)
        desc = rng.choice([
            f"A {f:.0f} Hz oscillation with {direction} amplitude envelope. The signal {'grows from silence to full amplitude' if direction == 'increasing' else 'fades from full amplitude to silence'}.",
            f"The oscillation frequency stays constant at {f:.0f} Hz, but the amplitude {'builds up gradually' if direction == 'increasing' else 'dies down gradually'}. This is a {'crescendo' if direction == 'increasing' else 'decrescendo'} pattern.",
            f"{'Growing' if direction == 'increasing' else 'Fading'} {f:.0f} Hz oscillation. The envelope is linear: amplitude {'starts at zero and reaches maximum' if direction == 'increasing' else 'starts at maximum and reaches zero'}.",
            f"Amplitude-modulated oscillation at {f:.0f} Hz with a linear {'rising' if direction == 'increasing' else 'falling'} envelope. The frequency doesn't change, only the intensity.",
        ])
        return sig, sr, desc

    def gen_trend_seasonal_noise():
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        t = np.linspace(0, duration, n)
        slope = round(np_rng.uniform(-2, 2), 1)
        trend = slope * t / duration
        trend_dir = "upward" if slope > 0 else "downward" if slope < 0 else "flat"
        n_cycles = round(np_rng.uniform(2, 6), 0)
        f_season = n_cycles / duration
        amp_season = round(np_rng.uniform(0.3, 1.5), 1)
        seasonal = amp_season * np.sin(2 * np.pi * f_season * t)
        noise_std = round(np_rng.uniform(0.1, 0.5), 2)
        noise = np_rng.normal(0, noise_std, n)
        sig = trend + seasonal + noise
        desc = rng.choice([
            f"A composite signal with three components: a {trend_dir} linear trend (slope {slope}), a seasonal oscillation with {int(n_cycles)} cycles (amplitude {amp_season}), and random noise (std {noise_std}).",
            f"Classical time series decomposition: trend ({trend_dir}, slope={slope}) + seasonality ({int(n_cycles)} cycles, amplitude={amp_season}) + noise (std={noise_std}). All three components are visible.",
            f"Three layers superimposed: (1) a {trend_dir} drift, (2) {int(n_cycles)} periodic oscillations with amplitude {amp_season}, (3) random noise with std {noise_std}. The trend dominates if slope is large, seasonality dominates if amplitude is large.",
            f"This signal contains trend, seasonality, and noise. The {trend_dir} trend (slope {slope}) shifts the overall level. The {int(n_cycles)}-cycle seasonal component (amplitude {amp_season}) adds periodic variation. Noise (std {noise_std}) adds randomness.",
        ])
        return sig, sr, desc

    def gen_frequency_shift():
        sr = _pick_sr()
        n = _n()
        duration = n / sr
        t = np.linspace(0, duration, n)
        max_freq = sr / 4
        f1 = round(min(np_rng.uniform(2, 8), max_freq), 0)
        f2 = round(min(np_rng.uniform(10, 30), max_freq), 0)
        split = np_rng.randint(n // 3, 2 * n // 3)
        t_split = round(split / sr, 3)
        pct = round(100 * split / n)
        sig = np.zeros(n)
        sig[:split] = np.sin(2 * np.pi * f1 * t[:split])
        sig[split:] = np.sin(2 * np.pi * f2 * t[split:])
        sig += np_rng.normal(0, 0.05, n)
        desc = rng.choice([
            f"A signal that abruptly changes frequency from {f1:.0f} Hz to {f2:.0f} Hz at t={t_split}s. The first segment oscillates slowly, the second faster.",
            f"Frequency transition at {pct}% of the signal. Before: {f1:.0f} Hz. After: {f2:.0f} Hz. The change is instantaneous — not a gradual chirp but a sudden switch.",
            f"Two different oscillation rates in one signal. The first part runs at {f1:.0f} Hz, then at t={t_split}s it jumps to {f2:.0f} Hz. The amplitude stays the same, only the frequency changes.",
            f"A non-stationary signal: the frequency is not constant. It holds at {f1:.0f} Hz, then shifts to {f2:.0f} Hz at the {pct}% mark. This represents a mode change or state transition.",
        ])
        return sig, sr, desc

    def gen_heartbeat_like():
        sr = rng.choice([100, 250, 500])
        n = _n()
        duration = n / sr
        sig = np_rng.normal(0, 0.03, n)
        hr = np_rng.randint(50, 120)
        period_samples = int(sr * 60 / hr)
        n_beats = 0
        for start in range(np_rng.randint(5, 20), n, period_samples):
            if start + 10 >= n:
                break
            if start + 3 < n:
                sig[start:start + 3] += 0.15
            qrs = start + 5
            if qrs + 4 < n:
                sig[qrs] -= 0.1
                sig[qrs + 1] += 1.0
                sig[qrs + 2] += 0.8
                sig[qrs + 3] -= 0.15
            t_wave = qrs + 8
            if t_wave + 4 < n:
                sig[t_wave:t_wave + 4] += 0.2
            n_beats += 1
        period_s = round(60 / hr, 2)
        desc = rng.choice([
            f"A synthetic heartbeat-like signal at approximately {hr} BPM ({n_beats} beats visible). Each beat has a small P wave, a sharp QRS complex (the tall peak), and a broad T wave.",
            f"Cardiac-like rhythm at {hr} beats per minute. The signal is mostly flat baseline with periodic sharp peaks (QRS complexes) every {period_s}s. {n_beats} complete cardiac cycles visible.",
            f"ECG-like pattern: repeating PQRST complexes at {hr} BPM. The dominant feature is the tall, narrow QRS spike recurring every {period_s}s. The P and T waves are smaller bumps before and after each spike.",
            f"Periodic sharp transients resembling heartbeats. Rate: {hr} BPM ({n_beats} beats in {duration:.1f}s). Each beat has a characteristic morphology: small bump (P), tall spike (QRS), broad bump (T).",
        ])
        return sig, sr, desc

    # --- Augmentation wrapper ---

    def _augment_and_describe(gen_fn):
        """Apply random augmentation to a base signal and update description."""
        sig, sr, desc = gen_fn()
        augmentations = []

        # Add noise at various SNR levels
        if np_rng.random() < 0.4:
            noise_std = round(np_rng.uniform(0.1, 0.8), 2)
            sig = sig + np_rng.normal(0, noise_std, len(sig))
            augmentations.append(f"additive noise (std={noise_std})")

        # Temporal shift (circular)
        if np_rng.random() < 0.3:
            shift = np_rng.randint(10, len(sig) - 10)
            sig = np.roll(sig, shift)
            shift_s = round(shift / sr, 3)
            augmentations.append(f"temporally shifted by {shift_s}s")

        # Amplitude scaling
        if np_rng.random() < 0.3:
            scale = round(np_rng.uniform(0.2, 5.0), 1)
            sig = sig * scale
            augmentations.append(f"amplitude scaled by {scale}x")

        # Baseline wander
        if np_rng.random() < 0.2:
            wander_freq = np_rng.uniform(0.1, 0.5)
            wander_amp = round(np_rng.uniform(0.2, 1.0), 1)
            t = np.linspace(0, len(sig) / sr, len(sig))
            sig = sig + wander_amp * np.sin(2 * np.pi * wander_freq * t)
            augmentations.append(f"baseline wander (amp={wander_amp})")

        if augmentations:
            desc += f" Additionally, the signal has been modified with: {', '.join(augmentations)}."

        return sig, sr, desc

    generators = [
        gen_sine, gen_dual_sine, gen_step, gen_ramp, gen_impulse,
        gen_periodic_impulse, gen_am, gen_chirp, gen_decay, gen_noise,
        gen_random_walk, gen_square, gen_sawtooth, gen_noisy_sine, gen_burst,
        gen_offset_sine, gen_varying_amplitude, gen_gravity_axis, gen_scaled_comparison,
        # New generators
        gen_trend_plus_oscillation, gen_multi_step, gen_exponential_growth,
        gen_plateau_ramp_plateau, gen_seasonal, gen_irregular_spikes,
        gen_crescendo, gen_trend_seasonal_noise, gen_frequency_shift,
        gen_heartbeat_like,
    ]

    # Generate samples — base signals + augmented variants
    # _n() is randomized before each call so generators produce variable lengths
    per_gen = max(1, max_samples // (len(generators) * 2))  # half base, half augmented
    for gen_fn in generators:
        # Base signals (no augmentation)
        for _ in range(per_gen):
            if len(samples) >= max_samples:
                break
            try:
                _set_length()
                signal, sr, description = gen_fn()
                signal = signal.astype(np.float32)
                description += f" Signal length: {len(signal)} samples."
            except Exception:
                continue

            samples.append({
                "time_series": [signal.tolist()],
                "time_series_text": ["Signal:"],
                "pre_prompt": f"Synthetic signal sampled at {sr} Hz.",
                "post_prompt": rng.choice(_CAPTION_QUESTIONS),
                "answer": description,
                "task": "captioning",
                "domain": "synthetic",
                "source": "align-synthetic",
            })

        # Augmented variants
        for _ in range(per_gen):
            if len(samples) >= max_samples:
                break
            try:
                _set_length()
                signal, sr, description = _augment_and_describe(gen_fn)
                signal = signal.astype(np.float32)
                description += f" Signal length: {len(signal)} samples."
            except Exception:
                continue

            samples.append({
                "time_series": [signal.tolist()],
                "time_series_text": ["Signal:"],
                "pre_prompt": f"Synthetic signal sampled at {sr} Hz.",
                "post_prompt": rng.choice(_CAPTION_QUESTIONS),
                "answer": description,
                "task": "captioning",
                "domain": "synthetic",
                "source": "align-synthetic-augmented",
            })

    rng.shuffle(samples)
    print(f"  Synthetic signals: {len(samples)} samples ({len(generators)} signal types, "
          f"~{sum(1 for s in samples if 'augmented' in s['source'])} augmented)")
    return samples


def build_alignment_dataset(
    max_m4: int = 10_000,
    max_hf_sensor: int = 20_000,
    max_opentslm: int = 10_000,
    max_synthetic: int = 10_000,
    max_ucr: int = 500,
    seed: int = 42,
) -> DatasetDict:
    """Build dataset for embedding alignment (stage0_align).

    Real signals with domain-expert descriptions + synthetic signals with
    precise feature descriptions. Teaches embeddings what signals look like.

    1. M4 Captions — real TS with LLM-generated descriptions
    2. ETT/Weather/Traffic/Electricity — real sensor data
    3. HAR/Sleep/ECG/Bearing/PAMAP2 — real signals with rationale descriptions
    4. Synthetic signals — sine, step, impulse, chirp, AM, etc. with precise captions
    5. UCR subset — 112 real-world classified signals

    Returns a DatasetDict with train/validation splits.
    """
    print("=" * 60)
    print("Building alignment dataset (stage0_align)")
    print("=" * 60)

    all_samples = []

    # 1. M4 captions — real TS with LLM-generated descriptions
    print("\n[1/5] M4 Captions (general forecasting):")
    all_samples.extend(load_m4_captions(max_m4, seed))

    # 2. HF sensor data — ETT, Weather, Traffic, Electricity
    print("\n[2/6] HF sensor data (electrical, weather, traffic, energy):")
    all_samples.extend(_load_hf_sensor_data(max_hf_sensor, seed))

    # 3. OpenTSLM signals — HAR, Sleep, ECG, Bearing (real signals, rationale descriptions)
    print("\n[3/6] OpenTSLM signals (HAR, Sleep, ECG, Bearing):")
    all_samples.extend(_load_opentslm_signals(max_opentslm, seed))

    # 4. Synthetic signals — sine, step, impulse, chirp, AM, etc.
    print("\n[4/6] Synthetic signals (signal feature vocabulary):")
    all_samples.extend(_build_synthetic_signals(max_synthetic, seed))

    # 5. UCR signals — real-world domain diversity
    print("\n[5/6] UCR signals (description tasks):")
    try:
        ucr_classification = load_ucr_classification(max_per_dataset=max_ucr, seed=seed)
        ucr_rng = random.Random(seed + 7)
        ucr_repurposed = 0
        for s in ucr_classification:
            ts_list = s.get("time_series", [])
            if not ts_list:
                continue
            signal = np.array(ts_list[0], dtype=np.float32)
            if len(signal) < 20 or signal.std() < 1e-10:
                continue
            task = ucr_rng.choice(_ALIGN_TASKS)
            qa = _compute_description(signal, task, s.get("domain", "time series"), ucr_rng)
            if qa is None:
                continue
            all_samples.append({
                "time_series": s["time_series"],
                "time_series_text": s["time_series_text"],
                "pre_prompt": s["pre_prompt"],
                "post_prompt": qa["question"],
                "answer": qa["answer"],
                "task": task,
                "domain": s.get("domain", "time series"),
                "source": s["source"],
            })
            ucr_repurposed += 1
        print(f"  UCR: {ucr_repurposed} samples repurposed as description tasks")
    except Exception as e:
        print(f"  UCR: SKIP ({e})")

    rng = random.Random(seed)
    rng.shuffle(all_samples)

    n = len(all_samples)
    val_n = max(200, int(0.05 * n))
    val = all_samples[:val_n]
    train = all_samples[val_n:]

    print(f"\n{'=' * 60}")
    print(f"Alignment dataset: {n} total (train={len(train)}, val={val_n})")
    tasks = Counter(s["task"] for s in train)
    sources = Counter(s["source"].split("/")[0] for s in train)
    print(f"Tasks: {dict(tasks)}")
    print(f"Sources: {dict(sources)}")
    print(f"{'=' * 60}")

    def to_hf_dict(samples):
        keys = ["time_series", "time_series_text", "pre_prompt", "post_prompt", "answer", "task", "domain", "source"]
        return {k: [s[k] for s in samples] for k in keys}

    return DatasetDict({
        "train": Dataset.from_dict(to_hf_dict(train)),
        "validation": Dataset.from_dict(to_hf_dict(val)),
    })
