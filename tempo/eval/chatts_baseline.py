"""ChatTS baseline evaluation on Time-MQA/TSQA.

ChatTS is a Time Series Multimodal LLM (8B/14B) that encodes time series
as patches concatenated with text tokens. This script evaluates it on
the same Time-MQA benchmark as TEMPO and GPT baselines.

Requires: pip install transformers torch numpy pandas

Usage:
    # Local (needs ~16GB VRAM for 8B, ~6GB for Int4)
    cd tempo && uv run python -m tempo.eval.chatts_baseline \
        --model-path ./ckpt/ChatTS-8B \
        --max-samples 200

    # Or specify HuggingFace model ID
    cd tempo && uv run python -m tempo.eval.chatts_baseline \
        --model-path NetManAIOps/ChatTS-8B \
        --max-samples 200
"""

from __future__ import annotations

import argparse
import json
import os
import re
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


# 1. Data loading (shared with gpt_baseline)

def load_time_mqa(max_per_task: int = 200, seed: int = 42,
                  data_dir: str | None = None) -> list[dict]:
    """Load Time-MQA/TSQA: classification, anomaly_detection, open_ended_qa."""
    import random
    import pandas as pd

    rng = random.Random(seed)
    samples = []

    task_files = {
        "classification": "classification.csv",
        "anomaly_detection": "anomaly_detection.csv",
        "open_ended_qa": "open_ended_QA.csv",
    }

    search_dirs = []
    if data_dir:
        search_dirs.append(Path(data_dir))
    search_dirs.extend([
        Path(os.environ.get("SM_CHANNEL_EVALDATA", "/opt/ml/input/data/evaldata")),
        Path("data/time_mqa"),
        Path(__file__).resolve().parent.parent.parent / "data" / "time_mqa",
    ])

    for task_name, csv_name in task_files.items():
        print(f"  Loading {task_name}...")
        try:
            local_path = None
            for d in search_dirs:
                candidate = d / csv_name
                if candidate.exists():
                    local_path = str(candidate)
                    break
            if local_path is None:
                from huggingface_hub import hf_hub_download
                hf_names = {"classification": "Classification/classification.csv",
                            "anomaly_detection": "Anomaly_Detection/anomaly_detection.csv",
                            "open_ended_qa": "Open_Ended_QA/open_ended_QA.csv"}
                local_path = hf_hub_download("Time-MQA/TSQA", hf_names[task_name], repo_type="dataset")

            df = pd.read_csv(local_path)
            n = min(max_per_task, len(df))
            indices = rng.sample(range(len(df)), n)

            for idx in indices:
                row = df.iloc[idx]
                qa_raw = row.get("QA_list", "")
                domain = row.get("application_domain", "general")

                q_match = re.search(r'"question"\s*:\s*"(.+?)"(?:\s*,\s*"answer")', qa_raw, re.DOTALL)
                a_match = re.search(r'"answer"\s*:\s*"(.+?)"', qa_raw, re.DOTALL)
                if not q_match or not a_match:
                    continue

                question = q_match.group(1).replace('\\"', '"')
                answer = a_match.group(1).replace('\\"', '"')

                # Extract time series values from question
                ts_match = re.search(r'\[([0-9eE+\-., \n]+)\]', question)
                ts_values = None
                if ts_match:
                    try:
                        ts_values = [float(x.strip()) for x in ts_match.group(1).split(",") if x.strip()]
                    except ValueError:
                        pass

                actual_task = task_name
                if task_name == "open_ended_qa":
                    sub = row.get("task_type", "")
                    if sub:
                        actual_task = sub

                samples.append({
                    "question": question,
                    "answer": answer,
                    "ts_values": ts_values,
                    "task": actual_task,
                    "domain": domain,
                })
            print(f"    {task_name}: {n} samples")
        except Exception as e:
            print(f"    {task_name}: SKIP ({e})")

    rng.shuffle(samples)
    print(f"  Total: {len(samples)} samples")
    task_counts = Counter(s["task"] for s in samples)
    for t, c in task_counts.most_common():
        print(f"    {t}: {c}")
    return samples


# 2. ChatTS inference

class ChatTSEvaluator:
    """Wraps ChatTS model for evaluation."""

    def __init__(self, model_path: str, device: str = "cuda", use_int4: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

        print(f"Loading ChatTS from {model_path}...")

        load_kwargs = {
            "trust_remote_code": True,
            "device_map": device,
        }
        if use_int4:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, tokenizer=self.tokenizer,
        )
        self.device = device
        print(f"  ChatTS loaded ({sum(p.numel() for p in self.model.parameters())/1e9:.1f}B params)")

    def answer(self, question: str, ts_values: list[float] | None = None,
               max_new_tokens: int = 300) -> str:
        """Run ChatTS inference on a single sample."""
        import torch

        # Build prompt — ChatTS uses <ts> placeholder for time series
        if ts_values and len(ts_values) > 0:
            # Replace the inline TS values with <ts> token and pass array separately
            prompt_text = re.sub(
                r'\[([0-9eE+\-., \n]+)\]',
                '<ts>',
                question,
                count=1,
            )
            ts_list = [np.array(ts_values, dtype=np.float64)]
        else:
            prompt_text = question
            ts_list = []

        # Format as chat
        messages = [{"role": "user", "content": prompt_text}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        try:
            if ts_list:
                inputs = self.processor(
                    text=prompt, timeseries=ts_list,
                    padding=True, return_tensors="pt",
                )
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt")

            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            # Decode only new tokens
            input_len = inputs["input_ids"].shape[1]
            response = self.tokenizer.decode(
                outputs[0][input_len:], skip_special_tokens=True,
            )
            return response.strip()

        except Exception as e:
            return f"[ERROR: {e}]"


# 3. Answer extraction and metrics

def extract_answer(text: str) -> str:
    """Extract answer from ChatTS output."""
    # "Answer: X" or "answer is X"
    m = re.search(r"(?:answer|Answer)\s*(?:is|:)\s*(.+?)(?:\.|$|\n)", text)
    if m:
        return m.group(1).strip().rstrip(".")

    m = re.search(r"[Cc]orrect answer:\s*(.+?)(?:\.|$|\n)", text)
    if m:
        return m.group(1).strip().rstrip(".")

    return re.split(r"[\.\n]", text.strip(), maxsplit=1)[0].strip()[:200]


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    for prefix in ["based on the given information, the activity is ",
                    "based on the given information, the answer is ",
                    "based on the given information, this time series includes ",
                    "based on the given information, "]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.strip().rstrip(".")


def compute_metrics(results: list[dict]) -> dict:
    from sklearn.metrics import f1_score

    task_results = defaultdict(list)
    for r in results:
        task_results[r["task"]].append(r)

    metrics = {}
    for task, items in sorted(task_results.items()):
        golds = [r["gold_normalized"] for r in items]
        preds = [r["pred_normalized"] for r in items]
        correct = [g == p for g, p in zip(golds, preds)]
        acc = sum(correct) / len(correct)

        try:
            f1_w = f1_score(golds, preds, average="weighted", zero_division=0)
            f1_m = f1_score(golds, preds, average="macro", zero_division=0)
        except Exception:
            f1_w = f1_m = 0.0

        metrics[task] = {
            "accuracy": round(acc, 4),
            "f1_weighted": round(f1_w, 4),
            "f1_macro": round(f1_m, 4),
            "n": len(items),
            "n_correct": sum(correct),
        }

    overall = sum(r["correct"] for r in results)
    metrics["overall"] = {
        "accuracy": round(overall / max(len(results), 1), 4),
        "n": len(results),
        "n_correct": overall,
    }
    return metrics


# 4. Main

def run_eval(
    model_path: str,
    max_samples: int = 200,
    max_new_tokens: int = 300,
    output_dir: str = "results/chatts_baseline",
    device: str = "cuda",
    use_int4: bool = False,
    data_dir: str | None = None,
    seed: int = 42,
):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(model_path).name or "ChatTS"
    run_dir = Path(output_dir) / f"{model_name}_{run_id}"

    print("=" * 60)
    print("ChatTS BASELINE EVALUATION — Time-MQA/TSQA")
    print(f"  Model:      {model_path}")
    print(f"  Device:     {device}")
    print(f"  Int4:       {use_int4}")
    print(f"  Samples:    {max_samples}")
    print(f"  Output:     {run_dir}")
    print("=" * 60)

    # Load model
    evaluator = ChatTSEvaluator(model_path, device=device, use_int4=use_int4)

    # Load data
    print("\nLoading Time-MQA/TSQA...")
    samples = load_time_mqa(max_per_task=max_samples, seed=seed, data_dir=data_dir)

    if not samples:
        print("ERROR: No samples loaded")
        return

    # Run inference
    print(f"\nRunning inference ({len(samples)} samples)...\n")
    results = []
    t0 = time.time()

    for i, sample in enumerate(samples):
        raw_output = evaluator.answer(
            sample["question"], sample["ts_values"],
            max_new_tokens=max_new_tokens,
        )

        pred = extract_answer(raw_output)
        gold = extract_answer(sample["answer"])
        pred_n = normalize_answer(pred)
        gold_n = normalize_answer(gold)
        correct = pred_n == gold_n

        results.append({
            "task": sample["task"],
            "domain": sample["domain"],
            "gold_raw": sample["answer"][:500],
            "gold_answer": gold,
            "gold_normalized": gold_n,
            "pred_answer": pred,
            "pred_normalized": pred_n,
            "raw_output": raw_output[:1000],
            "correct": correct,
        })

        if i < 20 or i % 50 == 0:
            tag = "OK" if correct else "WRONG"
            print(f"  [{i}/{len(samples)}] [{tag}] {sample['task']}: "
                  f"gold={gold_n[:50]}, pred={pred_n[:50]}")

    elapsed = time.time() - t0
    print(f"\nInference: {elapsed:.0f}s ({elapsed/max(len(results),1):.1f}s/sample)")

    # Compute metrics
    metrics = compute_metrics(results)

    # Print
    print(f"\n{'=' * 60}")
    print(f"RESULTS - ChatTS ({model_name})")
    print(f"{'=' * 60}")
    for task, m in sorted(metrics.items()):
        f1_str = f", F1w={m['f1_weighted']:.1%}" if "f1_weighted" in m else ""
        print(f"  {task:<25} Acc={m['accuracy']:.1%}{f1_str}  "
              f"({m['n_correct']}/{m['n']})")

    # Save
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": f"ChatTS ({model_name})",
            "model_path": model_path,
            "n_evaluated": len(results),
            "elapsed_sec": round(elapsed, 1),
            "metrics": metrics,
        }, f, indent=2, ensure_ascii=False)

    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    examples = defaultdict(lambda: {"correct": [], "wrong": []})
    for r in results:
        bucket = "correct" if r["correct"] else "wrong"
        if len(examples[r["task"]][bucket]) < 5:
            examples[r["task"]][bucket].append({
                "gold": r["gold_answer"],
                "pred": r["pred_answer"],
                "output": r["raw_output"][:500],
            })
    with open(run_dir / "examples.json", "w", encoding="utf-8") as f:
        json.dump(dict(examples), f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {run_dir}/")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="ChatTS baseline on Time-MQA")
    parser.add_argument("--model-path", default="NetManAIOps/ChatTS-8B",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--output-dir", default="results/chatts_baseline")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--int4", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--data-dir", default=None, help="Path to Time-MQA CSVs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_eval(
        model_path=args.model_path,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        device=args.device,
        use_int4=args.int4,
        data_dir=args.data_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
