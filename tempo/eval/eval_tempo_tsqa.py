"""Evaluate TEMPO on the same TSQA/EngineMT-QA benchmarks as the GPT baseline.

Produces directly comparable results: same data, same metrics, same output format.

Usage:
    cd tempo && uv run python -m tempo.eval.eval_tempo_tsqa \
        --checkpoint checkpoints/phase1_epoch4.pt \
        --fsq-ckpt fsq_transformer_rope_625_best.pt \
        --max-samples 200
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
import torch

# Load .env
_env = Path(__file__).resolve().parent.parent.parent / ".env"
if _env.exists():
    for line in _env.read_text().strip().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# Reuse loaders and metrics from gpt_baseline
from .gpt_baseline import (
    load_tsqa_test,
    load_engine_qa_test,
    extract_gpt_answer,
    compute_metrics,
)


def run_eval(
    checkpoint: str,
    fsq_ckpt: str,
    llm_id: str = "Qwen/Qwen3-1.7B",
    dataset: str = "all",
    max_samples: int = 200,
    max_new_tokens: int = 200,
    output_dir: str = "results/tempo_tsqa",
    device: str = "cuda",
    seed: int = 42,
):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_name = Path(checkpoint).stem
    run_dir = Path(output_dir) / f"tempo_{ckpt_name}_{run_id}"

    print("=" * 60)
    print("SHRIKE EVALUATION (TSQA/EngineMT-QA)")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Tokenizer:  {fsq_ckpt}")
    print(f"  LLM:        {llm_id}")
    print(f"  Dataset:    {dataset}")
    print(f"  Samples:    {max_samples}")
    print(f"  Device:     {device}")
    print(f"  Output:     {run_dir}")
    print("=" * 60)

    # Load model
    from tempo import TEMPO
    model = TEMPO.from_pretrained(
        checkpoint,
        llm_id=llm_id,
        tokenizer_type="fsq_transformer_rope",
        fsq_ckpt=fsq_ckpt,
        use_dora=True,
        device=device,
    )

    # Load data (same as GPT baseline)
    all_samples = []
    if dataset in ("tsqa", "all"):
        all_samples.extend(load_tsqa_test(max_samples, seed))
    if dataset in ("engine", "all"):
        all_samples.extend(load_engine_qa_test(max_samples, seed))

    if not all_samples:
        print("ERROR: No samples loaded")
        return

    print(f"\nTotal: {len(all_samples)} samples")

    # Run inference
    results = []
    t0 = time.time()

    for i, sample in enumerate(all_samples):
        ts = sample.get("time_series", [])
        question = sample.get("post_prompt", "")
        context = sample.get("pre_prompt", "")

        # Build signal tensor — Time-MQA embeds TS in the question text
        signal = None
        if ts and isinstance(ts[0], list) and len(ts[0]) > 0:
            signal = torch.tensor(ts[0], dtype=torch.float32)
        elif not ts or (isinstance(ts, list) and len(ts) == 0):
            # Try to extract from pre_prompt (Time-MQA format)
            import re
            ts_match = re.search(r'\[([0-9eE+\-., \n]+)\]', context)
            if ts_match:
                try:
                    vals = [float(x.strip()) for x in ts_match.group(1).split(",") if x.strip()]
                    if len(vals) > 4:
                        signal = torch.tensor(vals, dtype=torch.float32)
                except ValueError:
                    pass

        try:
            if signal is not None and len(signal) > 8:
                raw_output = model.analyze(
                    signal,
                    question=question + " /no_think",
                    signal_label=sample.get("time_series_text", ["Signal:"])[0],
                    context=context,
                    max_new_tokens=max_new_tokens,
                )
            else:
                # Text-only sample — use generate directly
                gen_sample = {
                    "pre_prompt": context,
                    "post_prompt": question + " /no_think",
                    "time_series": [],
                    "time_series_text": [],
                    "answer": "",
                }
                raw_output = model.generate([gen_sample], max_new_tokens=max_new_tokens)[0]
        except Exception as e:
            print(f"  [{i}] Error: {e}")
            raw_output = ""

        # Strip thinking tags
        import re
        raw_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
        raw_output = re.sub(r"<think>.*$", "", raw_output, flags=re.DOTALL).strip()
        raw_output = raw_output.replace("</think>", "").strip()

        pred = extract_gpt_answer(raw_output, sample["task"])
        gold = sample["answer"]
        gold_extracted = extract_gpt_answer(gold, sample["task"])

        correct = pred.lower().strip() == gold_extracted.lower().strip()

        results.append({
            "task": sample["task"],
            "domain": sample.get("domain", ""),
            "source": sample.get("source", ""),
            "gold_raw": gold[:500],
            "gold_answer": gold_extracted,
            "pred_answer": pred,
            "raw_output": raw_output[:1000],
            "correct": correct,
        })

        if i < 10 or i % 50 == 0:
            tag = "OK" if correct else "WRONG"
            print(f"  [{i}/{len(all_samples)}] [{tag}] {sample['task']}: "
                  f"gold={gold_extracted[:50]}, pred={pred[:50]}")

    elapsed = time.time() - t0
    print(f"\nInference: {elapsed:.0f}s ({elapsed/max(len(results),1):.1f}s/sample)")

    # Compute metrics
    metrics = compute_metrics(results)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"RESULTS - TEMPO ({ckpt_name})")
    print(f"{'=' * 60}")
    for task, m in sorted(metrics.items()):
        if "accuracy" in m:
            f1_str = f", F1w={m['f1_weighted']:.1%}" if "f1_weighted" in m else ""
            print(f"  {task:<25} Acc={m['accuracy']:.1%}{f1_str}  (n={m['n']})")
        elif "reasoning_rate" in m:
            print(f"  {task:<25} AvgLen={m['avg_answer_length']:.0f}, "
                  f"Reasoning={m['reasoning_rate']:.0%}  (n={m['n']})")

    # Save
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": "tempo",
            "checkpoint": checkpoint,
            "dataset": dataset,
            "max_samples": max_samples,
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
                "output_snippet": r["raw_output"][:500],
            })

    with open(run_dir / "examples.json", "w", encoding="utf-8") as f:
        json.dump(dict(examples), f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {run_dir}/")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate TEMPO on TSQA/EngineMT-QA")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fsq-ckpt", required=True)
    parser.add_argument("--llm-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset", default="all", choices=["tsqa", "engine", "all"])
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--output-dir", default="results/tempo_tsqa")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_eval(
        checkpoint=args.checkpoint,
        fsq_ckpt=args.fsq_ckpt,
        llm_id=args.llm_id,
        dataset=args.dataset,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
