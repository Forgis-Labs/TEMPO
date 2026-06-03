"""Structured evaluation for TEMPO — reproduces paper results.

Each supported dataset has a registered evaluator that:
  1. Loads the test split
  2. Generates predictions (streamed to predictions.jsonl)
  3. Scores with task-appropriate metrics
  4. Saves metrics.json + predictions.jsonl

This is NOT an example script — it's the official evaluation
pipeline. Results from this module are what goes in the paper.

Usage (from Python):
    from tempo.eval.evaluate import evaluate
    results = evaluate(model, dataset="har", output_dir="results/har_eval")

Usage (CLI):
    python -m tempo.eval.evaluate \\
        --checkpoint results/best_model.pt \\
        --dataset har \\
        --output_dir results/har_eval
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .scorer import extract_answer, score_one, score_predictions
from .sensitivity import sensitivity_test


# Result containers

@dataclass
class EvalResult:
    """Complete evaluation result — serializable to JSON."""
    dataset: str
    n_samples: int
    accuracy: float
    f1_weighted: float
    f1_macro: float
    sensitivity: dict[str, float]
    per_class: dict[str, Any]
    config: dict[str, Any]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")

    def __repr__(self) -> str:
        return (f"EvalResult(dataset={self.dataset}, n={self.n_samples}, "
                f"acc={self.accuracy:.1%}, f1w={self.f1_weighted:.3f}, "
                f"delta={self.sensitivity.get('delta', 0):.1%})")


# Dataset wrapper

class _EvalDataset(Dataset):
    def __init__(self, hf_dataset, eos: str):
        self.data = hf_dataset
        self.eos = eos

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return {
            "pre_prompt": row["pre_prompt"],
            "post_prompt": row["post_prompt"],
            "time_series": [torch.tensor(ch, dtype=torch.float32)
                            for ch in row.get("time_series", [])],
            "time_series_text": row.get("time_series_text", []),
            "answer": row.get("answer", "") + self.eos,
            "task": row.get("task", "classification"),
            "label": row.get("label", ""),
            "domain": row.get("domain", ""),
            "dataset": row.get("dataset", ""),
        }


# Core evaluation function

def evaluate(
    model,
    dataset,
    output_dir: str = "results/eval",
    dataset_name: str = "unknown",
    task: str = "classification",
    max_samples: int | None = None,
    max_new_tokens: int = 400,
    sensitivity_samples: int = 100,
    run_sensitivity: bool = True,
) -> EvalResult:
    """Evaluate a TEMPO model and save all outputs.

    This function is the single entry point for paper-reproducible
    evaluation. It always saves:
      - predictions.jsonl: one line per sample (raw + extracted)
      - metrics.json: aggregated scores

    Args:
        model: TEMPO model instance.
        dataset: HuggingFace Dataset (test split) or DatasetDict.
        output_dir: Where to save results.
        dataset_name: Name for the results file.
        task: Task type for answer extraction.
        max_samples: Cap on test samples.
        max_new_tokens: Max generation tokens.
        sensitivity_samples: Samples for the sensitivity test.
        run_sensitivity: Whether to run the sensitivity test.

    Returns:
        EvalResult with all metrics.
    """
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Handle DatasetDict vs Dataset vs PyTorch Dataset (from curriculum)
    if hasattr(dataset, "keys") and "test" in dataset:
        test_data = dataset["test"]
    else:
        test_data = dataset

    # PyTorch Datasets (e.g. OpenTSLMAdapter) don't have .select() —
    # use Subset for max_samples capping instead.
    if max_samples and max_samples < len(test_data):
        if hasattr(test_data, "select"):
            test_data = test_data.select(range(max_samples))
        else:
            from torch.utils.data import Subset
            test_data = Subset(test_data, range(max_samples))

    # If already a PyTorch Dataset with the right format, use directly.
    # Otherwise wrap HF dataset rows.
    if isinstance(test_data, Dataset):
        test_ds = test_data
    else:
        test_ds = _EvalDataset(test_data, model.get_eos_token())

    test_loader = DataLoader(test_ds, batch_size=1, collate_fn=lambda b: b, num_workers=0)

    # --- Sensitivity test ---
    sens = {"real_acc": 0, "noise_acc": 0, "delta": 0, "divergence": 0, "n_samples": 0}
    if run_sensitivity:
        print(f"Running sensitivity test ({sensitivity_samples} samples)...")
        sens = sensitivity_test(
            model, test_loader, task=task,
            max_samples=min(sensitivity_samples, len(test_ds)),
        )
        print(f"  Acc(real)={sens['real_acc']:.1%}  Delta={sens['delta']:.1%}")

    # --- Generate predictions (streamed to disk) ---
    pred_path = out / "predictions.jsonl"
    print(f"Generating {len(test_ds)} predictions -> {pred_path}")
    model.eval()
    golds, preds = [], []

    with torch.no_grad(), pred_path.open("w", encoding="utf-8") as fh:
        for i, batch in enumerate(test_loader):
            sample = batch[0]
            sample_task = sample.get("task", task)
            gold_raw = sample["answer"]
            output = model.generate([sample], max_new_tokens=max_new_tokens)
            pred_raw = output[0] if output else ""

            gold_ans = extract_answer(gold_raw, sample_task)
            pred_ans = extract_answer(pred_raw, sample_task)
            golds.append(gold_ans)
            preds.append(pred_ans)

            # Stream to disk
            fh.write(json.dumps({
                "idx": i,
                "task": sample_task,
                "label": sample.get("label", ""),
                "domain": sample.get("domain", ""),
                "gold_raw": gold_raw[-300:],
                "pred_raw": pred_raw[-300:],
                "gold_answer": gold_ans,
                "pred_answer": pred_ans,
                "correct": bool(gold_ans and pred_ans and
                                (gold_ans.lower() == pred_ans.lower() or
                                 gold_ans.lower() in pred_ans.lower())),
            }, ensure_ascii=False) + "\n")
            fh.flush()

            if (i + 1) % 50 == 0:
                acc_so_far = sum(1 for g, p in zip(golds, preds)
                                if g.lower() == p.lower()) / len(golds)
                print(f"  {i+1}/{len(test_ds)}  running_acc={acc_so_far:.1%}")

    # --- Compute metrics ---
    acc = accuracy_score(
        [g.lower() for g in golds],
        [p.lower() for p in preds],
    )
    f1w = f1_score(
        [g.lower() for g in golds],
        [p.lower() for p in preds],
        average="weighted", zero_division=0,
    )
    f1m = f1_score(
        [g.lower() for g in golds],
        [p.lower() for p in preds],
        average="macro", zero_division=0,
    )

    # Per-class report
    try:
        report = classification_report(
            [g.lower() for g in golds],
            [p.lower() for p in preds],
            zero_division=0, output_dict=True,
        )
    except Exception:
        report = {}

    result = EvalResult(
        dataset=dataset_name,
        n_samples=len(golds),
        accuracy=round(acc, 4),
        f1_weighted=round(f1w, 4),
        f1_macro=round(f1m, 4),
        sensitivity=sens,
        per_class=report,
        config={
            "max_new_tokens": max_new_tokens,
            "max_samples": max_samples,
            "task": task,
        },
    )
    result.save(out / "metrics.json")

    # --- Print summary ---
    print(f"\n{'=' * 50}")
    print(f"  {dataset_name} EVALUATION (n={len(golds)})")
    print(f"{'=' * 50}")
    print(f"  Accuracy:      {acc:.1%}")
    print(f"  F1 (weighted): {f1w:.3f}")
    print(f"  F1 (macro):    {f1m:.3f}")
    if run_sensitivity:
        print(f"  Sensitivity:   Delta={sens['delta']:.1%}")
    print(f"  Saved: {out / 'metrics.json'}")
    print(f"         {pred_path}")

    return result


# Curriculum dataset loading

# Maps friendly names to curriculum stages and their task types.
CURRICULUM_DATASETS = {
    "tsqa":      {"stage": "stage1_mcq",        "task": "classification"},
    "m4":        {"stage": "stage2_captioning",  "task": "captioning"},
    "har":       {"stage": "stage3_cot",         "task": "classification"},
    "har_cot":   {"stage": "stage3_cot",         "task": "classification"},
    "sleep":     {"stage": "stage4_sleep_cot",   "task": "classification"},
    "sleep_cot": {"stage": "stage4_sleep_cot",   "task": "classification"},
    "ecg":       {"stage": "stage5_ecg_cot",     "task": "classification"},
    "ecg_cot":   {"stage": "stage5_ecg_cot",     "task": "classification"},
}


def load_curriculum_dataset(name: str, split: str, eos_token: str) -> Dataset:
    """Load a curriculum dataset by friendly name.

    Uses the same OpenTSLM dataset classes and adapter as training,
    so evaluation data format matches training exactly.
    """
    info = CURRICULUM_DATASETS.get(name)
    if info is None:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Choose from: {list(CURRICULUM_DATASETS.keys())}, "
            f"or use --dataset_path / --dataset_hub for custom datasets."
        )

    from tempo.train.curriculum import _get_dataset_class, OpenTSLMAdapter
    cls = _get_dataset_class(info["stage"])
    return OpenTSLMAdapter(cls(split, EOS_TOKEN=eos_token), eos_token)


# CLI entry point

def main():
    """CLI for evaluation: python -m tempo.eval.evaluate ...

    Examples:
        # Evaluate HAR with FSQ transformer tokenizer:
        python -m tempo.eval.evaluate \\
            --checkpoint results/fsq_transformer_curriculum/stage3_cot/best_model.pt \\
            --tokenizer fsq_transformer \\
            --fsq_ckpt tempo/tokenizer/fsq_transformer_625_no_temp_best.pt \\
            --dataset har

        # Evaluate sleep staging:
        python -m tempo.eval.evaluate \\
            --checkpoint results/.../best_model.pt \\
            --tokenizer fsq_transformer --fsq_ckpt ... \\
            --dataset sleep

        # Evaluate with a custom HF dataset:
        python -m tempo.eval.evaluate \\
            --checkpoint results/.../best_model.pt \\
            --tokenizer fsq_transformer --fsq_ckpt ... \\
            --dataset_path /path/to/hf_dataset
    """
    import argparse
    _tempo_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(_tempo_root))
    # Also add OpenTSLM src/ for opentslm datasets (HAR, Sleep, etc.)
    _project_root = _tempo_root.parent
    for _candidate in [_project_root / "src", _tempo_root.parent / "src"]:
        if _candidate.exists():
            sys.path.insert(0, str(_candidate))
            break
    from tempo import TEMPO

    ap = argparse.ArgumentParser(
        description="Evaluate a TEMPO model on curriculum or custom datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    ap.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--llm_id", default="Qwen/Qwen3-4B")
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--use_dora", action="store_true")

    # Tokenizer (support all types)
    ap.add_argument("--tokenizer", default="fsq_transformer",
                    choices=["totem", "fsq", "fsq_transformer", "fsq_transformer_rope"],
                    help="Time series tokenizer type (default: fsq_transformer)")
    ap.add_argument("--totem_ckpt", default=None, help="TOTEM checkpoint path")
    ap.add_argument("--fsq_ckpt", default=None, help="FSQ/FSQ-Transformer checkpoint path")

    # Dataset — curriculum name OR custom path
    ap.add_argument("--dataset", default=None,
                    help=f"Curriculum dataset name: {list(CURRICULUM_DATASETS.keys())}")
    ap.add_argument("--dataset_path", default=None, help="Local HF dataset path")
    ap.add_argument("--dataset_hub", default=None, help="HF Hub dataset ID")
    ap.add_argument("--split", default="test", help="Dataset split (default: test)")

    # Eval config
    ap.add_argument("--task", default=None,
                    help="Task type for scoring (auto-detected for curriculum datasets)")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--output_dir", default="results/eval")
    ap.add_argument("--no_sensitivity", action="store_true")
    ap.add_argument("--sensitivity_samples", type=int, default=100)
    args = ap.parse_args()

    # Load model
    model = TEMPO.from_pretrained(
        args.checkpoint,
        llm_id=args.llm_id,
        tokenizer_type=args.tokenizer,
        totem_ckpt=args.totem_ckpt,
        fsq_ckpt=args.fsq_ckpt,
        lora_r=args.lora_r,
        use_dora=args.use_dora,
    )

    # Load dataset
    eos = model.get_eos_token()
    dataset_name = "unknown"

    if args.dataset:
        # Curriculum dataset
        dataset_name = args.dataset
        task = args.task or CURRICULUM_DATASETS[args.dataset]["task"]
        test_ds = load_curriculum_dataset(args.dataset, args.split, eos)
        # evaluate() expects something with __len__ and __getitem__ —
        # OpenTSLMAdapter already provides this. Wrap in a minimal proxy
        # so evaluate()'s DatasetDict detection doesn't trigger.
        dataset = test_ds
    elif args.dataset_path:
        from datasets import load_from_disk
        dataset = load_from_disk(args.dataset_path)
        task = args.task or "classification"
    elif args.dataset_hub:
        from datasets import load_dataset
        dataset = load_dataset(args.dataset_hub)
        task = args.task or "classification"
    else:
        ap.error("Specify --dataset, --dataset_path, or --dataset_hub")

    # Set output dir to include dataset name
    output_dir = args.output_dir
    if args.dataset and output_dir == "results/eval":
        output_dir = f"results/eval/{dataset_name}"

    result = evaluate(
        model, dataset,
        output_dir=output_dir,
        dataset_name=dataset_name,
        task=task,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        sensitivity_samples=args.sensitivity_samples,
        run_sensitivity=not args.no_sensitivity,
    )
    print(f"\n{result}")


if __name__ == "__main__":
    main()
