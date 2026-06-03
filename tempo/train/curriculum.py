"""Curriculum training for TEMPO.

Multi-stage training through OpenTSLM datasets with increasing complexity:
  stage1_mcq -> stage2_captioning -> stage3_cot -> stage4_sleep_cot -> stage5_ecg_cot

Each stage:
  1. Trains on the stage's dataset
  2. Evaluates on the test split (saves predictions.jsonl + metrics.json)
  3. Loads the best checkpoint and proceeds to the next stage

Usage (programmatic):
    from tempo import TEMPO, TEMPOConfig
    from tempo.train import train_curriculum

    model = TEMPO(config)
    result = train_curriculum(model, stages=["stage1_mcq", "stage2_captioning"], ...)

Usage (CLI):
    python -m tempo.train --strategy curriculum \\
        --stages stage1_mcq stage2_captioning stage3_cot
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

# Use trainer_v2 (SFTTrainer + DDP) by default — 5-10x faster than v1 (Accelerate + FSDP).
# Set SHRIKE_TRAINER=v1 to use the old trainer if needed.
_trainer_version = os.environ.get("SHRIKE_TRAINER", "v2")
if _trainer_version == "v1":
    from .trainer import train
else:
    from .trainer_v2 import train

# Stage configuration: token budget and default epochs per stage.
# Tokenizer compresses 4:1, so max_ts_tokens x 4 = max raw signal length.
STAGE_CONFIG = {
    "stage1_mcq":        {"max_ts_tokens": 64,  "epochs": 10, "task": "mcq",            "eval": True},
    "stage2_captioning": {"max_ts_tokens": 64,  "epochs": 10, "task": "free_text",       "eval": False},
    "stage3_cot":        {"max_ts_tokens": 64,  "epochs": 10, "task": "classification",  "eval": True},
    "stage4_sleep_cot":  {"max_ts_tokens": 375, "epochs": 10, "task": "classification",  "eval": True},
    "stage5_ecg_cot":    {"max_ts_tokens": 750, "epochs": 10, "task": "classification",  "eval": False},
}

ALL_STAGES = list(STAGE_CONFIG.keys())


def _get_dataset_class(stage: str):
    """Lazy import of OpenTSLM dataset classes."""
    from opentslm.time_series_datasets.TSQADataset import TSQADataset
    from opentslm.time_series_datasets.m4.M4QADataset import M4QADataset
    from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
    from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
    from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset

    mapping = {
        "stage1_mcq": TSQADataset,
        "stage2_captioning": M4QADataset,
        "stage3_cot": HARCoTQADataset,
        "stage4_sleep_cot": SleepEDFCoTQADataset,
        "stage5_ecg_cot": ECGQACoTQADataset,
    }
    if stage not in mapping:
        raise ValueError(f"Unknown stage: {stage}. Choose from {list(mapping.keys())}")
    return mapping[stage]


class OpenTSLMAdapter(Dataset):
    """Adapts an OpenTSLM dataset to the format tempo.train expects.

    OpenTSLM datasets return samples with:
        time_series: list[Tensor] or Tensor
        pre_prompt, post_prompt, answer: str
        time_series_text: list[str] (optional)

    This adapter normalizes to:
        time_series: list[Tensor]
        time_series_text: list[str]
        answer: str (with EOS appended)
    """

    def __init__(self, opentslm_ds, eos_token: str):
        self.ds = opentslm_ds
        self.eos = eos_token

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        s = self.ds[idx]

        ts = s.get("time_series", [])
        if isinstance(ts, torch.Tensor):
            ts = [ts]
        elif isinstance(ts, list):
            ts = [t if isinstance(t, torch.Tensor)
                  else torch.tensor(t, dtype=torch.float32) for t in ts]

        ts_text = s.get("time_series_text", [])
        if not ts_text:
            ts_text = ["Signal:"] * len(ts) if ts else []
        elif isinstance(ts_text, str):
            ts_text = [ts_text]

        answer = s.get("answer", "")
        if not answer.endswith(self.eos):
            answer += self.eos

        return {
            "pre_prompt": s.get("pre_prompt", ""),
            "post_prompt": s.get("post_prompt", ""),
            "time_series": ts,
            "time_series_text": ts_text,
            "answer": answer,
        }


class _DatasetProxy:
    """Mimics an HF DatasetDict for the trainer (dict-like with train/validation/test keys)."""

    def __init__(self, train_ds, val_ds, test_ds=None):
        self._splits = {"train": train_ds, "validation": val_ds}
        if test_ds is not None:
            self._splits["test"] = test_ds

    def __getitem__(self, key):
        return self._splits[key]

    def keys(self):
        return self._splits.keys()


def _build_stage_data(stage: str, eos_token: str) -> _DatasetProxy:
    """Instantiate train + val + test datasets for a curriculum stage."""
    cls = _get_dataset_class(stage)
    train_ds = OpenTSLMAdapter(cls("train", EOS_TOKEN=eos_token), eos_token)
    val_ds = OpenTSLMAdapter(cls("validation", EOS_TOKEN=eos_token), eos_token)
    test_ds = OpenTSLMAdapter(cls("test", EOS_TOKEN=eos_token), eos_token)
    return _DatasetProxy(train_ds, val_ds, test_ds)


def train_curriculum(
    model,
    stages: list[str] | None = None,
    checkpoint: str | None = None,
    output_dir: str = "results",
    epochs: int | None = None,
    patience: int = 3,
    batch_size: int = 4,
    grad_accum: int = 8,
    lr: float = 2e-5,
    pretokenize: bool = False,
    max_eval_samples: int = 0,
    use_packing: bool = True,
    pack_length: int = 2048,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
) -> dict[str, Any]:
    """Train TEMPO through a multi-stage curriculum.

    Starts from stage1_mcq (no alignment stage). After each stage,
    evaluates on the test split and saves predictions.jsonl + metrics.json.

    Args:
        model: TEMPO model instance (already on device).
        stages: List of stages to run. Default: all stages (stage1-5).
        checkpoint: Starting checkpoint path. Loaded before first stage.
        output_dir: Root directory for stage outputs (each stage gets a subdir).
        epochs: Override epochs for all stages. None uses per-stage defaults.
        patience: Early stopping patience.
        batch_size: Per-device batch size.
        grad_accum: Gradient accumulation steps.
        lr: Learning rate.
        pretokenize: If True, run tokenizer once over each dataset before training.
        max_eval_samples: Cap on test samples for evaluation (0 = all).
        wandb_project: WandB project name.
        wandb_run_name: WandB run name prefix.

    Returns:
        Dict with training + evaluation results from all completed stages.
    """
    from tempo.eval.evaluate import evaluate
    import torch.distributed as dist

    if stages is None:
        stages = ALL_STAGES

    # Detect distributed setup
    use_ddp = dist.is_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = local_rank == 0

    def _barrier():
        if use_ddp:
            dist.barrier()

    # v2 trainer doesn't need Accelerator — SFTTrainer handles DDP internally.
    # Only create Accelerator if using the v1 trainer.
    accelerator = None
    if _trainer_version == "v1":
        from accelerate import Accelerator
        accelerator = Accelerator(
            gradient_accumulation_steps=grad_accum,
            mixed_precision="bf16",
        )
        is_main = accelerator.is_main_process

    # Load starting checkpoint (all ranks load to stay in sync)
    if checkpoint is not None:
        if is_main:
            print(f"Loading checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if is_main:
            print(f"  Loaded epoch {ckpt.get('epoch', '?')}, "
                  f"val_loss={ckpt.get('val_loss', '?')}")
    _barrier()

    eos = model.get_eos_token()
    all_results = {}

    for stage in stages:
        stage_cfg = STAGE_CONFIG[stage]
        max_ts = stage_cfg["max_ts_tokens"]
        task_type = stage_cfg.get("task", "classification")
        n_epochs = epochs if epochs is not None else stage_cfg["epochs"]
        stage_dir = os.path.join(output_dir, stage)

        # Update model's token budget for this stage
        model.config.max_ts_tokens = max_ts

        if is_main:
            print(f"\n{'='*70}")
            print(f"STAGE: {stage}")
            print(f"  max_ts_tokens = {max_ts} ({max_ts * model.config.compression_factor} raw pts)")
            print(f"  epochs = {n_epochs}, batch = {batch_size}, grad_accum = {grad_accum}")
            print(f"  output = {stage_dir}")
            print(f"{'='*70}")

        # Skip if already completed (all ranks check + load)
        best_ckpt = os.path.join(stage_dir, "best_model.pt")
        if os.path.exists(best_ckpt):
            if is_main:
                print(f"  Found existing checkpoint -- loading and skipping to next stage")
            ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            if is_main:
                print(f"  Loaded epoch {ckpt.get('epoch', '?')}, "
                      f"val_loss={ckpt.get('val_loss', '?'):.4f}")
            _barrier()
            all_results[stage] = {"skipped": True, "val_loss": ckpt.get("val_loss")}
            continue

        # Build datasets (train + val + test)
        if is_main:
            print(f"  Loading {stage} datasets...")
        dataset = _build_stage_data(stage, eos)
        if is_main:
            print(f"  Train: {len(dataset['train'])}, "
                  f"Val: {len(dataset['validation'])}, "
                  f"Test: {len(dataset['test'])}")

        # Pre-tokenize if requested
        if pretokenize:
            from tempo.data.pretokenize import pretokenize_pytorch_dataset, PretokenizedDataset
            print(f"  Pre-tokenizing (one-time cost)...")
            train_samples = pretokenize_pytorch_dataset(
                dataset["train"], model, max_ts_tokens=max_ts)
            val_samples = pretokenize_pytorch_dataset(
                dataset["validation"], model, max_ts_tokens=max_ts)
            dataset = _DatasetProxy(
                PretokenizedDataset(train_samples),
                PretokenizedDataset(val_samples),
                dataset["test"],  # keep test raw for evaluation
            )

        # Pre-flight: check sequence length fits
        if is_main:
            sample = dataset["train"][0]
            sample_text = model.build_text(sample) + " " + sample["answer"]
            n_tokens = len(model.tokenizer(sample_text, add_special_tokens=False).input_ids)
            print(f"  Sample sequence length: {n_tokens} tokens "
                  f"(max_length={model.config.max_length})")
            if n_tokens > model.config.max_length:
                print(f"  WARNING: sequence ({n_tokens}) > max_length "
                      f"({model.config.max_length}), answers may be truncated!")
        sys.stdout.flush()

        # Stage-specific overrides
        stage_lr = stage_cfg.get("lr", lr)

        # Train
        run_name = f"{wandb_run_name or 'curriculum'}_{stage}" if wandb_project else None
        train_kwargs = dict(
            output_dir=stage_dir,
            epochs=n_epochs,
            patience=patience,
            batch_size=batch_size,
            grad_accum=grad_accum,
            lr=stage_lr,
            use_packing=use_packing,
            pack_length=pack_length,
            wandb_project=wandb_project,
            wandb_run_name=run_name,
        )
        if _trainer_version == "v1" and accelerator is not None:
            train_kwargs["accelerator"] = accelerator
        result = train(model, dataset, **train_kwargs)

        if is_main:
            print(f"\n  {stage} training complete: val_loss={result['best_val_loss']:.4f}, "
                  f"epoch={result['best_epoch']}")
        sys.stdout.flush()

        # Load best checkpoint for next stage (all ranks load to stay in sync)
        best_path = result.get("checkpoint_path", best_ckpt)
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state"])
        _barrier()

        if stage_cfg.get("eval", False) and is_main:
            eval_dir = os.path.join(stage_dir, "eval")
            print(f"\n  Evaluating {stage} on test split...")
            sys.stdout.flush()

            eval_result = evaluate(
                model,
                dataset["test"],
                output_dir=eval_dir,
                dataset_name=stage,
                task=task_type,
                max_samples=max_eval_samples if max_eval_samples > 0 else None,
                run_sensitivity=False,
            )

            print(f"  {stage} test eval: accuracy={eval_result.accuracy:.1%}, "
                  f"f1={eval_result.f1_weighted:.3f}")
            print(f"  Results saved to {eval_dir}/")
            sys.stdout.flush()

            from dataclasses import asdict
            result["eval"] = {
                "metrics": asdict(eval_result),
                "eval_dir": eval_dir,
            }
        elif is_main:
            print(f"\n  Skipping test evaluation for {stage}")
        _barrier()

        all_results[stage] = result

    if is_main:
        print(f"\n{'='*70}")
        print("CURRICULUM COMPLETE")
        for stage, res in all_results.items():
            if res.get("skipped"):
                print(f"  {stage}: skipped (checkpoint existed)")
            else:
                val = res.get("best_val_loss", "?")
                acc = res.get("eval", {}).get("metrics", {}).get("overall_accuracy", "?")
                print(f"  {stage}: val_loss={val:.4f}, test_accuracy={acc}")
        print(f"  Results in: {output_dir}")
        print(f"{'='*70}")

    return all_results
