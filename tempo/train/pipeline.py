"""Two-phase training pipeline for TEMPO.

Phase 0 — Embedding Alignment:
    Only TS token embeddings train (627 params × hidden_dim).
    DoRA adapters are frozen. High LR (1e-3). 2-3 epochs.
    Purpose: ground random TS embeddings in the LLM's representation space.

Phase 1 — Full Training:
    DoRA adapters + TS embeddings train together.
    Normal LR (2e-5). ~10 epochs.
    Purpose: learn time series reasoning across all tasks.

Usage:
    from tempo.train.pipeline import run_pipeline, PipelineConfig

    config = PipelineConfig(
        phase0_data="data/pretokenized/stage0_alignment",
        phase1_data="data/pretokenized/stage1_base",
    )
    results = run_pipeline(model, config)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

import os as _os
_trainer_version = _os.environ.get("TEMPO_TRAINER", "v2")
if _trainer_version == "v1":
    from .trainer import train
else:
    from .trainer_v2 import train
from .parquet_dataset import load_parquet_splits


@dataclass
class PipelineConfig:
    """Configuration for the 2-phase training pipeline."""

    # Data paths (env vars take precedence over defaults)
    phase0_data: str = ""
    phase1_data: str = ""

    # Output
    output_dir: str = "results"

    # Phase 0: Alignment
    phase0_epochs: int = 3
    phase0_lr: float = 1e-3
    phase0_patience: int = 3

    # Phase 1: Training
    phase1_epochs: int = 10
    phase1_lr: float = 2e-5
    phase1_embed_lr: float | None = None  # separate LR for TS embeddings
    phase1_patience: int = 3

    # Shared training params
    batch_size: int = 8
    grad_accum: int = 4
    num_workers: int = 4
    warmup_frac: float = 0.10

    # Logging
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    # Packing
    use_packing: bool = False
    pack_length: int = 2048

    # Resume
    checkpoint: str | None = None  # starting checkpoint
    skip_phase0: bool = False  # skip alignment (e.g., resuming from phase0 checkpoint)
    phase0_only: bool = False  # run only alignment, skip phase 1

    def __post_init__(self):
        # Env vars override defaults
        if not self.phase0_data:
            self.phase0_data = os.environ.get(
                "TEMPO_PHASE0_DATA", "data/pretokenized/stage0_alignment",
            )
        if not self.phase1_data:
            self.phase1_data = os.environ.get(
                "TEMPO_PHASE1_DATA", "data/pretokenized/stage1_base",
            )
        if not self.output_dir or self.output_dir == "results":
            self.output_dir = os.environ.get("TEMPO_OUTPUT_DIR", "results")


def run_pipeline(model, config: PipelineConfig) -> dict[str, Any]:
    """Run the 2-phase training pipeline.

    Args:
        model: TEMPO model instance (on CPU — Accelerate handles device placement)
        config: Pipeline configuration

    Returns:
        Dict with results from each phase.
    """
    results = {}
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load starting checkpoint if provided
    if config.checkpoint:
        print(f"Loading checkpoint: {config.checkpoint}")
        ckpt = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])

    eos = model.get_eos_token()

    if not config.skip_phase0:
        phase0_dir = str(output_dir / "phase0_alignment")
        phase0_best = Path(phase0_dir) / "best_model.pt"

        # Skip if already completed
        if phase0_best.exists():
            print(f"\nPhase 0 already completed — loading {phase0_best}")
            ckpt = torch.load(str(phase0_best), map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            results["phase0"] = {"skipped": True, "val_loss": ckpt.get("val_loss")}
        else:
            print("\n" + "=" * 60)
            print("PHASE 0: EMBEDDING ALIGNMENT")
            print(f"  Data: {config.phase0_data}")
            print(f"  LR: {config.phase0_lr}, Epochs: {config.phase0_epochs}")
            print(f"  Trainable: TS embeddings only (DoRA frozen)")
            print("=" * 60)

            # v2 trainer builds ChatML text itself — don't pretokenize here.
            # v1 trainer uses pretokenized input_ids — pass model to pretokenize.
            _pass_model = model if _trainer_version == "v1" else None
            dataset = load_parquet_splits(config.phase0_data, eos_token=eos, model=_pass_model)

            results["phase0"] = train(
                model,
                dataset,
                output_dir=phase0_dir,
                epochs=config.phase0_epochs,
                patience=config.phase0_patience,
                batch_size=config.batch_size,
                grad_accum=config.grad_accum,
                lr=config.phase0_lr,
                warmup_frac=config.warmup_frac,
                align_only=True,  # freezes DoRA, only TS embeddings train
                num_workers=config.num_workers,
                wandb_project=config.wandb_project,
                wandb_run_name=f"{config.wandb_run_name}_phase0" if config.wandb_run_name else None,
                use_packing=config.use_packing,
                pack_length=config.pack_length,
            )

            # Load best phase0 checkpoint for phase1
            best_path = results["phase0"].get("checkpoint_path", str(phase0_best))
            if os.path.exists(best_path):
                ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
                model.load_state_dict(ckpt["model_state"])

    if config.phase0_only:
        print("\nPhase 0 only — skipping phase 1.")
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE (phase 0 only)")
        for phase, res in results.items():
            if res.get("skipped"):
                print(f"  {phase}: skipped (val_loss={res.get('val_loss', '?')})")
            else:
                print(f"  {phase}: val_loss={res.get('best_val_loss', '?')}, "
                      f"epoch={res.get('best_epoch', '?')}")
        print(f"  Output: {output_dir}")
        print("=" * 60)
        return results

    phase1_dir = str(output_dir / "phase1_training")
    phase1_best = Path(phase1_dir) / "best_model.pt"

    if phase1_best.exists():
        print(f"\nPhase 1 already completed — loading {phase1_best}")
        ckpt = torch.load(str(phase1_best), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        results["phase1"] = {"skipped": True, "val_loss": ckpt.get("val_loss")}
    else:
        print("\n" + "=" * 60)
        print("PHASE 1: FULL TRAINING")
        print(f"  Data: {config.phase1_data}")
        print(f"  LR: {config.phase1_lr}, Epochs: {config.phase1_epochs}")
        print(f"  Trainable: DoRA adapters + TS embeddings")
        print("=" * 60)

        _pass_model = model if _trainer_version == "v1" else None
        dataset = load_parquet_splits(config.phase1_data, eos_token=eos, model=_pass_model)

        results["phase1"] = train(
            model,
            dataset,
            output_dir=phase1_dir,
            epochs=config.phase1_epochs,
            patience=config.phase1_patience,
            batch_size=config.batch_size,
            grad_accum=config.grad_accum,
            lr=config.phase1_lr,
            embed_lr=config.phase1_embed_lr,
            warmup_frac=config.warmup_frac,
            align_only=False,  # DoRA unfrozen
            num_workers=config.num_workers,
            wandb_project=config.wandb_project,
            wandb_run_name=f"{config.wandb_run_name}_phase1" if config.wandb_run_name else None,
            use_packing=config.use_packing,
            pack_length=config.pack_length,
        )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    for phase, res in results.items():
        if res.get("skipped"):
            print(f"  {phase}: skipped (val_loss={res.get('val_loss', '?')})")
        else:
            print(f"  {phase}: val_loss={res.get('best_val_loss', '?')}, "
                  f"epoch={res.get('best_epoch', '?')}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    return results
