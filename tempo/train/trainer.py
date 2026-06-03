"""Single training loop for TEMPO — with optional multi-GPU via Accelerate.

Usage:
    # Single GPU:
    from tempo.train import train
    result = train(model, dataset, output_dir="results/stage1")

    # Multi-GPU (launch with accelerate):
    accelerate launch -m tempo.train --strategy curriculum ...
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


class _HFDatasetWrapper(Dataset):
    """Thin wrapper: HuggingFace dataset -> PyTorch Dataset."""

    def __init__(self, hf_dataset, eos_token: str) -> None:
        self.data = hf_dataset
        self.eos = eos_token

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data[idx]
        return {
            "pre_prompt": row["pre_prompt"],
            "post_prompt": row["post_prompt"],
            "time_series": [torch.tensor(ch, dtype=torch.float32)
                            for ch in row.get("time_series", [])],
            "time_series_text": row.get("time_series_text", []),
            "answer": row.get("answer", "") + self.eos,
        }


def train(
    model,
    dataset,
    output_dir: str = "results",
    epochs: int = 5,
    patience: int = 2,
    batch_size: int = 4,
    grad_accum: int = 8,
    lr: float = 2e-5,
    embed_lr: float | None = None,
    warmup_frac: float = 0.10,
    augment: bool = False,
    align_only: bool = False,
    num_workers: int = 0,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    use_packing: bool = False,
    pack_length: int = 2048,
    accelerator=None,
) -> dict[str, Any]:
    """Train a TEMPO model. Auto-detects multi-GPU via Accelerate if available.

    Args:
        model: TEMPO model instance.
        dataset: DatasetDict with 'train' and optionally 'validation' splits.
        output_dir: Where to save checkpoints.
        epochs: Maximum training epochs.
        patience: Early stopping patience.
        batch_size: Per-device batch size.
        grad_accum: Gradient accumulation steps.
        lr: Learning rate for DoRA/LoRA parameters.
        embed_lr: Separate learning rate for TS embeddings (default: same as lr).
        warmup_frac: Fraction of total steps for warmup.
        augment: Apply time series augmentation during training.
        align_only: If True, freeze DoRA adapters (only TS embeddings train).
        accelerator: Pre-existing Accelerator instance (reused across curriculum
            stages). If None, a new one is created.

    Returns:
        Dict with best_val_loss, best_epoch, checkpoint_path.
    """
    from accelerate import Accelerator

    if accelerator is None:
        accelerator = Accelerator(
            gradient_accumulation_steps=grad_accum,
            mixed_precision="bf16",
        )
    is_main = accelerator.is_main_process
    device = accelerator.device

    n_gpus = accelerator.num_processes
    if is_main:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        print(f"  Accelerate: {n_gpus} GPU(s), mixed_precision={accelerator.mixed_precision}")
        if use_packing:
            effective_batch = batch_size * 5 * n_gpus
            print(f"  Packing enabled: pack_length={pack_length}, "
                  f"fetch_batch={batch_size * 5}/gpu x {n_gpus} gpus = "
                  f"{effective_batch} samples/step")

    # WandB logging
    use_wandb = wandb_project is not None and is_main
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            name=wandb_run_name or Path(output_dir).name,
            config={
                "lr": lr, "embed_lr": embed_lr, "batch_size": batch_size,
                "grad_accum": grad_accum, "epochs": epochs, "patience": patience,
                "align_only": align_only, "augment": augment,
                "output_dir": output_dir,
            },
        )

    eos = model.get_eos_token()
    raw_train = dataset["train"]
    train_ds = raw_train if isinstance(raw_train, Dataset) else _HFDatasetWrapper(raw_train, eos)

    has_val = "validation" in dataset if hasattr(dataset, '__contains__') else True
    if has_val:
        try:
            raw_val = dataset["validation"]
            val_ds = raw_val if isinstance(raw_val, Dataset) else _HFDatasetWrapper(raw_val, eos)
        except (KeyError, IndexError):
            has_val = False

    # Detect pre-tokenized dataset (fast path)
    from .parquet_dataset import ParquetMapDataset, LengthGroupedSampler
    use_pretokenized = (isinstance(train_ds, ParquetMapDataset)
                        and train_ds._pretokenized)
    if is_main and use_pretokenized:
        print("  Using pre-tokenized fast path (no per-batch tokenization)")

    # Pre-tokenized data is already in RAM — workers just add IPC overhead
    effective_workers = 0 if use_pretokenized else num_workers
    pin = effective_workers > 0
    dl_kwargs = dict(
        collate_fn=lambda b: b, num_workers=effective_workers, pin_memory=pin,
        **({"prefetch_factor": 4, "persistent_workers": True}
           if effective_workers > 0 else {}),
    )

    # With packing, each "batch" is many samples that get packed into fewer sequences.
    # Fetch more samples per batch so the packer has enough to fill sequences.
    fetch_batch_size = batch_size * 5 if use_packing else batch_size

    # Length-grouped batching reduces padding waste by ~50-70%
    if use_pretokenized and not use_packing:
        sampler = LengthGroupedSampler(
            train_ds._seq_lengths, batch_size=fetch_batch_size, shuffle=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=fetch_batch_size, sampler=sampler,
            drop_last=True, **dl_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=fetch_batch_size, shuffle=True,
            drop_last=True, **dl_kwargs,
        )

    if has_val:
        val_batch = batch_size if use_pretokenized else 1
        val_loader = DataLoader(
            val_ds, batch_size=val_batch, shuffle=False, **dl_kwargs,
        )

    # Gradient checkpointing: trades ~30% speed for ~40% memory savings.
    # Disable when VRAM is plentiful (e.g., 4B model on 80GB H100).
    raw_model_ref = model
    use_grad_ckpt = os.environ.get("GRAD_CHECKPOINT", "true").lower() != "false"
    if use_grad_ckpt and hasattr(raw_model_ref, "llm"):
        raw_model_ref.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if is_main:
            print("  Gradient checkpointing enabled")
    elif is_main:
        print("  Gradient checkpointing DISABLED (GRAD_CHECKPOINT=false)")

    # torch.compile for 15-30% speedup on A100/H100.
    # Disable with TORCH_COMPILE=false if it causes issues (e.g., dynamic shapes).
    use_compile = os.environ.get("TORCH_COMPILE", "false").lower() == "true"
    if use_compile and hasattr(raw_model_ref, "llm") and hasattr(torch, "compile"):
        try:
            raw_model_ref.llm = torch.compile(raw_model_ref.llm, mode="reduce-overhead")
            if is_main:
                print("  torch.compile enabled (reduce-overhead)")
        except Exception as e:
            if is_main:
                print(f"  torch.compile failed, continuing without: {e}")
    elif is_main and not use_compile:
        print("  torch.compile DISABLED (TORCH_COMPILE=false)")

    # Alignment mode: freeze DoRA, only TS embeddings train
    if align_only:
        n_frozen = raw_model_ref.freeze_lora()
        if is_main:
            print(f"  Alignment mode: froze {n_frozen:,} DoRA params, only TS embeddings trainable")

    # Optimizer: separate LR groups for embeddings vs adapters
    if embed_lr is not None:
        embed_params, other_params = [], []
        for name, param in raw_model_ref.named_parameters():
            if not param.requires_grad:
                continue
            if 'embed' in name.lower():
                embed_params.append(param)
            else:
                other_params.append(param)
        param_groups = [
            {"params": embed_params, "lr": embed_lr},
            {"params": other_params, "lr": lr},
        ]
        if is_main:
            n_embed = sum(p.numel() for p in embed_params)
            n_other = sum(p.numel() for p in other_params)
            print(f"  LR groups: embeddings ({n_embed:,} params, lr={embed_lr:.1e}) + "
                  f"adapters ({n_other:,} params, lr={lr:.1e})")
        optimizer = AdamW(param_groups)
    else:
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
        )

    # Accelerate prepares model, optimizer, dataloaders
    # Must happen BEFORE scheduler so total_steps reflects the sharded dataloader
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader,
    )
    if has_val:
        val_loader = accelerator.prepare(val_loader)

    # Scheduler uses post-prepare dataloader length (correct for multi-GPU)
    total_steps = (len(train_loader) // grad_accum) * epochs
    warmup_steps = max(1, int(warmup_frac * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: min(1.0, s / warmup_steps) if s < warmup_steps
                            else max(0.1, 1.0 - (s - warmup_steps) / max(1, total_steps - warmup_steps)),
    )
    scheduler = accelerator.prepare(scheduler)

    best_val = float("inf")
    patience_left = patience

    # Unwrap model for validation and checkpoint access (not for forward pass)
    raw_model = accelerator.unwrap_model(model)

    if augment:
        from .augment import augment_ts

    # Checkpoint directory — SageMaker auto-syncs this to S3
    ckpt_dir = os.environ.get("SM_HP_CHECKPOINT_DIR",
               os.environ.get("CHECKPOINT_DIR", "/opt/ml/checkpoints"))
    # Create a job-specific subfolder with clear naming
    job_name = os.environ.get("SAGEMAKER_JOB_NAME",
               os.environ.get("SM_HP_SAGEMAKER_JOB_NAME", "local"))
    ckpt_job_dir = Path(ckpt_dir) / job_name
    if is_main:
        ckpt_job_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Checkpoints -> {ckpt_job_dir}  (synced to S3)")

    # Resume from mid-epoch checkpoint if available (spot recovery).
    # Uses accelerator.save_state/load_state so all ranks stay in sync
    # (works on both shared and non-shared filesystems).
    start_epoch = 1
    skip_batches = 0
    mid_ckpt_dir = ckpt_job_dir / "mid_epoch_state"
    mid_ckpt_meta = ckpt_job_dir / "mid_epoch_meta.pt"
    if mid_ckpt_dir.exists() and mid_ckpt_meta.exists():
        if is_main:
            print(f"  Resuming from {mid_ckpt_dir}...", flush=True)
        accelerator.load_state(str(mid_ckpt_dir))
        meta = torch.load(str(mid_ckpt_meta), map_location="cpu", weights_only=False)
        start_epoch = meta.get("epoch", 1)
        skip_batches = meta.get("batch_idx", 0)
        if is_main:
            print(f"  Resumed: epoch {start_epoch}, batch {skip_batches}, "
                  f"loss {meta.get('train_loss', '?')}", flush=True)
        accelerator.wait_for_everyone()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running, n_batches = 0.0, 0
        for batch_idx, batch in enumerate(train_loader):
            # Skip batches if resuming mid-epoch
            if epoch == start_epoch and batch_idx < skip_batches:
                continue
            if augment:
                for sample in batch:
                    ts = sample.get("time_series", [])
                    if ts and isinstance(ts, list):
                        sample["time_series"] = [
                            augment_ts(ch) if isinstance(ch, torch.Tensor) else ch
                            for ch in ts
                        ]

            with accelerator.accumulate(model):
                if use_packing:
                    loss = model(batch, mode="packed", pack_length=pack_length)
                elif use_pretokenized:
                    loss = model(batch, mode="pretokenized")
                else:
                    loss = model(batch, mode="loss")
                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item()
            n_batches += 1
            global_step = (epoch - 1) * len(train_loader) + batch_idx
            if is_main and batch_idx % 1000 == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"  [{epoch}/{epochs}] batch {batch_idx}/{len(train_loader)} "
                      f"loss={loss.item():.4f} lr={cur_lr:.2e}")
                sys.stdout.flush()

            # Mid-epoch checkpoint for spot instance recovery (all ranks sync)
            if batch_idx > 0 and batch_idx % 1000 == 0:
                accelerator.save_state(str(mid_ckpt_dir))
                if is_main:
                    torch.save({
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "global_step": global_step,
                        "train_loss": running / n_batches,
                    }, mid_ckpt_meta)
                    print(f"    [checkpoint] batch {batch_idx} saved", flush=True)
                    if use_wandb:
                        cur_lr = optimizer.param_groups[0]['lr']
                        wandb.log({"train/loss": loss.item(), "train/lr": cur_lr,
                                   "train/step": global_step}, step=global_step)
                accelerator.wait_for_everyone()
        avg_train = running / max(n_batches, 1)

        # Validate (aggregate loss across all GPUs)
        avg_val = float("inf")
        if has_val:
            model.eval()
            v_loss, v_n = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    try:
                        if use_pretokenized:
                            val_item = raw_model.compute_loss_pretokenized(batch).item()
                        else:
                            val_item = raw_model.compute_loss(batch).item()
                        if val_item == val_item:
                            v_loss += val_item
                            v_n += 1
                    except Exception as e:
                        if v_n == 0 and is_main:
                            print(f"  Val error: {e}")
                        continue
            # Aggregate val loss across all ranks for consistent early stopping
            v_stats = torch.tensor([v_loss, float(v_n)], device=device)
            v_stats = accelerator.reduce(v_stats, reduction="sum")
            total_v_loss = v_stats[0].item()
            total_v_n = int(v_stats[1].item())
            avg_val = total_v_loss / max(total_v_n, 1) if total_v_n > 0 else float("inf")

        if is_main:
            if has_val:
                print(f"  Epoch {epoch}: train={avg_train:.4f}  val={avg_val:.4f}  "
                      f"(best={best_val:.4f}, patience={patience_left})")
            else:
                print(f"  Epoch {epoch}: train={avg_train:.4f}")
            if use_wandb:
                log = {"epoch/train_loss": avg_train, "epoch": epoch}
                if has_val:
                    log["epoch/val_loss"] = avg_val
                wandb.log(log, step=global_step)

            save_model = raw_model
            from dataclasses import asdict
            model_config = asdict(raw_model.config) if hasattr(raw_model, "config") else {}

            epoch_state = {
                "model_state": save_model.state_dict(),
                "epoch": epoch,
                "train_loss": avg_train,
                "val_loss": avg_val if has_val else None,
                "config": model_config,
            }

            # Save to output dir (local)
            torch.save(epoch_state, Path(output_dir) / "last_model.pt")

            # Save every epoch to checkpoint dir with clear naming (synced to S3)
            phase_tag = "phase0" if align_only else "phase1"
            epoch_name = f"{phase_tag}_epoch{epoch}_val{avg_val:.4f}.pt" if has_val \
                    else f"{phase_tag}_epoch{epoch}_train{avg_train:.4f}.pt"
            torch.save(epoch_state, ckpt_job_dir / epoch_name)
            print(f"  -> Saved {epoch_name} to S3 checkpoint dir")

            # Track best + early stopping
            if has_val and avg_val < best_val - 1e-4:
                best_val = avg_val
                patience_left = patience
                torch.save(epoch_state, Path(output_dir) / "best_model.pt")
                torch.save(epoch_state, ckpt_job_dir / f"{phase_tag}_best.pt")
                print(f"  -> New best (val={avg_val:.4f})")
            elif has_val:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

        sys.stdout.flush()

    # Restore DoRA training after alignment stage
    if align_only:
        raw_model_ref.unfreeze_lora()
        if is_main:
            print("  Alignment complete — DoRA adapters unfrozen for next stage")

    best_path = Path(output_dir) / "best_model.pt"
    if not best_path.exists():
        best_path = Path(output_dir) / "last_model.pt"
    return {"best_val_loss": best_val, "best_epoch": epoch, "checkpoint_path": str(best_path)}
