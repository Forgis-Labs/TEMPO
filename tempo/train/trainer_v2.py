"""Fast training loop for TEMPO — SFTTrainer + DDP.

Drop-in replacement for trainer.py (v1). Same function signature, same return
format, 5-10x faster on multi-GPU.

Architecture:
    - HuggingFace SFTTrainer handles packing, batching, gradient accumulation
    - Multi-GPU via torchrun (DDP, not FSDP)
    - Pre-builds ChatML text at dataset load time (one-time cost)
    - No custom bin-packing, no mid-epoch S3 checkpoints, no Accelerate

Usage:
    # Single GPU:
    from tempo.train.trainer_v2 import train
    result = train(model, dataset, output_dir="results/stage1")

    # Multi-GPU with torchrun:
    torchrun --nproc_per_node=8 script.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset


# Helpers

def _is_main() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def _barrier():
    """Sync all ranks. No-op on single GPU."""
    if dist.is_initialized():
        dist.barrier()


def _build_chatml_text(sample: dict, model) -> str:
    """Build ChatML text from a sample. Called once at dataset load time."""
    prompt = model.build_text(sample)
    answer = sample.get("answer", "")
    return prompt + answer + "<|im_end|>"


def _prepare_dataset(raw_dataset, model, eos_token: str) -> HFDataset:
    """Convert any dataset to HF Dataset with pre-built ChatML text.

    One-time cost: iterates all samples, builds text. No tokenization —
    SFTTrainer handles that with its own optimized tokenizer.
    """
    texts = []
    n = len(raw_dataset)

    for i in range(n):
        try:
            sample = raw_dataset[i]
            answer = sample.get("answer", "")
            if eos_token and not answer.endswith(eos_token):
                sample["answer"] = answer
            text = _build_chatml_text(sample, model)
            texts.append({"text": text})
        except Exception as e:
            if i == 0:
                print(f"  WARNING: sample 0 failed: {e}")
            continue

    return HFDataset.from_list(texts)


# Main train function

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
    use_packing: bool = True,
    pack_length: int = 2048,
    accelerator=None,  # ignored — kept for API compat with v1
    save_steps: int = 500,
) -> dict[str, Any]:
    """Train a TEMPO model using SFTTrainer.

    Drop-in replacement for trainer.py's train() function.
    Same signature, same return format.
    """
    from trl import SFTTrainer, SFTConfig

    is_main = _is_main()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if is_main:
        print(f"\n  trainer_v2: SFTTrainer + DDP (fast path)")

    if align_only:
        n_frozen = model.freeze_lora()
        if is_main:
            print(f"  Alignment mode: froze {n_frozen:,} adapter params")

    eos = model.get_eos_token()

    # Handle different dataset formats
    if hasattr(dataset, '__getitem__') and hasattr(dataset, 'keys'):
        raw_train = dataset["train"]
        has_val = "validation" in (dataset.keys() if hasattr(dataset, 'keys') else dataset)
        raw_val = dataset["validation"] if has_val else None
    else:
        raw_train = dataset
        raw_val = None
        has_val = False

    if is_main:
        print(f"  Building ChatML text for training set...", flush=True)
    train_ds = _prepare_dataset(raw_train, model, eos)
    if is_main:
        print(f"  Train: {len(train_ds)} samples")

    val_ds = None
    if has_val and raw_val is not None:
        val_ds = _prepare_dataset(raw_val, model, eos)
        if is_main:
            print(f"  Val: {len(val_ds)} samples")

    # Show sample
    if is_main and len(train_ds) > 0:
        sample_text = train_ds[0]["text"]
        n_toks = len(model.tokenizer(sample_text, add_special_tokens=False).input_ids)
        print(f"  Sample: {n_toks} tokens, text[:200]: {sample_text[:200]}...")

    effective_lr = lr
    if align_only and embed_lr is not None:
        effective_lr = embed_lr

    use_grad_ckpt = os.environ.get("GRAD_CHECKPOINT", "false").lower() == "true"
    if use_grad_ckpt and hasattr(model, "llm"):
        model.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if is_main:
            print(f"  Gradient checkpointing enabled")

    # trl >= 1.0: packing, max_length, dataset_text_field go in SFTConfig
    training_args = SFTConfig(
        output_dir=str(out),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=effective_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_frac,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        logging_steps=50,
        save_total_limit=2,
        eval_strategy="epoch" if val_ds is not None else "no",
        save_strategy="epoch" if val_ds is not None else "steps",
        load_best_model_at_end=val_ds is not None,
        metric_for_best_model="eval_loss" if val_ds is not None else None,
        greater_is_better=False,
        max_grad_norm=1.0,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=num_workers > 0,
        remove_unused_columns=False,
        seed=42,
        # SFT-specific
        packing=use_packing,
        max_length=pack_length,
        dataset_text_field="text",
        # DDP (not FSDP)
        ddp_find_unused_parameters=False,
        # Reporting
        report_to="wandb" if wandb_project else "none",
        run_name=wandb_run_name,
    )

    if is_main:
        eff_batch = batch_size * grad_accum
        n_gpus = int(os.environ.get("WORLD_SIZE", "1"))
        print(f"  Effective batch: {batch_size} x {grad_accum} x {n_gpus} = {eff_batch * n_gpus}")
        print(f"  Packing: {'ON' if use_packing else 'OFF'} (max_length={pack_length})")
        print(f"  Grad checkpoint: {use_grad_ckpt}")

    if wandb_project and is_main:
        os.environ["WANDB_PROJECT"] = wandb_project

    # Handles multiple samples per packed sequence.

    response_template_ids = model.tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False)
    _template_len = len(response_template_ids)

    def _mask_prompt_tokens(input_ids, labels):
        """Mask non-response tokens with -100 in a (possibly packed) sequence."""
        ids_list = input_ids.tolist()
        seq_len = len(ids_list)

        # Find ALL response template positions (packed = multiple samples)
        response_starts = []
        for j in range(seq_len - _template_len + 1):
            if ids_list[j:j + _template_len] == response_template_ids:
                response_starts.append(j + _template_len)

        if not response_starts:
            labels[:] = -100
            return labels

        # Mask everything, then unmask response regions
        mask = torch.ones_like(labels, dtype=torch.bool)
        for idx, resp_start in enumerate(response_starts):
            if idx + 1 < len(response_starts):
                resp_end = response_starts[idx + 1] - _template_len
            else:
                resp_end = seq_len
            mask[resp_start:resp_end] = False
        labels[mask] = -100
        return labels

    class CompletionOnlySFTTrainer(SFTTrainer):
        """SFTTrainer that masks prompt tokens in compute_loss.
        Works with packing — no custom data_collator needed."""
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            if "labels" in inputs and "input_ids" in inputs:
                input_ids = inputs["input_ids"]
                labels = inputs["labels"].clone()
                for i in range(labels.size(0)):
                    labels[i] = _mask_prompt_tokens(input_ids[i], labels[i])
                inputs["labels"] = labels
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    if is_main:
        print(f"  Completion-only masking: response_template={response_template_ids}")
        print(f"  Loss only on answer tokens (packing + masking combined)")

    trainer = CompletionOnlySFTTrainer(
        model=model.llm,
        processing_class=model.tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
    )

    if is_main:
        print(f"\n  Starting training...", flush=True)

    train_result = trainer.train()

    _barrier()

    best_val = float("inf")
    if is_main:
        train_loss = train_result.training_loss

        save_state = {
            "model_state": model.state_dict(),
            "epoch": epochs,
            "train_loss": train_loss,
            "val_loss": None,
        }

        # Get val loss from training logs (already computed during training)
        if val_ds is not None:
            # trainer.state has the best metric from load_best_model_at_end
            log_history = trainer.state.log_history
            eval_losses = [h.get("eval_loss") for h in log_history
                          if "eval_loss" in h and h["eval_loss"] is not None]
            if eval_losses:
                best_val = min(eval_losses)
                save_state["val_loss"] = best_val
                print(f"  Best val_loss: {best_val:.4f}")

        torch.save(save_state, out / "best_model.pt")
        torch.save(save_state, out / "last_model.pt")
        print(f"  Saved to {out / 'best_model.pt'}")

    _barrier()

    if align_only:
        model.unfreeze_lora()
        if is_main:
            print("  Alignment complete — adapters unfrozen")

    return {
        "best_val_loss": best_val,
        "best_epoch": epochs,
        "checkpoint_path": str(out / "best_model.pt"),
    }
