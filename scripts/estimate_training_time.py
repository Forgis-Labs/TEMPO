"""Estimate training time and cost for TEMPO on various GPU configurations.

Uses actual model profiling to measure per-step time, then extrapolates.

Usage:
    uv run python tempo/estimate_training_time.py
"""

import sys
import time
import argparse
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def measure_step_time(
    model_id: str = "Qwen/Qwen3-4B",
    lora_r: int = 32,
    seq_len: int = 1024,
    batch_size: int = 4,
    n_warmup: int = 3,
    n_measure: int = 10,
    align_only: bool = False,
    device: str = "cuda",
):
    """Measure actual forward+backward time per step on this GPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        use_dora=True,
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    if align_only:
        # Freeze LoRA, only embeddings train
        for name, param in model.named_parameters():
            if 'lora' in name.lower() or 'dora' in name.lower():
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Measure VRAM
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated() / 1e9

    # Create dummy batch
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5
    )

    model.train()

    # Warmup
    print(f"  Warming up ({n_warmup} steps)...")
    for _ in range(n_warmup):
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()

    # Measure
    print(f"  Measuring ({n_measure} steps, batch_size={batch_size}, seq_len={seq_len})...")
    times = []
    for _ in range(n_measure):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    mean_time = np.mean(times)
    std_time = np.std(times)

    print(f"  Step time: {mean_time:.3f} +/- {std_time:.3f} sec")
    print(f"  VRAM: {peak_mem:.1f} GB peak")
    print(f"  Tokens/sec: {batch_size * seq_len / mean_time:.0f}")

    del model, optimizer
    torch.cuda.empty_cache()

    return {
        "mean_step_time": mean_time,
        "std_step_time": std_time,
        "peak_vram_gb": peak_mem,
        "tokens_per_sec": batch_size * seq_len / mean_time,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "trainable_params": trainable,
    }


def estimate_training(
    step_time: float,
    n_samples: int,
    batch_size: int,
    grad_accum: int,
    n_gpus: int,
    epochs: int,
    patience_epochs: int | None = None,
    cost_per_hour: float = 0.57,
):
    """Calculate training time from measured step time."""
    effective_batch = batch_size * n_gpus * grad_accum
    steps_per_epoch = n_samples / (batch_size * n_gpus)  # forward/backward steps
    optimizer_steps_per_epoch = steps_per_epoch / grad_accum

    # DDP overhead: ~5% per additional GPU for gradient sync
    ddp_overhead = 1.0 + 0.05 * max(0, n_gpus - 1)
    adjusted_step_time = step_time * ddp_overhead

    total_steps = steps_per_epoch * epochs
    total_seconds = total_steps * adjusted_step_time
    total_hours = total_seconds / 3600

    # With early stopping
    if patience_epochs:
        likely_epochs = min(epochs, epochs * 0.6)  # rough: converges at ~60% of max
        likely_hours = total_hours * (likely_epochs / epochs)
    else:
        likely_hours = total_hours

    cost = total_hours * cost_per_hour
    likely_cost = likely_hours * cost_per_hour

    return {
        "effective_batch_size": effective_batch,
        "steps_per_epoch": int(steps_per_epoch),
        "optimizer_steps_per_epoch": int(optimizer_steps_per_epoch),
        "total_steps": int(total_steps),
        "step_time_adjusted": adjusted_step_time,
        "total_hours_max": total_hours,
        "total_hours_likely": likely_hours,
        "cost_max": cost,
        "cost_likely": likely_cost,
    }


def main():
    parser = argparse.ArgumentParser(description="Estimate TEMPO training time")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--measure", action="store_true",
                        help="Actually measure step time on local GPU (requires CUDA)")
    parser.add_argument("--step-time", type=float, default=None,
                        help="Override step time in seconds (skip measurement)")
    args = parser.parse_args()

    if args.measure and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")
        print()

        # Measure phase 0 (align only)
        print("=== Phase 0 (alignment, DoRA frozen) ===")
        p0_result = measure_step_time(
            model_id=args.model, batch_size=4, seq_len=512,
            align_only=True,
        )
        print()

        # Measure phase 1 (full training)
        print("=== Phase 1 (full training, DoRA active) ===")
        p1_result = measure_step_time(
            model_id=args.model, batch_size=4, seq_len=1024,
            align_only=False,
        )

        step_time_p0 = p0_result["mean_step_time"]
        step_time_p1 = p1_result["mean_step_time"]
        measured_gpu = gpu_name
    else:
        if args.step_time:
            step_time_p0 = args.step_time * 0.6  # align is ~60% of full
            step_time_p1 = args.step_time
            measured_gpu = "manual"
        else:
            # Published estimates for Qwen3-4B LoRA on various GPUs
            # Based on research: ~3000-4000 tok/sec on A10G for 4B LoRA
            # At seq_len=1024, batch=4: 4096 tokens/step
            # 4096 / 3500 = ~1.17 sec/step for full training
            step_time_p0 = 0.5   # alignment only, less compute
            step_time_p1 = 1.1   # full LoRA+DoRA training
            measured_gpu = "A10G (estimated from benchmarks)"

        print(f"Using estimated step times for {measured_gpu}")
        print(f"  Phase 0: {step_time_p0:.2f} sec/step (batch=4, seq=512)")
        print(f"  Phase 1: {step_time_p1:.2f} sec/step (batch=4, seq=1024)")
        print()

    configs = [
        # (name, n_gpus, batch_size, grad_accum, cost_per_hr_spot, instance)
        ("1x A10G (g5.2xlarge)", 1, 4, 8, 0.57, "ml.g5.2xlarge"),
        ("4x A10G (g5.12xlarge)", 4, 4, 2, 2.00, "ml.g5.12xlarge"),
        ("8x A10G (g5.48xlarge)", 8, 4, 1, 5.00, "ml.g5.48xlarge"),
        ("1x A100-40G (p4d, 1 GPU)", 1, 8, 4, 11.30, "ml.p4d.24xlarge"),
    ]

    # Scale step time for A100 (roughly 2x faster than A10G)
    a100_factor = 0.5

    print("=" * 80)
    print("PHASE 0: EMBEDDING ALIGNMENT")
    print(f"  Samples: 49,121 | Epochs: 3 | Patience: 3")
    print("=" * 80)
    print(f"{'Config':<30} {'Eff.Batch':>9} {'Steps/ep':>9} {'Time(max)':>10} {'Time(likely)':>12} {'Cost':>8}")
    print("-" * 80)

    for name, n_gpus, bs, ga, cost_hr, instance in configs:
        st = step_time_p0 * (a100_factor if "A100" in name else 1.0)
        est = estimate_training(st, 49121, bs, ga, n_gpus, epochs=3,
                                patience_epochs=3, cost_per_hour=cost_hr)
        print(f"{name:<30} {est['effective_batch_size']:>9} {est['steps_per_epoch']:>9} "
              f"{est['total_hours_max']:>9.1f}h {est['total_hours_likely']:>11.1f}h "
              f"${est['cost_likely']:>6.1f}")

    print()
    print("=" * 80)
    print("PHASE 1: FULL TRAINING (DoRA + embeddings)")
    print(f"  Samples: 300,000 | Epochs: 10 | Patience: 3")
    print("=" * 80)
    print(f"{'Config':<30} {'Eff.Batch':>9} {'Steps/ep':>9} {'Time(max)':>10} {'Time(likely)':>12} {'Cost':>8}")
    print("-" * 80)

    for name, n_gpus, bs, ga, cost_hr, instance in configs:
        st = step_time_p1 * (a100_factor if "A100" in name else 1.0)
        est = estimate_training(st, 300000, bs, ga, n_gpus, epochs=10,
                                patience_epochs=3, cost_per_hour=cost_hr)
        print(f"{name:<30} {est['effective_batch_size']:>9} {est['steps_per_epoch']:>9} "
              f"{est['total_hours_max']:>9.1f}h {est['total_hours_likely']:>11.1f}h "
              f"${est['cost_likely']:>6.1f}")

    print()
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print("  Phase 0: g5.2xlarge (1x A10G) — cheapest, fast enough")
    print("  Phase 1: g5.12xlarge (4x A10G, DDP) — best speed/cost tradeoff")
    print()
    print("  NOTE: These estimates assume ~3500 tokens/sec on A10G for Qwen3-4B LoRA.")
    print("  Run with --measure on a GPU machine for actual numbers.")
    print("  DDP overhead estimated at 5% per additional GPU (gradient sync only).")


if __name__ == "__main__":
    main()
