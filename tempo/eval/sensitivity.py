"""Sensitivity test: the core diagnostic for TS-LLM signal usage.

Measures whether a model actually uses its time series input by
replacing the real signal with Gaussian noise and measuring the
accuracy drop (Delta = Acc_real - Acc_noise).

Delta > 0  → model uses the signal
Delta ≈ 0  → model ignores the signal (exploits text shortcuts)

Usage:
    from tempo.eval import sensitivity_test

    results = sensitivity_test(model, test_loader, extract_fn=extract_label)
    print(f"Acc={results['real_acc']:.1%}  Delta={results['delta']:.1%}")
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import DataLoader

from .scorer import extract_answer


def sensitivity_test(
    model,
    test_loader: DataLoader,
    task: str = "classification",
    max_samples: int = 200,
    max_new_tokens: int = 400,
) -> dict[str, float]:
    """Run the sensitivity test on a model + test set.

    For each sample:
    1. Generate with real signal → extract answer → compare to gold
    2. Generate with noise signal → extract answer → compare to gold
    3. Check if outputs differ (divergence)

    Args:
        model: TEMPO model (must have .generate() and .build_text()).
        test_loader: DataLoader yielding batches of sample dicts.
        task: Task type for answer extraction.
        max_samples: Cap on test samples.
        max_new_tokens: Tokens to generate per sample.

    Returns:
        Dict with real_acc, noise_acc, delta, divergence, n_samples.
    """
    model.eval()
    real_correct, noise_correct, diverge_count, total = 0, 0, 0, 0

    with torch.no_grad(), torch.amp.autocast(model.device_str, dtype=torch.bfloat16):
        for batch in test_loader:
            if total >= max_samples:
                break

            for sample in (batch if isinstance(batch, list) else [batch]):
                if total >= max_samples:
                    break

                gold = sample.get("answer", "")
                gold_ans = extract_answer(gold, task)

                # Generate with real signal
                real_out = model.generate([sample], max_new_tokens=max_new_tokens)
                real_text = real_out[0] if real_out else ""
                real_ans = extract_answer(real_text, task)

                # Generate with noise signal
                noise_sample = _replace_signal_with_noise(sample)
                noise_out = model.generate([noise_sample], max_new_tokens=max_new_tokens)
                noise_text = noise_out[0] if noise_out else ""
                noise_ans = extract_answer(noise_text, task)

                # Score
                if gold_ans and real_ans and (gold_ans.lower() in real_ans.lower() or
                                               real_ans.lower() in gold_ans.lower()):
                    real_correct += 1
                if gold_ans and noise_ans and (gold_ans.lower() in noise_ans.lower() or
                                                noise_ans.lower() in gold_ans.lower()):
                    noise_correct += 1
                if real_text[:300] != noise_text[:300]:
                    diverge_count += 1

                total += 1

    real_acc = real_correct / max(total, 1)
    noise_acc = noise_correct / max(total, 1)
    return {
        "real_acc": real_acc,
        "noise_acc": noise_acc,
        "delta": real_acc - noise_acc,
        "divergence": diverge_count / max(total, 1),
        "n_samples": total,
    }


def _replace_signal_with_noise(sample: dict) -> dict:
    """Replace time series with Gaussian noise of matched statistics."""
    noise_sample = dict(sample)
    ts = sample.get("time_series", [])

    if isinstance(ts, list) and len(ts) > 0:
        noisy = []
        for t in ts:
            if isinstance(t, torch.Tensor):
                noisy.append(torch.randn_like(t) * t.std() + t.mean())
            else:
                noisy.append(t)
        noise_sample["time_series"] = noisy
    elif isinstance(ts, torch.Tensor):
        noise_sample["time_series"] = torch.randn_like(ts) * ts.std() + ts.mean()

    return noise_sample
