"""Time series augmentation for training.

Applied BEFORE tokenization (on raw signal) so that the TOTEM codes
change each epoch, giving the model diverse views of each sample.

Only augmentations that survive z-score normalization are included:
- Jitter: additive Gaussian noise changes local shape
- Window crop: random offset changes code alignment
- Time warp: local stretching/compression changes temporal patterns

Amplitude scaling is NOT included — z-score normalization erases it.

Usage:
    from tempo.train.augment import augment_ts

    # In training loop, augment each sample's time_series before compute_loss
    for sample in batch:
        sample["time_series"] = [augment_ts(ch) for ch in sample["time_series"]]
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F


def augment_ts(
    ts: torch.Tensor,
    jitter_std: float = 0.05,
    crop_ratio: float = 0.1,
    warp_prob: float = 0.3,
    warp_segments: int = 4,
    warp_strength: float = 0.2,
) -> torch.Tensor:
    """Apply random augmentations to a time series tensor.

    Args:
        ts: 1D or 2D time series tensor.
        jitter_std: Std of additive Gaussian noise (relative to signal std).
        crop_ratio: Max fraction to crop from start/end (0.1 = up to 10%).
        warp_prob: Probability of applying time warping.
        warp_segments: Number of segments for time warping.
        warp_strength: Max relative speed change per segment.

    Returns:
        Augmented tensor (same shape as input).
    """
    orig_shape = ts.shape
    ts = ts.float().flatten()
    n = len(ts)

    if n < 8:
        return ts.view(orig_shape)

    # 1. Jitter — additive noise proportional to signal std
    if jitter_std > 0:
        noise_scale = jitter_std * ts.std()
        ts = ts + torch.randn_like(ts) * noise_scale

    # 2. Window crop — random start/end offset, then resize back
    if crop_ratio > 0 and random.random() < 0.5:
        max_crop = max(1, int(n * crop_ratio))
        start = random.randint(0, max_crop)
        end = n - random.randint(0, max_crop)
        if end - start >= n // 2:  # don't crop more than half
            cropped = ts[start:end]
            # Interpolate back to original length
            ts = F.interpolate(
                cropped.unsqueeze(0).unsqueeze(0),
                size=n, mode='linear', align_corners=False,
            ).squeeze()

    # 3. Time warp — locally stretch/compress segments
    if warp_prob > 0 and random.random() < warp_prob:
        ts = _time_warp(ts, warp_segments, warp_strength)

    return ts.view(orig_shape)


def _time_warp(ts: torch.Tensor, n_segments: int, strength: float) -> torch.Tensor:
    """Warp time axis by locally speeding up / slowing down segments."""
    n = len(ts)
    seg_len = n // n_segments

    if seg_len < 4:
        return ts

    # Generate random speed factors per segment
    speeds = [1.0 + random.uniform(-strength, strength) for _ in range(n_segments)]

    # Build warped time indices
    warped_pieces = []
    for i, speed in enumerate(speeds):
        start = i * seg_len
        end = start + seg_len if i < n_segments - 1 else n
        seg = ts[start:end]

        # Resample segment: speed > 1 = compress (fewer samples), speed < 1 = stretch
        new_len = max(2, int(len(seg) / speed))
        resampled = F.interpolate(
            seg.unsqueeze(0).unsqueeze(0),
            size=new_len, mode='linear', align_corners=False,
        ).squeeze()
        warped_pieces.append(resampled)

    warped = torch.cat(warped_pieces)

    # Resize back to original length
    if len(warped) != n:
        warped = F.interpolate(
            warped.unsqueeze(0).unsqueeze(0),
            size=n, mode='linear', align_corners=False,
        ).squeeze()

    return warped
