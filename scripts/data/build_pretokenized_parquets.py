"""Build pre-tokenized parquet files for TEMPO training.

Uses the existing data builders (build_alignment_dataset for phase0,
stage1_unified for phase1), tokenizes all time series with the specified
tokenizer, and saves as parquet files ready for SageMaker.

Usage:
    # Build with RoPE tokenizer
    uv run python tempo/build_pretokenized_parquets.py \
        --tokenizer fsq_transformer_rope \
        --tokenizer-ckpt checkpoints/tokenizers/fsq_transformer_rope_625_best.pt \
        --output-dir data/pretokenized_rope

    # Then upload to S3
    aws s3 sync data/pretokenized_rope/phase0 s3://<your-s3-bucket>/tempo/phase0/
    aws s3 sync data/pretokenized_rope/phase1 s3://<your-s3-bucket>/tempo/phase1/
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm

# Path bootstrap: add project root for `tempo.*` imports, and the parent
# OpenTSLM directory's src/ for `from opentslm import ...`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARENT_ROOT = PROJECT_ROOT.parent  # OpenTSLM root (has src/opentslm)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PARENT_ROOT / "src"))


def load_tokenizer(tokenizer_type: str, ckpt_path: str, device: str = "cpu"):
    """Load the specified tokenizer."""
    if tokenizer_type == "fsq_transformer_rope":
        from tempo.tokenizer.fsq_transformer_rope import FSQTransformerRoPETokenizer
        return FSQTransformerRoPETokenizer.from_pretrained(ckpt_path, device=device)
    elif tokenizer_type == "fsq_transformer":
        from tempo.tokenizer.fsq_transformer import FSQTransformerTokenizer
        return FSQTransformerTokenizer.from_pretrained(ckpt_path, device=device)
    elif tokenizer_type == "fsq":
        from tempo.tokenizer.fsq import FSQTokenizer
        return FSQTokenizer.from_pretrained(ckpt_path, device=device)
    elif tokenizer_type == "totem":
        from tempo.tokenizer.totem import TOTEMTokenizer
        return TOTEMTokenizer.from_pretrained(ckpt_path, device=device)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def _tokenize_single_channel(tokenizer, ts_flat: np.ndarray, max_ts_tokens: int,
                              device: str) -> tuple[list[int], float, float]:
    """Tokenize one 1D signal. Returns (codes, mean, std)."""
    ts_flat = ts_flat[~np.isnan(ts_flat)]
    if len(ts_flat) < 8 or ts_flat.std() < 1e-8:
        return [], 0.0, 0.0

    mean_val = float(ts_flat.mean())
    std_val = float(ts_flat.std())
    ts_norm = (ts_flat - mean_val) / std_val

    max_raw = max_ts_tokens * 4
    if len(ts_norm) > max_raw:
        idx = np.linspace(0, len(ts_norm) - 1, max_raw).astype(int)
        ts_norm = ts_norm[idx]

    pad = (4 - len(ts_norm) % 4) % 4
    if pad > 0:
        ts_norm = np.pad(ts_norm, (0, pad), mode='edge')

    with torch.no_grad():
        t = torch.from_numpy(ts_norm).float().unsqueeze(0).to(device)
        codes = tokenizer.tokenize(t)

    return codes.squeeze(0).cpu().tolist(), mean_val, std_val


def tokenize_signal(tokenizer, signal, max_ts_tokens: int, device: str) -> tuple[list[int], float, float]:
    """Tokenize a time series (single or multi-channel).

    Each channel is normalized independently and tokenized separately.
    Codes are concatenated. Mean/std are global (for metadata only).

    Returns (codes, mean, std).
    """
    try:
        if isinstance(signal, list) and len(signal) > 0 and isinstance(signal[0], list):
            # Multi-channel: tokenize each independently
            all_codes = []
            all_vals = []
            per_ch_tokens = max_ts_tokens // len(signal)  # split budget
            for ch in signal:
                ch_arr = np.array(ch, dtype=np.float32).flatten()
                codes, _, _ = _tokenize_single_channel(tokenizer, ch_arr, per_ch_tokens, device)
                all_codes.extend(codes)
                all_vals.extend(ch)
            all_vals = np.array(all_vals, dtype=np.float32)
            return all_codes, float(all_vals.mean()), float(all_vals.std())
        else:
            ts_flat = np.array(signal, dtype=np.float32).flatten()
            return _tokenize_single_channel(tokenizer, ts_flat, max_ts_tokens, device)
    except (ValueError, TypeError):
        return [], 0.0, 0.0


def pretokenize_hf_dataset(hf_dataset_dict, tokenizer, max_ts_tokens: int,
                            device: str) -> dict[str, list[dict]]:
    """Tokenize all time series in an HF DatasetDict.

    Returns dict of split_name -> list of sample dicts with ts_codes added.
    If a sample already has ts_codes (e.g., imputation), preserves them.
    """
    result = {}
    for split_name, split_data in hf_dataset_dict.items():
        print(f"\n  Tokenizing {split_name} ({len(split_data)} samples)...")
        samples = []
        for i in tqdm(range(len(split_data)), desc=f"  {split_name}"):
            row = split_data[i]
            ts = row.get("time_series", [])

            # Preserve pre-existing ts_codes (e.g., imputation samples with mask sentinels)
            existing_codes = row.get("ts_codes", [])
            if existing_codes and isinstance(existing_codes, list) and len(existing_codes) > 0:
                codes = existing_codes
                mean = row.get("ts_mean", 0.0)
                std = row.get("ts_std", 0.0)
            elif ts and isinstance(ts, list) and len(ts) > 0:
                codes, mean, std = tokenize_signal(tokenizer, ts, max_ts_tokens, device)
            else:
                codes, mean, std = [], 0.0, 0.0

            samples.append({
                "time_series": ts if ts else [],
                "time_series_text": row.get("time_series_text", []),
                "pre_prompt": row.get("pre_prompt", ""),
                "post_prompt": row.get("post_prompt", ""),
                "answer": row.get("answer", ""),
                "ts_codes": codes,
                "ts_mean": mean,
                "ts_std": std,
                "task": row.get("task", ""),
                "domain": row.get("domain", ""),
            })

        result[split_name] = samples
        n_with_ts = sum(1 for s in samples if s["ts_codes"])
        print(f"    {len(samples)} samples ({n_with_ts} with TS codes, "
              f"{len(samples) - n_with_ts} text-only)")
    return result


def _extract_ts_from_text(text: str) -> list[float] | None:
    """Extract numerical time series array from question/prompt text.

    Handles patterns like:
      "Given the time series [-0.31, 0.29, 0.27, ...], is there ..."
      "The input Time Serie is [1.2, 3.4, 5.6]."
    """
    m = re.search(
        r'\[[\s\n]*(-?[\d.]+(?:[\s\n]*,[\s\n]*-?[\d.eE+-]+){4,})[\s\n]*\]',
        text,
    )
    if not m:
        return None
    try:
        vals = [float(x.strip()) for x in m.group(1).split(",") if x.strip()]
        return vals if len(vals) >= 8 else None
    except ValueError:
        return None


def _strip_ts_from_text(text: str) -> str:
    """Remove raw numerical arrays from text (replaced by TOTEM codes)."""
    return re.sub(
        r'\[[\s\n]*-?[\d.]+(?:[\s\n]*,[\s\n]*-?[\d.eE+-]+)*[\s\n]*\]',
        '[signal provided above]',
        text,
    )


def pretokenize_raw_samples(splits: dict[str, list[dict]], tokenizer,
                             max_ts_tokens: int, device: str) -> dict[str, list[dict]]:
    """Tokenize raw sample dicts (bypasses HF Dataset Arrow conversion).

    For samples with time_series data: tokenize and strip raw numbers from text.
    For text-only samples: try to extract time series from question text,
    tokenize it, and strip the raw numbers.
    """
    result = {}
    for split_name, raw_samples in splits.items():
        print(f"\n  Tokenizing {split_name} ({len(raw_samples)} samples)...")
        samples = []
        n_extracted = 0
        for i in tqdm(range(len(raw_samples)), desc=f"  {split_name}"):
            row = raw_samples[i]
            ts = row.get("time_series", [])

            # Handle all the weird types: None, [], [[]], list of floats, list of lists
            has_ts = False
            if ts and isinstance(ts, list) and len(ts) > 0:
                if isinstance(ts[0], list) and len(ts[0]) > 0:
                    has_ts = True
                elif isinstance(ts[0], (int, float)):
                    has_ts = True

            pre_prompt = str(row.get("pre_prompt", ""))
            post_prompt = str(row.get("post_prompt", ""))

            # If no time_series field, try extracting from question text
            if not has_ts:
                extracted = _extract_ts_from_text(post_prompt)
                if extracted is None:
                    extracted = _extract_ts_from_text(pre_prompt)
                if extracted is not None:
                    ts = [extracted]  # single channel
                    has_ts = True
                    n_extracted += 1

            if has_ts:
                codes, mean, std = tokenize_signal(tokenizer, ts, max_ts_tokens, device)
                # Strip raw numbers from text — model gets TOTEM codes instead
                if codes:
                    pre_prompt = _strip_ts_from_text(pre_prompt)
                    post_prompt = _strip_ts_from_text(post_prompt)
            else:
                codes, mean, std = [], 0.0, 0.0

            samples.append({
                "pre_prompt": pre_prompt,
                "post_prompt": post_prompt,
                "answer": str(row.get("answer", "")),
                "ts_codes": codes,
                "ts_mean": mean,
                "ts_std": std,
                "task": str(row.get("task", "")),
                "domain": str(row.get("domain", "")),
                "source": str(row.get("source", "")),
            })

        result[split_name] = samples
        n_with_ts = sum(1 for s in samples if s["ts_codes"])
        print(f"    {len(samples)} samples ({n_with_ts} with TS codes, "
              f"{len(samples) - n_with_ts} text-only, "
              f"{n_extracted} extracted from text)")
    return result


def save_as_parquet(splits: dict[str, list[dict]], output_dir: str):
    """Save split data as train.parquet + validation.parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, data in splits.items():
        if not data:
            continue
        columns = {
            "time_series": [s.get("time_series", []) for s in data],
            "time_series_text": [s.get("time_series_text", []) for s in data],
            "pre_prompt": [s["pre_prompt"] for s in data],
            "post_prompt": [s["post_prompt"] for s in data],
            "answer": [s["answer"] for s in data],
            "ts_codes": [s["ts_codes"] for s in data],
            "ts_mean": [s["ts_mean"] for s in data],
            "ts_std": [s["ts_std"] for s in data],
            "task": [s.get("task", "") for s in data],
            "domain": [s.get("domain", "") for s in data],
        }
        table = pa.table(columns)
        path = output_dir / f"{split_name}.parquet"
        pq.write_table(table, path, compression="zstd")
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {split_name}: {len(data)} samples -> {path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Build pre-tokenized parquets for TEMPO training")
    parser.add_argument("--tokenizer", default="fsq_transformer_rope",
                        choices=["fsq_transformer_rope", "fsq_transformer", "fsq", "totem"])
    parser.add_argument("--tokenizer-ckpt", required=True,
                        help="Path to tokenizer checkpoint")
    parser.add_argument("--output-dir", default="data/pretokenized_rope",
                        help="Output directory for parquet files")
    parser.add_argument("--max-ts-tokens", type=int, default=750)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    # Phase 0 sizing
    parser.add_argument("--phase0-m4", type=int, default=10000)
    parser.add_argument("--phase0-sensor", type=int, default=20000)
    parser.add_argument("--phase0-opentslm", type=int, default=10000)
    parser.add_argument("--phase0-synthetic", type=int, default=10000)
    parser.add_argument("--phase0-ucr", type=int, default=500)
    # Phase 1 sizing
    parser.add_argument("--phase1-samples", type=int, default=300000)
    # Phase selection
    parser.add_argument("--only-phase0", action="store_true", help="Build only phase 0")
    parser.add_argument("--only-phase1", action="store_true", help="Build only phase 1")
    parser.add_argument("--phase0-version", type=int, default=1, choices=[1, 2, 3, 4, 6],
                        help="Phase 0 dataset version: 1=original, 2=discriminative, 3=+noise, 4=3+imputation, 6=4+captions(30%%)")
    parser.add_argument("--phase0-v2-samples", type=int, default=200000,
                        help="Total samples for phase0 v2/v3/v4")
    parser.add_argument("--noise-prob", type=float, default=0.5,
                        help="Probability of adding noise to each signal (v3/v4)")
    args = parser.parse_args()

    print("=" * 60)
    print("Building pre-tokenized training data")
    print(f"  Tokenizer: {args.tokenizer}")
    print(f"  Checkpoint: {args.tokenizer_ckpt}")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)

    tokenizer = load_tokenizer(args.tokenizer, args.tokenizer_ckpt, args.device)
    print(f"  Codebook: {tokenizer.codebook_size} codes\n")

    build_phase0 = not args.only_phase1
    build_phase1 = not args.only_phase0

    p0_total, p1_total = 0, 0

    if build_phase0:
        print(f"PHASE 0: ALIGNMENT DATA (v{args.phase0_version})")
        print("-" * 40)

        if args.phase0_version in (2, 3, 4, 6):
            from tempo.data.stage0_v2 import build_alignment_dataset_v2

            if args.phase0_version == 6:
                # v6: 50% discriminative+noise + 20% imputation + 30% M4 captions
                n_v3 = int(args.phase0_v2_samples * 0.50)
                n_v4 = int(args.phase0_v2_samples * 0.20)
                n_m4 = args.phase0_v2_samples - n_v3 - n_v4
            elif args.phase0_version == 4:
                # v4: 75% discriminative+noise (v3) + 25% imputation
                n_v3 = int(args.phase0_v2_samples * 0.75)
                n_v4 = args.phase0_v2_samples - n_v3
                n_m4 = 0
            else:
                n_v3 = args.phase0_v2_samples
                n_v4 = 0
                n_m4 = 0

            phase0_hf = build_alignment_dataset_v2(
                max_samples=n_v3,
                seed=args.seed,
                noise=(args.phase0_version >= 3),
                noise_prob=args.noise_prob,
            )

            if n_v4 > 0:
                from tempo.data.stage0_v4_imputation import build_imputation_dataset
                v4_hf = build_imputation_dataset(
                    tokenizer, max_samples=n_v4,
                    seed=args.seed + 100, device=args.device,
                )
                from datasets import concatenate_datasets
                phase0_hf = {
                    split: concatenate_datasets([phase0_hf[split], v4_hf[split]])
                    for split in phase0_hf
                    if split in v4_hf
                }

            if n_m4 > 0:
                from tempo.data.stage1 import load_m4_captions
                from datasets import Dataset, concatenate_datasets as concat_ds
                print(f"\n  Adding M4 captions ({n_m4} samples)...")
                m4_samples = load_m4_captions(max_samples=n_m4, seed=args.seed + 200)
                if m4_samples:
                    # Convert to HF Dataset format matching phase0_hf
                    m4_dicts = []
                    for s in m4_samples:
                        m4_dicts.append({
                            "time_series": s.get("time_series", []),
                            "time_series_text": s.get("time_series_text", []),
                            "pre_prompt": s.get("pre_prompt", ""),
                            "post_prompt": s.get("post_prompt", ""),
                            "answer": s.get("answer", ""),
                            "task": "caption",
                            "domain": s.get("domain", "forecasting"),
                        })
                    # 95/5 train/val split
                    n_val = max(1, len(m4_dicts) // 20)
                    m4_train = Dataset.from_list(m4_dicts[n_val:])
                    m4_val = Dataset.from_list(m4_dicts[:n_val])
                    phase0_hf = {
                        "train": concat_ds([phase0_hf["train"], m4_train]),
                        "validation": concat_ds([phase0_hf["validation"], m4_val]),
                    }
                    print(f"  M4 captions added: {len(m4_dicts)} samples")
        else:
            from tempo.data.stage1 import build_alignment_dataset
            phase0_hf = build_alignment_dataset(
                max_m4=args.phase0_m4,
                max_hf_sensor=args.phase0_sensor,
                max_opentslm=args.phase0_opentslm,
                max_synthetic=args.phase0_synthetic,
                max_ucr=args.phase0_ucr,
                seed=args.seed,
            )

        phase0_splits = pretokenize_hf_dataset(
            phase0_hf, tokenizer, args.max_ts_tokens, args.device,
        )
        phase0_dir = os.path.join(args.output_dir, "phase0")
        save_as_parquet(phase0_splits, phase0_dir)
        p0_total = sum(len(v) for v in phase0_splits.values())

    if build_phase1:
        print("\n\nPHASE 1: UNIFIED TRAINING DATA")
        print("-" * 40)
        # Collect raw samples directly (bypass HF Dataset Arrow conversion)
        from tempo.data.stage1_unified import (
            load_tsqa, load_engine_qa, load_opentslm_cot, load_text_instructions,
        )
        from tempo.data.stage1 import load_m4_captions
        import random as _random

        all_samples = []

        # ECG removed (not a primary benchmark, caused 30% imbalance).
        # Bearing uses vision-grounded CoT (from build_bearing_cot_vision.py).
        # HAR uses full dataset (primary benchmark).
        #
        # Expected mix:
        #   TSQA 139K + HAR 68K + Bearing 5K (vision) + Sleep 9K
        #   + PAMAP2 20K + Engine 13K + text 50K ≈ 304K
        tsqa_cap = 139000
        cot_cap = 999999      # uncapped — ECG removed, remaining sources are reasonable
        engine_cap = 20000
        text_cap = 50000

        # TSQA — multi-domain QA (largest, most diverse)
        all_samples.extend(load_tsqa(tsqa_cap, args.seed))

        # Engine QA with HDF5
        engine_h5 = None
        for p in ["data/engine_qa/time_series_data.h5",
                   os.path.join(os.path.expanduser("~"), "data", "engine_qa", "time_series_data.h5")]:
            if os.path.exists(p):
                engine_h5 = p
                break
        all_samples.extend(load_engine_qa(engine_cap, args.seed, h5_path=engine_h5))

        # OpenTSLM CoT — capped at 40K per source
        # Sleep has only ~9K (uses all), others capped from 68-159K to 40K
        all_samples.extend(load_opentslm_cot(cot_cap, args.seed))

        # Text instructions — fixed at 50K
        all_samples.extend(load_text_instructions(text_cap, args.seed))
        n_ts = sum(1 for s in all_samples if s.get("time_series"))
        n_text = len(all_samples) - n_ts
        pct_text = n_text / len(all_samples) * 100 if all_samples else 0
        print(f"  Text instructions: {n_text} samples ({pct_text:.0f}% of {len(all_samples)} total)")

        _random.Random(args.seed).shuffle(all_samples)
        n = len(all_samples)
        val_n = max(500, int(0.05 * n))

        print(f"\n  Phase 1 total: {n} samples (train={n-val_n}, val={val_n})")
        has_ts = sum(1 for s in all_samples if s.get("time_series"))
        print(f"  With TS: {has_ts}, Text-only: {n - has_ts}")

        # Tokenize directly from raw sample dicts
        phase1_splits = pretokenize_raw_samples(
            {"train": all_samples[val_n:], "validation": all_samples[:val_n]},
            tokenizer, args.max_ts_tokens, args.device,
        )
        phase1_dir = os.path.join(args.output_dir, "phase1")
        save_as_parquet(phase1_splits, phase1_dir)
        p1_total = sum(len(v) for v in phase1_splits.values())

    print("\n" + "=" * 60)
    print("DONE")
    if build_phase0:
        print(f"  Phase 0: {p0_total} samples -> {os.path.join(args.output_dir, 'phase0')}")
    if build_phase1:
        print(f"  Phase 1: {p1_total} samples -> {os.path.join(args.output_dir, 'phase1')}")

    bucket = os.environ.get("SAGEMAKER_BUCKET", "")
    print(f"\nUpload to S3:")
    if build_phase0:
        print(f"  aws s3 sync {phase0_dir} s3://{bucket}/tempo/phase0/")
    if build_phase1:
        print(f"  aws s3 sync {phase1_dir} s3://{bucket}/tempo/phase1/")
    print(f"  aws s3 cp {args.tokenizer_ckpt} s3://{bucket}/tempo/tokenizer/")


if __name__ == "__main__":
    main()
