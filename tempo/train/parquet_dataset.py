"""Parquet dataset loader for pre-tokenized time series data.

Loads parquet files with ts_codes column directly into PyTorch Datasets
compatible with TEMPO's compute_loss() or compute_loss_pretokenized().

When a TEMPO model is passed, text is tokenized once at load time and
cached — eliminating repeated tokenizer calls during training.

Usage:
    from tempo.train.parquet_dataset import load_parquet_splits

    # Standard (tokenize text every batch — slow)
    splits = load_parquet_splits("data/pretokenized/stage1_base")

    # Pre-tokenized (tokenize text once at load — fast)
    splits = load_parquet_splits("data/pretokenized/stage1_base", model=tempo_model)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset, Sampler


class ParquetMapDataset(Dataset):
    """Map-style PyTorch Dataset backed by a parquet file.

    Reads the full parquet into memory (fine for <500K rows / ~2GB).
    Returns dicts compatible with TEMPO.compute_loss().
    """

    def __init__(self, parquet_path: str, eos_token: str = ""):
        table = pq.read_table(parquet_path)
        self.data = table.to_pydict()
        self.n = table.num_rows
        self.eos_token = eos_token
        self.columns = table.column_names

        # Precompute ts_mean/ts_std if not in parquet
        if "ts_mean" not in self.columns:
            self.data["ts_mean"] = [0.0] * self.n
            self.data["ts_std"] = [0.0] * self.n
            if "time_series" in self.columns:
                for i in range(self.n):
                    ts = self.data["time_series"][i]
                    if ts and len(ts) > 0 and len(ts[0]) > 0:
                        vals = np.array(ts[0], dtype=np.float32)
                        self.data["ts_mean"][i] = float(vals.mean())
                        self.data["ts_std"][i] = float(vals.std())

        # Pre-tokenized cache (populated by pretokenize_text)
        self._pretokenized = False
        self._input_ids: list[list[int]] = []
        self._prompt_lengths: list[int] = []
        self._seq_lengths: list[int] = []

    def pretokenize_text(self, model) -> None:
        """Tokenize all text once using the TEMPO model's tokenizer.

        After calling this, __getitem__ returns 'input_ids' and
        'prompt_length' keys, and trainer should use compute_loss_pretokenized().
        """
        print(f"  Pre-tokenizing {self.n:,} samples...", end=" ", flush=True)
        tokenizer = model.tokenizer

        for i in range(self.n):
            sample = {col: self.data[col][i] for col in self.columns}
            # Flatten ts_codes
            ts_codes = sample.get("ts_codes")
            if ts_codes and isinstance(ts_codes, list) and len(ts_codes) > 0:
                if isinstance(ts_codes[0], list):
                    sample["ts_codes"] = [c for ch in ts_codes for c in ch]

            prompt = model.build_text(sample)
            answer = sample.get("answer", "")

            # Tokenize prompt and answer separately to get exact boundary,
            # then concatenate (avoids tokenizing the prompt twice)
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            answer_ids = tokenizer(answer + "<|im_end|>", add_special_tokens=False).input_ids
            full_ids = (prompt_ids + answer_ids)[:model.config.max_length]

            self._input_ids.append(full_ids)
            self._prompt_lengths.append(len(prompt_ids))
            self._seq_lengths.append(len(full_ids))

        self._pretokenized = True
        avg_len = np.mean(self._seq_lengths)
        max_len = max(self._seq_lengths)
        print(f"done (avg={avg_len:.0f}, max={max_len} tokens)")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self._pretokenized:
            return {
                "input_ids": self._input_ids[idx],
                "prompt_length": self._prompt_lengths[idx],
            }

        sample = {col: self.data[col][idx] for col in self.columns}

        # Flatten ts_codes for single-channel (most common case)
        ts_codes = sample.get("ts_codes")
        if ts_codes and isinstance(ts_codes, list) and len(ts_codes) > 0:
            if isinstance(ts_codes[0], list):
                sample["ts_codes"] = [c for ch in ts_codes for c in ch]

        # Add EOS to answer
        answer = sample.get("answer", "")
        if answer and self.eos_token and not answer.endswith(self.eos_token):
            sample["answer"] = answer

        return sample


class LengthGroupedSampler(Sampler):
    """Sampler that groups sequences of similar length into batches.

    Reduces padding waste by ~50-70% compared to random batching.
    Sorts by length, then shuffles within buckets of size bucket_size.
    """

    def __init__(self, lengths: list[int], batch_size: int,
                 shuffle: bool = True, seed: int = 42):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        # Sort indices by length
        indices = list(range(len(self.lengths)))
        indices.sort(key=lambda i: self.lengths[i])

        # Group into mega-batches, shuffle within each
        rng = np.random.RandomState(self.seed + self.epoch)
        bucket_size = self.batch_size * 100  # shuffle within ~100 batches
        buckets = [indices[i:i + bucket_size] for i in range(0, len(indices), bucket_size)]

        if self.shuffle:
            for bucket in buckets:
                rng.shuffle(bucket)
            rng.shuffle(buckets)

        for bucket in buckets:
            yield from bucket

    def __len__(self):
        return len(self.lengths)

    def set_epoch(self, epoch: int):
        self.epoch = epoch


def load_parquet_splits(
    data_dir: str,
    eos_token: str = "",
    model=None,
) -> dict[str, ParquetMapDataset]:
    """Load train/validation parquet files from a directory.

    Args:
        data_dir: Directory containing train.parquet and validation.parquet.
        eos_token: EOS token to append to answers.
        model: If provided, pre-tokenize all text at load time for fast training.

    Returns dict with "train" and "validation" keys.
    """
    data_dir = Path(data_dir)
    splits = {}

    for split in ["train", "validation"]:
        path = data_dir / f"{split}.parquet"
        if path.exists():
            ds = ParquetMapDataset(str(path), eos_token=eos_token)
            if model is not None:
                ds.pretokenize_text(model)
            splits[split] = ds
            print(f"  Loaded {split}: {len(ds):,} samples from {path}")
        else:
            raise FileNotFoundError(f"Missing {path}")

    return splits
