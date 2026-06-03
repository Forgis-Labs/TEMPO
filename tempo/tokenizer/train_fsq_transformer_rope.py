"""Train FSQ Transformer tokenizer with RoPE positional encoding.

Supports variable-length training: each batch samples a random window length
from [128, 2048], so the RoPE positions are calibrated across all lengths
the model will see at inference.

Usage:
    python -m tempo.tokenizer.train_fsq_transformer_rope
    python -m tempo.tokenizer.train_fsq_transformer_rope --levels 5 5 5 5 --epochs 100
    python -m tempo.tokenizer.train_fsq_transformer_rope --min-length 128 --max-length 2048
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .fsq_transformer import FSQTransformerConfig
from .fsq_transformer_rope import FSQTransformerRoPETokenizer


# Losses

def spectral_loss(x_recon, x, n_fft=64):
    """Multi-scale spectral loss."""
    loss = 0.0
    for fft_size in [n_fft, n_fft * 2, n_fft * 4]:
        if fft_size > x.shape[-1]:
            continue
        window = torch.hann_window(fft_size, device=x.device)
        hop = fft_size // 4
        x_stft = torch.stft(x, fft_size, hop_length=hop, window=window,
                            return_complex=True, onesided=True)
        r_stft = torch.stft(x_recon, fft_size, hop_length=hop, window=window,
                            return_complex=True, onesided=True)
        loss += F.l1_loss(torch.log1p(r_stft.abs()), torch.log1p(x_stft.abs()))
    return loss


# Variable-Length Dataset

class VariableLengthTSDataset(Dataset):
    """Time series dataset that stores raw signals of varying lengths.

    Each __getitem__ returns a z-normalized signal. The DataLoader collates
    batches of the same length via the custom collate_fn.
    """

    def __init__(self, signals: list[np.ndarray]):
        self.signals = signals

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        x = np.asarray(self.signals[idx], dtype=np.float32).flatten()
        std = x.std()
        if std > 1e-8:
            x = (x - x.mean()) / std
        return torch.tensor(x, dtype=torch.float32)


def variable_length_collate(batch, min_length, max_length, patch_size, rng):
    """Collate by picking a random window length for this batch.

    All signals in the batch are cropped/resampled to the same length
    (required for batched Transformer forward pass).
    """
    # Sample a window length (multiple of patch_size)
    n_patches_min = min_length // patch_size
    n_patches_max = max_length // patch_size
    n_patches = rng.integers(n_patches_min, n_patches_max + 1)
    target_len = n_patches * patch_size

    result = []
    for x in batch:
        n = len(x)
        if n == target_len:
            result.append(x)
        elif n > target_len:
            # Random crop
            start = rng.integers(0, n - target_len + 1)
            result.append(x[start:start + target_len])
        else:
            # Resample up via linear interpolation
            x_np = x.numpy().flatten()
            x_resampled = np.interp(
                np.linspace(0, len(x_np) - 1, target_len),
                np.arange(len(x_np)), x_np,
            )
            result.append(torch.tensor(x_resampled, dtype=torch.float32))

    return torch.stack(result)


# Data Collection

def collect_signals(max_total=500000, min_length=32, max_length=4096):
    """Collect diverse time series at their NATIVE lengths.

    Unlike train_fsq_tokenizer.py (which resamples everything to 256),
    this keeps signals at their original length for variable-length training.
    Signals shorter than min_length or longer than max_length are skipped.
    """
    from collections import defaultdict
    import warnings
    warnings.filterwarnings("ignore")

    domain_signals = defaultdict(list)
    max_per_domain = max_total // 7

    def _add_windows(signal, domain, stride=None):
        """Extract windows of varying lengths from a long signal."""
        n = len(signal)
        if n < min_length:
            return
        if n <= max_length:
            if signal.std() > 1e-12:
                domain_signals[domain].append(signal)
            return
        # Long signal: extract windows at multiple scales
        for win_len in [256, 512, 1024, 2048]:
            if win_len > n:
                continue
            s = stride or win_len // 2
            for start in range(0, n - win_len + 1, s):
                w = signal[start:start + win_len]
                if w.std() > 1e-12:
                    domain_signals[domain].append(w)
                if len(domain_signals[domain]) >= max_per_domain:
                    return

    # 1. Forecasting
    print("[1/6] Forecasting datasets...")
    try:
        from datasets import load_dataset
        for config in ["ETTh1", "ETTh2", "ETTm1", "weather"]:
            if len(domain_signals["forecast"]) >= max_per_domain:
                break
            try:
                ds = load_dataset("thuml/Time-Series-Library", config, split="train")
                cols = [c for c in ds.column_names if c != "date"][:20]
                for col in cols:
                    vals = np.array(ds[col], dtype=np.float64)
                    _add_windows(vals, "forecast")
                    if len(domain_signals["forecast"]) >= max_per_domain:
                        break
            except Exception:
                pass
        print(f"  Forecast: {len(domain_signals['forecast'])} windows")
    except ImportError:
        print("  SKIP (datasets not installed)")

    # 2. UCR (native lengths — many are NOT 256)
    print("[2/6] UCR archive...")
    try:
        from aeon.datasets import load_classification
        from aeon.datasets.tsc_datasets import univariate_equal_length
        for name in sorted(univariate_equal_length):
            if len(domain_signals["ucr"]) >= max_per_domain:
                break
            try:
                X, _ = load_classification(name, split="train")
                for i in range(min(len(X), 100)):
                    sig = X[i][0].astype(np.float64)
                    if len(sig) >= min_length and sig.std() > 1e-12:
                        if len(sig) <= max_length:
                            # Keep at native length (might be 128, 176, 256, 512, etc.)
                            domain_signals["ucr"].append(sig)
                        else:
                            _add_windows(sig, "ucr")
            except Exception:
                pass
        print(f"  UCR: {len(domain_signals['ucr'])} windows")
    except ImportError:
        print("  SKIP (aeon not installed)")

    # 3. ECG (MIT-BIH — native 360Hz, windows at multiple scales)
    print("[3/6] ECG...")
    try:
        import wfdb
        records = ["100", "101", "102", "103", "104", "105", "106", "107"]
        for rec_id in records:
            if len(domain_signals["ecg"]) >= max_per_domain:
                break
            try:
                record = wfdb.rdrecord(rec_id, pn_dir="mitdb")
                for ch in range(record.p_signal.shape[1]):
                    sig = record.p_signal[:, ch].astype(np.float64)
                    sig = sig[~np.isnan(sig)]
                    _add_windows(sig, "ecg")
            except Exception:
                pass
        print(f"  ECG: {len(domain_signals['ecg'])} windows")
    except ImportError:
        print("  SKIP (wfdb not installed)")

    # 4. Bearing vibration (CWRU — raw 12kHz, no resampling)
    print("[4/9] Bearing vibration (CWRU)...")
    try:
        import glob
        from scipy.io import loadmat

        cwru_dir = "data/cwru_github"
        mat_files = sorted(glob.glob(os.path.join(cwru_dir, "**", "*.mat"), recursive=True))
        if not mat_files:
            print(f"  CWRU: no .mat files in {cwru_dir}")
        else:
            for fpath in mat_files:
                if len(domain_signals["vibration"]) >= max_per_domain:
                    break
                try:
                    mat = loadmat(fpath)
                    for key in mat:
                        if 'DE_time' in key:
                            sig = mat[key].flatten().astype(np.float64)
                            # Raw 12kHz — no resampling, preserve fault harmonics
                            _add_windows(sig, "vibration")
                            break
                except Exception:
                    pass
            print(f"  CWRU: {len(domain_signals['vibration'])} windows from {len(mat_files)} files")
    except Exception as e:
        print(f"  CWRU: SKIP ({e})")

    print("[5/9] MFPT bearing...")
    try:
        mfpt_path = "data/mfpt"
        if os.path.exists(mfpt_path):
            import glob
            from scipy.io import loadmat
            for f in sorted(glob.glob(os.path.join(mfpt_path, "**", "*.mat"), recursive=True)):
                if len(domain_signals["vibration"]) >= max_per_domain:
                    break
                try:
                    mat = loadmat(f, struct_as_record=False, squeeze_me=True)
                    sig = np.array(mat["bearing"].gs).flatten().astype(np.float64)
                    _add_windows(sig, "vibration")
                except Exception:
                    pass
            print(f"  MFPT: {len(domain_signals['vibration'])} vibration total")
        else:
            print("  MFPT: not available (skipping)")
    except Exception as e:
        print(f"  MFPT: SKIP ({e})")

    # 6. Synthetic (mixed lengths)
    print("[6/9] Synthetic (mixed lengths)...")
    rng = np.random.RandomState(42)
    possible_lengths = [128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
    for _ in range(max_per_domain):
        length = rng.choice(possible_lengths)
        kind = rng.choice(["sine", "chirp", "ar", "noise", "damped",
                           "modulated", "step", "random_walk"])
        t = np.linspace(0, 1, length)
        if kind == "sine":
            sig = np.sin(2 * np.pi * rng.uniform(1, 20) * t + rng.uniform(0, 6.28))
            sig += rng.randn(length) * 0.1
        elif kind == "chirp":
            f0, f1 = rng.uniform(1, 5), rng.uniform(10, 30)
            sig = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / 2) * t)
        elif kind == "ar":
            sig = np.zeros(length)
            a = rng.uniform(0.5, 0.99)
            for i in range(1, length):
                sig[i] = a * sig[i - 1] + rng.randn() * 0.3
        elif kind == "noise":
            sig = rng.randn(length)
        elif kind == "damped":
            sig = np.exp(-rng.uniform(1, 5) * t) * np.sin(2 * np.pi * rng.uniform(3, 15) * t)
        elif kind == "modulated":
            fc, fm = rng.uniform(5, 20), rng.uniform(0.5, 3)
            sig = np.sin(2 * np.pi * fc * t) * (1 + 0.5 * np.sin(2 * np.pi * fm * t))
        elif kind == "step":
            pos = rng.randint(length // 4, 3 * length // 4)
            sig = np.zeros(length)
            sig[pos:] = rng.uniform(1, 5)
            sig += rng.randn(length) * 0.1
        elif kind == "random_walk":
            sig = np.cumsum(rng.randn(length) * 0.1)
        if np.std(sig) > 1e-12:
            domain_signals["synthetic"].append(sig.astype(np.float64))
    print(f"  Synthetic: {len(domain_signals['synthetic'])} windows")

    # 7. UEA Multivariate
    print("[7/9] UEA Multivariate...")
    try:
        from aeon.datasets import load_classification
        from aeon.datasets.tsc_datasets import multivariate_equal_length as UEA
        good_uea = ["MotorImagery", "Heartbeat", "SelfRegulationSCP1",
                     "FingerMovements", "UWaveGestureLibrary"]
        for name in good_uea:
            if len(domain_signals["uea"]) >= max_per_domain:
                break
            try:
                X, _ = load_classification(name, split="train")
                n_samples, n_ch, length = X.shape
                for i in range(min(n_samples, 100)):
                    for ch in range(min(n_ch, 5)):
                        sig = X[i, ch, :].astype(np.float64)
                        if len(sig) >= min_length and sig.std() > 1e-12:
                            if len(sig) <= max_length:
                                domain_signals["uea"].append(sig)
                            else:
                                _add_windows(sig, "uea")
            except Exception:
                pass
        print(f"  UEA: {len(domain_signals['uea'])} windows")
    except ImportError:
        print("  SKIP")

    # 8. PAMAP2 IMU
    print("[8/9] PAMAP2 IMU...")
    try:
        import glob
        pamap_path = "data/PAMAP2_Dataset/Protocol"
        if os.path.exists(pamap_path):
            import pandas as pd
            for f in sorted(glob.glob(os.path.join(pamap_path, "subject*.dat"))):
                if len(domain_signals["imu"]) >= max_per_domain:
                    break
                try:
                    df = pd.read_csv(f, sep=" ", header=None)
                    for col_start in [4, 21, 38]:
                        for offset in range(3):
                            vals = df.iloc[:, col_start + offset].dropna().values.astype(np.float64)
                            vals = vals[~np.isnan(vals)]
                            _add_windows(vals, "imu")
                except Exception:
                    pass
            print(f"  PAMAP2: {len(domain_signals['imu'])} IMU windows")
        else:
            print("  PAMAP2: not available (skipping)")
    except Exception as e:
        print(f"  PAMAP2: SKIP ({e})")

    # 9. Financial
    print("[9/9] Financial...")
    try:
        import yfinance as yf
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "BTC-USD"]
        for ticker in tickers:
            if len(domain_signals["financial"]) >= max_per_domain:
                break
            try:
                data = yf.download(ticker, period="max", interval="1d", progress=False)
                for col in ["Close"]:
                    if col in data.columns:
                        vals = data[col].dropna().values.astype(np.float64)
                        _add_windows(vals, "financial")
            except Exception:
                pass
        print(f"  Financial: {len(domain_signals['financial'])} windows")
    except ImportError:
        print("  SKIP")

    # Combine
    print("\nDomain summary:")
    all_signals = []
    for domain, sigs in sorted(domain_signals.items()):
        cap = min(len(sigs), max_per_domain)
        selected = sigs[:cap]
        all_signals.extend(selected)
        lens = [len(s) for s in selected]
        if lens:
            print(f"  {domain:12s}: {len(selected):6d} windows, "
                  f"lengths=[{min(lens)}, {int(np.median(lens))}, {max(lens)}]")

    np.random.seed(42)
    np.random.shuffle(all_signals)
    if len(all_signals) > max_total:
        all_signals = all_signals[:max_total]

    lengths = [len(s) for s in all_signals]
    print(f"\nTotal: {len(all_signals)} windows, "
          f"lengths=[{min(lengths)}, {int(np.median(lengths))}, {max(lengths)}]")
    return all_signals


def _collect_mixed_length(max_total=50000):
    """Fallback: synthetic signals at varying lengths."""
    rng = np.random.RandomState(42)
    signals = []
    # Deliberately vary signal lengths
    possible_lengths = [128, 192, 256, 384, 512, 768, 1024, 1536, 2048]

    for _ in range(max_total):
        length = rng.choice(possible_lengths)
        kind = rng.choice(["sine", "chirp", "ar", "noise", "damped",
                           "modulated", "step", "random_walk"])
        t = np.linspace(0, 1, length)

        if kind == "sine":
            sig = np.sin(2 * np.pi * rng.uniform(1, 20) * t + rng.uniform(0, 6.28))
            sig += rng.randn(length) * 0.1
        elif kind == "chirp":
            f0, f1 = rng.uniform(1, 5), rng.uniform(10, 30)
            sig = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / 2) * t)
        elif kind == "ar":
            sig = np.zeros(length)
            a = rng.uniform(0.5, 0.99)
            for i in range(1, length):
                sig[i] = a * sig[i - 1] + rng.randn() * 0.3
        elif kind == "noise":
            sig = rng.randn(length)
        elif kind == "damped":
            sig = np.exp(-rng.uniform(1, 5) * t) * np.sin(2 * np.pi * rng.uniform(3, 15) * t)
        elif kind == "modulated":
            fc, fm = rng.uniform(5, 20), rng.uniform(0.5, 3)
            sig = np.sin(2 * np.pi * fc * t) * (1 + 0.5 * np.sin(2 * np.pi * fm * t))
        elif kind == "step":
            pos = rng.randint(length // 4, 3 * length // 4)
            sig = np.zeros(length)
            sig[pos:] = rng.uniform(1, 5)
            sig += rng.randn(length) * 0.1
        elif kind == "random_walk":
            sig = np.cumsum(rng.randn(length) * 0.1)

        if np.std(sig) > 1e-12:
            signals.append(sig.astype(np.float64))

    print(f"Mixed-length synthetic: {len(signals)} signals")
    lengths = [len(s) for s in signals]
    print(f"  Length range: {min(lengths)}-{max(lengths)}, "
          f"median={sorted(lengths)[len(lengths)//2]}")
    return signals


# Training

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = FSQTransformerConfig(
        levels=args.levels,
        patch_size=args.patch_size,
        input_length=256,  # nominal, not a hard limit with RoPE
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_layers,
        n_decoder_layers=args.n_layers,
        d_ff=args.d_model * 4,
        dropout=args.dropout,
        temporal_weight=args.temporal_weight,
    )

    print("=" * 60)
    print(f"FSQ Transformer Tokenizer with RoPE")
    print(f"  Codes: {config.codebook_size} (levels={config.levels})")
    print(f"  Patch size: {config.patch_size}")
    print(f"  Transformer: d={config.d_model}, heads={config.n_heads}, "
          f"layers={config.n_encoder_layers}+{config.n_decoder_layers}")
    print(f"  Training lengths: [{args.min_length}, {args.max_length}] "
          f"(variable per batch)")
    print(f"  Temporal weight: {config.temporal_weight}")
    print("=" * 60)

    # Data
    signals = collect_signals(args.max_signals)

    # Ensure all signals have at least min_length points
    # (shorter ones get filtered out or resampled during collation)
    print(f"Total signals: {len(signals)}")
    lengths = np.array([len(s) for s in signals])
    print(f"  Length distribution: min={lengths.min()}, median={int(np.median(lengths))}, "
          f"max={lengths.max()}")
    print(f"  Signals >= {args.min_length}: {(lengths >= args.min_length).sum()}")
    print(f"  Signals >= {args.max_length}: {(lengths >= args.max_length).sum()}")

    n_val = max(500, len(signals) // 10)
    train_ds = VariableLengthTSDataset(signals[n_val:])
    val_ds = VariableLengthTSDataset(signals[:n_val])

    batch_rng = np.random.default_rng(args.seed)
    val_rng = np.random.default_rng(args.seed + 1)

    def train_collate(batch):
        return variable_length_collate(
            batch, args.min_length, args.max_length, config.patch_size, batch_rng)

    def val_collate(batch):
        # Validate at fixed 256 length for consistent metrics
        return variable_length_collate(
            batch, 256, 256, config.patch_size, val_rng)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=train_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=val_collate)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Model
    model = FSQTransformerRoPETokenizer(config).to(device)
    summary = model.trainable_summary()
    print(f"Parameters: {summary['total']:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_val = float("inf")
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        tl, tl_mse, tl_spec, tl_temp = 0, 0, 0, 0
        batch_lengths = []
        for batch in train_loader:
            x = batch.to(device)
            batch_lengths.append(x.shape[1])
            recon, indices, z_pre = model(x)

            mse = F.mse_loss(recon, x)
            spec = spectral_loss(recon, x)
            temp = model.temporal_smoothness_loss(z_pre)
            loss = mse + args.spectral_weight * spec + config.temporal_weight * temp

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tl += loss.item()
            tl_mse += mse.item()
            tl_spec += spec.item()
            tl_temp += temp.item()

        n_batches = len(train_loader)
        tl /= n_batches
        tl_mse /= n_batches
        tl_spec /= n_batches
        tl_temp /= n_batches

        # Length distribution this epoch
        bl = np.array(batch_lengths)
        len_str = f"lengths=[{bl.min()},{int(np.median(bl))},{bl.max()}]"

        # Validate (at fixed 256 length)
        model.eval()
        vl, vl_mse, unique_codes = 0, 0, set()
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                recon, indices, z_pre = model(x)
                mse_v = F.mse_loss(recon, x)
                spec_v = spectral_loss(recon, x)
                temp_v = model.temporal_smoothness_loss(z_pre)
                vl += (mse_v + args.spectral_weight * spec_v + config.temporal_weight * temp_v).item()
                vl_mse += mse_v.item()

                flat = model.fsq.codes_to_flat(indices)
                unique_codes.update(flat.cpu().numpy().flatten().tolist())

        vl /= len(val_loader)
        vl_mse /= len(val_loader)

        scheduler.step()

        print(f"  Epoch {epoch:3d}: loss={tl:.4f} (mse={tl_mse:.4f} spec={tl_spec:.4f} "
              f"temp={tl_temp:.4f}) val={vl:.4f} codes={len(unique_codes):,}/{config.codebook_size:,} "
              f"{len_str}")
        sys.stdout.flush()

        # Save best
        if vl < best_val - 1e-6:
            best_val = vl
            patience_left = args.patience
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "levels": config.levels,
                    "patch_size": config.patch_size,
                    "input_length": 256,  # nominal
                    "compression": config.patch_size,
                    "d_model": config.d_model,
                    "n_heads": config.n_heads,
                    "n_encoder_layers": config.n_encoder_layers,
                    "n_decoder_layers": config.n_decoder_layers,
                    "d_ff": config.d_ff,
                    "dropout": config.dropout,
                    "temporal_weight": config.temporal_weight,
                    "positional_encoding": "rope",
                    "train_min_length": args.min_length,
                    "train_max_length": args.max_length,
                },
                "epoch": epoch,
                "val_loss": vl,
                "unique_codes": len(unique_codes),
            }, args.output)
            print(f"    -> Saved best (codes={len(unique_codes)})")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"Done. Best val={best_val:.4f}")

    # Final test: tokenize at various lengths
    print("\nVariable-length tokenization test:")
    model.eval()
    for length in [128, 256, 512, 1024, 2048]:
        x = torch.randn(1, length).to(device)
        with torch.no_grad():
            codes = model.tokenize(x)
        print(f"  {length:5d} pts -> {codes.shape[1]:4d} codes")


def main():
    p = argparse.ArgumentParser(description="Train FSQ Transformer tokenizer with RoPE")
    p.add_argument("--levels", type=int, nargs="+", default=[5, 5, 5, 5])
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--temporal_weight", type=float, default=0.1)
    p.add_argument("--spectral_weight", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--max_signals", type=int, default=500000)
    p.add_argument("--min_length", type=int, default=128,
                   help="Min window length during training")
    p.add_argument("--max_length", type=int, default=2048,
                   help="Max window length during training")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str,
                   default="checkpoints/tokenizers/fsq_transformer_rope_625_best.pt")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
