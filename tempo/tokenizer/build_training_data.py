"""Build tokenizer training data as a single parquet file.

Collects signals from all 9 domains at native lengths (no resampling),
saves as parquet. Upload to S3 once, use forever.

Usage:
    python -m tempo.tokenizer.build_training_data --output data/tokenizer_training.parquet
    aws s3 cp data/tokenizer_training.parquet s3://<your-s3-bucket>/tempo/data/
"""

import os
import sys
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _add_windows(signal, domain, domain_signals, max_per_domain,
                 min_length=32, max_length=4096):
    """Extract windows of varying lengths from a long signal."""
    n = len(signal)
    if n < min_length:
        return
    if signal.std() < 1e-12:
        return
    if n <= max_length:
        domain_signals[domain].append(signal)
        return
    for win_len in [256, 512, 1024, 2048]:
        if win_len > n:
            continue
        stride = win_len // 2
        for start in range(0, n - win_len + 1, stride):
            w = signal[start:start + win_len]
            if w.std() > 1e-12:
                domain_signals[domain].append(w)
            if len(domain_signals[domain]) >= max_per_domain:
                return


def collect_all(max_per_domain=100000, min_length=128, max_length=4096,
                data_root="data"):
    """Collect signals from all 9 domains at native lengths.

    Args:
        data_root: Root directory for local datasets (cwru_github, PAMAP2, mfpt).
                   Pass the path where your data lives relative to cwd.
    """
    import warnings
    warnings.filterwarnings("ignore")

    domain_signals = defaultdict(list)

    # 1. Forecasting (ETT, weather)
    print("[1/9] Forecasting...")
    try:
        from datasets import load_dataset
        for config in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "traffic"]:
            if len(domain_signals["forecast"]) >= max_per_domain:
                break
            try:
                ds = load_dataset("thuml/Time-Series-Library", config, split="train")
                cols = [c for c in ds.column_names if c != "date"][:50]
                for col in cols:
                    vals = np.array(ds[col], dtype=np.float64)
                    _add_windows(vals, "forecast", domain_signals, max_per_domain,
                                 min_length, max_length)
                    if len(domain_signals["forecast"]) >= max_per_domain:
                        break
                print(f"  {config}: {len(domain_signals['forecast'])} total")
            except Exception as e:
                print(f"  {config}: SKIP ({e})")
    except ImportError:
        print("  SKIP (datasets not installed)")

    # 2. UCR (native lengths)
    print("[2/9] UCR archive...")
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
                            domain_signals["ucr"].append(sig)
                        else:
                            _add_windows(sig, "ucr", domain_signals, max_per_domain,
                                         min_length, max_length)
            except Exception:
                pass
        print(f"  UCR: {len(domain_signals['ucr'])} windows")
    except ImportError:
        print("  SKIP (aeon not installed)")

    # 3. CWRU bearing (raw 12kHz, no resampling)
    print("[3/9] CWRU bearing...")
    try:
        import glob
        from scipy.io import loadmat
        cwru_dir = os.path.join(data_root, "cwru_github")
        mat_files = sorted(glob.glob(os.path.join(cwru_dir, "**", "*.mat"), recursive=True))
        for fpath in mat_files:
            if len(domain_signals["vibration"]) >= max_per_domain:
                break
            try:
                mat = loadmat(fpath)
                for key in mat:
                    if 'DE_time' in key:
                        sig = mat[key].flatten().astype(np.float64)
                        _add_windows(sig, "vibration", domain_signals, max_per_domain,
                                     min_length, max_length)
                        break
            except Exception:
                pass
        print(f"  CWRU: {len(domain_signals['vibration'])} windows from {len(mat_files)} files")
    except Exception as e:
        print(f"  CWRU: SKIP ({e})")

    # 4. Synthetic (mixed lengths)
    print("[4/8] Synthetic...")
    rng = np.random.RandomState(42)
    possible_lengths = [128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
    for _ in range(max_per_domain):
        length = rng.choice(possible_lengths)
        kind = rng.choice(["sine", "chirp", "ar", "noise", "damped",
                           "modulated", "step", "random_walk", "sawtooth", "pulse"])
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
        elif kind == "sawtooth":
            freq = rng.uniform(2, 10)
            sig = 2 * (t * freq - np.floor(t * freq + 0.5))
        elif kind == "pulse":
            sig = rng.randn(length) * 0.1
            for _ in range(rng.randint(1, 5)):
                pos = rng.randint(0, length)
                sig[pos] = rng.uniform(3, 10) * rng.choice([-1, 1])
        if np.std(sig) > 1e-12:
            domain_signals["synthetic"].append(sig.astype(np.float64))
    print(f"  Synthetic: {len(domain_signals['synthetic'])} windows")

    # 5. ECG (MIT-BIH)
    print("[5/8] ECG...")
    try:
        import wfdb
        records = [str(i) for i in [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                                     111, 112, 113, 114, 115, 116, 117, 118, 119, 121,
                                     122, 123, 124, 200, 201, 202, 203, 205, 207, 208]]
        for rec_id in records:
            if len(domain_signals["ecg"]) >= max_per_domain:
                break
            try:
                record = wfdb.rdrecord(rec_id, pn_dir="mitdb")
                for ch in range(record.p_signal.shape[1]):
                    sig = record.p_signal[:, ch].astype(np.float64)
                    sig = sig[~np.isnan(sig)]
                    _add_windows(sig, "ecg", domain_signals, max_per_domain,
                                 min_length, max_length)
            except Exception:
                pass
        print(f"  ECG: {len(domain_signals['ecg'])} windows")
    except ImportError:
        print("  SKIP (wfdb not installed)")

    # 6. UEA Multivariate
    print("[6/8] UEA Multivariate...")
    try:
        from aeon.datasets import load_classification
        from aeon.datasets.tsc_datasets import multivariate_equal_length as UEA
        good_uea = ["MotorImagery", "Heartbeat", "SelfRegulationSCP1",
                     "SelfRegulationSCP2", "FingerMovements", "UWaveGestureLibrary"]
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
                                _add_windows(sig, "uea", domain_signals, max_per_domain,
                                             min_length, max_length)
            except Exception:
                pass
        print(f"  UEA: {len(domain_signals['uea'])} windows")
    except ImportError:
        print("  SKIP")

    # 7. PAMAP2 IMU
    print("[7/8] PAMAP2 IMU...")
    try:
        import glob
        import pandas as pd
        pamap_path = os.path.join(data_root, "PAMAP2_Dataset", "Protocol")
        if os.path.exists(pamap_path):
            for f in sorted(glob.glob(os.path.join(pamap_path, "subject*.dat"))):
                if len(domain_signals["imu"]) >= max_per_domain:
                    break
                try:
                    df = pd.read_csv(f, sep=" ", header=None)
                    for col_start in [4, 21, 38]:
                        for offset in range(3):
                            vals = df.iloc[:, col_start + offset].dropna().values.astype(np.float64)
                            vals = vals[~np.isnan(vals)]
                            _add_windows(vals, "imu", domain_signals, max_per_domain,
                                         min_length, max_length)
                except Exception:
                    pass
            print(f"  PAMAP2: {len(domain_signals['imu'])} windows")
        else:
            print("  PAMAP2: not available")
    except Exception as e:
        print(f"  PAMAP2: SKIP ({e})")

    # 8. Financial
    print("[8/8] Financial...")
    try:
        import yfinance as yf
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "SPY",
                   "BTC-USD", "ETH-USD", "GLD", "QQQ"]
        for ticker in tickers:
            if len(domain_signals["financial"]) >= max_per_domain:
                break
            try:
                data = yf.download(ticker, period="max", interval="1d", progress=False)
                for col in ["Close"]:
                    if col in data.columns:
                        vals = data[col].dropna().values.astype(np.float64).flatten()
                        _add_windows(vals, "financial", domain_signals, max_per_domain,
                                     min_length, max_length)
            except Exception:
                pass
        print(f"  Financial: {len(domain_signals['financial'])} windows")
    except ImportError:
        print("  SKIP (yfinance not installed)")

    return domain_signals


def save_parquet(domain_signals, output_path, max_total=500000, seed=42):
    """Save all signals as a single parquet file."""
    rng = np.random.RandomState(seed)

    # Balance domains
    max_per = max_total // max(len(domain_signals), 1)
    all_signals = []
    all_domains = []
    all_lengths = []

    print(f"\nDomain summary:")
    for domain in sorted(domain_signals.keys()):
        sigs = domain_signals[domain]
        cap = min(len(sigs), max_per)
        selected = sigs[:cap]
        if not selected:
            print(f"  {domain:12s}:      0 windows (skipped)")
            continue
        lens = [len(s) for s in selected]
        print(f"  {domain:12s}: {len(selected):6d} windows, "
              f"lengths=[{min(lens)}, {int(np.median(lens))}, {max(lens)}]")
        for s in selected:
            all_signals.append(s.tolist())
            all_domains.append(domain)
            all_lengths.append(len(s))

    # Shuffle
    indices = list(range(len(all_signals)))
    rng.shuffle(indices)
    all_signals = [all_signals[i] for i in indices]
    all_domains = [all_domains[i] for i in indices]
    all_lengths = [all_lengths[i] for i in indices]

    if not all_signals:
        print("\nERROR: No signals collected. Check data paths and dependencies.")
        return
    print(f"\nTotal: {len(all_signals)} windows, "
          f"lengths=[{min(all_lengths)}, {int(np.median(all_lengths))}, {max(all_lengths)}]")

    # Save as parquet
    table = pa.table({
        "signal": pa.array(all_signals, type=pa.list_(pa.float64())),
        "domain": pa.array(all_domains),
        "length": pa.array(all_lengths),
    })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pq.write_table(table, output_path, compression="zstd")
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nSaved: {output_path} ({size_mb:.1f} MB, {len(all_signals)} rows)")


def main():
    p = argparse.ArgumentParser(description="Build tokenizer training data as parquet")
    p.add_argument("--output", default="data/tokenizer_training.parquet")
    p.add_argument("--data-root", default="data",
                   help="Root dir for local datasets (cwru_github, PAMAP2, mfpt)")
    p.add_argument("--max-per-domain", type=int, default=100000)
    p.add_argument("--max-total", type=int, default=500000)
    p.add_argument("--min-length", type=int, default=128)
    p.add_argument("--max-length", type=int, default=4096)
    args = p.parse_args()

    domain_signals = collect_all(
        max_per_domain=args.max_per_domain,
        min_length=args.min_length,
        max_length=args.max_length,
        data_root=args.data_root,
    )

    save_parquet(domain_signals, args.output, max_total=args.max_total)
    print(f"\nUpload to S3:")
    print(f"  aws s3 cp {args.output} s3://<your-s3-bucket>/tempo/data/")


if __name__ == "__main__":
    main()
