"""Analyze FSQ Transformer tokenization on PTB-XL ECG data.

Loads 12-lead ECG directly via wfdb, no OpenTSLM dependency.
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from collections import Counter, defaultdict
from sklearn.manifold import TSNE
import pandas as pd

sys.path.insert(0, ".")

PTBXL_DIR = "src/data/ptbxl"
RECORDS_DIR = os.path.join(PTBXL_DIR, "records500")
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def load_tokenizer():
    from tempo.tokenizer.fsq_transformer import FSQTransformerTokenizer
    tok = FSQTransformerTokenizer.from_pretrained(
        "checkpoints/2026-04-23_FSQ_Transformer_Qwen3-4B/fsq_transformer.pt"
    )
    tok.eval()
    return tok


def tokenize(tokenizer, signal):
    sig_t = torch.tensor(signal, dtype=torch.float32)
    sig_t = (sig_t - sig_t.mean()) / max(sig_t.std().item(), 1e-8)
    with torch.no_grad():
        codes = tokenizer.tokenize(sig_t.unsqueeze(0))
    return codes.squeeze(0).numpy(), sig_t.numpy()


def load_ecg_signals(max_per_class=30, max_records=500):
    """Load PTB-XL ECG signals with diagnostic labels."""
    import wfdb

    # Load the metadata
    meta_path = os.path.join(PTBXL_DIR, "ptbxl_database.csv")
    if not os.path.exists(meta_path):
        # Try to find it
        for f in os.listdir(PTBXL_DIR):
            if f.endswith(".csv") and "database" in f.lower():
                meta_path = os.path.join(PTBXL_DIR, f)
                break

    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
        print(f"  Metadata: {len(meta)} records")
    else:
        meta = None
        print("  No metadata CSV found, using folder structure")

    # Scan records
    signals, labels = [], []
    record_count = 0

    for folder in sorted(os.listdir(RECORDS_DIR)):
        folder_path = os.path.join(RECORDS_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        for fname in sorted(os.listdir(folder_path)):
            if not fname.endswith("_hr.hea"):
                continue
            if record_count >= max_records:
                break

            record_base = os.path.join(folder_path, fname.replace(".hea", ""))
            try:
                record = wfdb.rdrecord(record_base)
                sig = record.p_signal  # (n_samples, n_leads)
                if sig is None or sig.shape[1] < 12:
                    continue

                # Get a simple label from metadata or folder
                ecg_id = int(fname.split("_")[0])
                if meta is not None and ecg_id in meta["ecg_id"].values:
                    row = meta[meta["ecg_id"] == ecg_id].iloc[0]
                    scp_codes = str(row.get("scp_codes", "{}"))
                    # Simple classification: normal vs abnormal
                    if "NORM" in scp_codes:
                        label = "Normal"
                    elif "MI" in scp_codes or "IMI" in scp_codes:
                        label = "Myocardial Infarction"
                    elif "STTC" in scp_codes or "STD" in scp_codes:
                        label = "ST-T Change"
                    elif "HYP" in scp_codes or "LVH" in scp_codes:
                        label = "Hypertrophy"
                    elif "CD" in scp_codes or "LBBB" in scp_codes or "RBBB" in scp_codes:
                        label = "Conduction Dist."
                    else:
                        label = "Other"
                else:
                    label = "Unknown"

                # Use Lead II (most common for analysis), take first 2.5s at 500Hz = 1250 pts
                lead_ii = sig[:1250, 1].astype(np.float32)
                if np.isnan(lead_ii).any() or lead_ii.std() < 1e-10:
                    continue

                signals.append(lead_ii)
                labels.append(label)
                record_count += 1

            except Exception:
                continue

        if record_count >= max_records:
            break

    # Balance
    by_class = defaultdict(list)
    for sig, lab in zip(signals, labels):
        by_class[lab].append(sig)

    balanced_sigs, balanced_labs = [], []
    for lab in sorted(by_class.keys()):
        for sig in by_class[lab][:max_per_class]:
            balanced_sigs.append(sig)
            balanced_labs.append(lab)

    return balanced_sigs, balanced_labs


def plot_usage_heatmap(all_codes, n_codes, ax):
    flat = np.concatenate(all_codes)
    counts = Counter(flat)
    grid_size = int(np.ceil(np.sqrt(n_codes)))
    grid = np.zeros((grid_size, grid_size))
    for code_id in range(n_codes):
        r, c = divmod(code_id, grid_size)
        if r < grid_size and c < grid_size:
            grid[r, c] = counts.get(code_id, 0)
    grid[grid == 0] = 0.5
    im = ax.imshow(grid, cmap="Reds",
                   norm=LogNorm(vmin=max(1, grid[grid > 0].min()), vmax=grid.max()))
    ax.set_title("Token Usage Heatmap", fontsize=13, fontweight="bold")
    ax.set_xlabel("Code ID (column)")
    ax.set_ylabel("Code ID (row)")
    plt.colorbar(im, ax=ax, label="Count", shrink=0.8)


def plot_tsne_by_class(all_codes, all_labels, n_codes, ax):
    freq_vectors = []
    for codes in all_codes:
        freq = np.zeros(n_codes)
        for c in codes:
            if c < n_codes:
                freq[c] += 1
        freq = freq / max(freq.sum(), 1)
        freq_vectors.append(freq)
    X = np.array(freq_vectors)
    labels_arr = np.array(all_labels)
    tsne = TSNE(n_components=2, perplexity=min(30, len(X) - 1),
                random_state=42, max_iter=1000)
    embedding = tsne.fit_transform(X)
    unique_labels = sorted(set(all_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        mask = labels_arr == label
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[colors[i]], label=label, s=25, alpha=0.7)
    ax.legend(fontsize=7, loc="best")
    ax.set_title("t-SNE of Code Distributions by Diagnosis", fontsize=13, fontweight="bold")
    ax.set_xlabel("Dim-1")
    ax.set_ylabel("Dim-2")


def plot_tokenized_signals(tokenizer, signals, labels, n_codes, patch_size, ax,
                           n_per_class=1):
    """Plot example ECG signals with color-coded tokens."""
    by_class = defaultdict(list)
    for sig, lab in zip(signals, labels):
        by_class[lab].append(sig)

    examples = []
    for lab in sorted(by_class.keys()):
        for sig in by_class[lab][:n_per_class]:
            examples.append((sig, lab))

    cmap = plt.cm.hsv
    n_examples = len(examples)
    max_pts = 500  # Show ~1s at 500Hz

    for i, (signal, label) in enumerate(examples):
        sig_short = signal[:max_pts]
        codes, sig_norm = tokenize(tokenizer, sig_short)
        n_pts = len(sig_norm)
        y_offset = (n_examples - 1 - i) * 4.5

        for p in range(len(codes)):
            start = p * patch_size
            end = min(start + patch_size + 1, n_pts)
            if start >= n_pts:
                break
            x_seg = np.arange(start, min(end, n_pts))
            y_seg = sig_norm[start:min(end, n_pts)] + y_offset
            color = cmap(codes[p] / max(n_codes, 1))
            ax.plot(x_seg, y_seg, color=color, linewidth=1.2, solid_capstyle="round")

        ax.text(-5, y_offset, label, fontsize=6, ha="right", va="center",
                fontweight="bold")

    ax.set_xlim(-80, max_pts + 5)
    ax.set_ylim(-3, n_examples * 4.5 + 1)
    ax.set_xlabel("Sample (500 Hz) — Lead II, first 1s")
    ax.set_yticks([])
    ax.set_title("ECG Tokenization (Lead II)", fontsize=13, fontweight="bold")


def main():
    import os
    os.makedirs("results", exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    n_codes = tokenizer.codebook_size
    patch_size = tokenizer.config.patch_size

    print("Loading ECG signals...")
    signals, labels = load_ecg_signals(max_per_class=30, max_records=500)
    print(f"  {len(signals)} signals, classes: {sorted(set(labels))}")
    for lab in sorted(set(labels)):
        count = sum(1 for l in labels if l == lab)
        print(f"    {lab}: {count}")

    print("Tokenizing...")
    all_codes, all_labels = [], []
    for sig, lab in zip(signals, labels):
        codes, _ = tokenize(tokenizer, sig)
        all_codes.append(codes)
        all_labels.append(lab)

    flat = np.concatenate(all_codes)
    n_unique = len(set(flat))
    print(f"  {n_unique}/{n_codes} codes used ({100*n_unique/n_codes:.0f}%)")
    print(f"  avg {np.mean([len(c) for c in all_codes]):.0f} codes/signal")

    fig = plt.figure(figsize=(20, 7))
    fig.suptitle("FSQ Transformer (625 codes) — PTB-XL ECG Analysis (Lead II)",
                 fontsize=16, fontweight="bold")

    ax1 = fig.add_subplot(131)
    plot_usage_heatmap(all_codes, n_codes, ax1)

    ax2 = fig.add_subplot(132)
    plot_tsne_by_class(all_codes, all_labels, n_codes, ax2)

    ax3 = fig.add_subplot(133)
    plot_tokenized_signals(tokenizer, signals, labels, n_codes, patch_size, ax3)

    plt.tight_layout()
    out_path = "results/ecg_tokenizer_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
