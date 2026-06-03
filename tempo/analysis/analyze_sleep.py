"""Analyze FSQ Transformer tokenization on Sleep-EDF EEG data.

Produces:
1. Token usage heatmap
2. t-SNE of code distributions by sleep stage
3. Example EEG signals with color-coded tokenization
"""

import sys
import ast
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from collections import Counter, defaultdict
from sklearn.manifold import TSNE
import pandas as pd

sys.path.insert(0, ".")


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


def load_sleep_signals(max_per_class=50):
    """Load Sleep-EDF signals with stage labels."""
    print("  Loading sleep CSV (this takes a moment)...")
    df = pd.read_csv("src/data/sleep/sleep_cot.csv", nrows=5000)

    signals, labels = [], []
    stage_map = {
        "W": "Wake", "N1": "N1 (light)", "N2": "N2 (spindles)",
        "N3": "N3 (deep)", "REM": "REM",
    }

    for _, row in df.iterrows():
        label = str(row["label"]).strip()
        if label not in stage_map:
            continue
        try:
            ts_raw = ast.literal_eval(row["time_series"])
            if isinstance(ts_raw, list) and len(ts_raw) == 1 and isinstance(ts_raw[0], list):
                signal = ts_raw[0]
            elif isinstance(ts_raw, list):
                signal = ts_raw
            else:
                continue
        except (ValueError, SyntaxError):
            continue

        if len(signal) < 100:
            continue

        signals.append(signal)
        labels.append(stage_map[label])

    # Balance
    by_class = defaultdict(list)
    for sig, lab in zip(signals, labels):
        by_class[lab].append(sig)

    balanced_sigs, balanced_labs = [], []
    for lab in ["Wake", "N1 (light)", "N2 (spindles)", "N3 (deep)", "REM"]:
        for sig in by_class.get(lab, [])[:max_per_class]:
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

    stage_colors = {
        "Wake": "#e41a1c",
        "N1 (light)": "#ff7f00",
        "N2 (spindles)": "#4daf4a",
        "N3 (deep)": "#377eb8",
        "REM": "#984ea3",
    }
    for label in ["Wake", "N1 (light)", "N2 (spindles)", "N3 (deep)", "REM"]:
        mask = labels_arr == label
        if mask.sum() > 0:
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=stage_colors[label], label=label, s=25, alpha=0.7)
    ax.legend(fontsize=8, loc="best")
    ax.set_title("t-SNE of Code Distributions by Sleep Stage", fontsize=13, fontweight="bold")
    ax.set_xlabel("Dim-1")
    ax.set_ylabel("Dim-2")


def plot_tokenized_signals(tokenizer, signals, labels, n_codes, patch_size, ax,
                           n_per_stage=2):
    """Plot example EEG signals with color-coded tokens, grouped by stage."""
    stages = ["Wake", "N1 (light)", "N2 (spindles)", "N3 (deep)", "REM"]

    # Pick examples per stage
    by_stage = defaultdict(list)
    for sig, lab in zip(signals, labels):
        by_stage[lab].append(sig)

    examples = []
    for stage in stages:
        for sig in by_stage.get(stage, [])[:n_per_stage]:
            examples.append((sig, stage))

    cmap = plt.cm.hsv
    n_examples = len(examples)

    # Subsample long signals for visibility (show first 256 points = ~5s at 50Hz)
    max_pts = 256

    for i, (signal, label) in enumerate(examples):
        sig_short = signal[:max_pts]
        codes, sig_norm = tokenize(tokenizer, sig_short)
        n_pts = len(sig_norm)

        y_offset = (n_examples - 1 - i) * 4.0

        for p in range(len(codes)):
            start = p * patch_size
            end = min(start + patch_size + 1, n_pts)
            if start >= n_pts:
                break
            x_seg = np.arange(start, min(end, n_pts))
            y_seg = sig_norm[start:min(end, n_pts)] + y_offset
            color = cmap(codes[p] / max(n_codes, 1))
            ax.plot(x_seg, y_seg, color=color, linewidth=1.2, solid_capstyle="round")

        ax.text(-5, y_offset, label, fontsize=7, ha="right", va="center",
                fontweight="bold")

    ax.set_xlim(-60, max_pts + 5)
    ax.set_ylim(-3, n_examples * 4.0 + 1)
    ax.set_xlabel("Sample (50 Hz) — first 5.12s of 30s epoch")
    ax.set_yticks([])
    ax.set_title("EEG Tokenization by Sleep Stage", fontsize=13, fontweight="bold")


def main():
    import os
    os.makedirs("results", exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    n_codes = tokenizer.codebook_size
    patch_size = tokenizer.config.patch_size
    print(f"  {n_codes} codes, patch_size={patch_size}")

    print("Loading sleep signals...")
    signals, labels = load_sleep_signals(max_per_class=50)
    print(f"  {len(signals)} signals, stages: {sorted(set(labels))}")

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

    # Figure
    fig = plt.figure(figsize=(20, 7))
    fig.suptitle("FSQ Transformer (625 codes) — Sleep-EDF EEG Analysis",
                 fontsize=16, fontweight="bold")

    ax1 = fig.add_subplot(131)
    plot_usage_heatmap(all_codes, n_codes, ax1)

    ax2 = fig.add_subplot(132)
    plot_tsne_by_class(all_codes, all_labels, n_codes, ax2)

    ax3 = fig.add_subplot(133)
    plot_tokenized_signals(tokenizer, signals, labels, n_codes, patch_size, ax3,
                           n_per_stage=2)

    plt.tight_layout()
    out_path = "results/sleep_tokenizer_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
