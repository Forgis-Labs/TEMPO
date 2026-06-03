"""Analyze FSQ Transformer and TOTEM tokenizers.

For each tokenizer produces one figure with:
1. Token usage heatmap
2. t-SNE of code distributions by class
3. Code waveform archetypes
4. Example signals with color-coded tokenization

Usage:
    python analyze_tokenizer.py
"""

import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection
from collections import Counter, defaultdict
from sklearn.manifold import TSNE

sys.path.insert(0, ".")


def load_tokenizer(name):
    if name == "fsq_transformer":
        from tempo.tokenizer.fsq_transformer import FSQTransformerTokenizer
        tok = FSQTransformerTokenizer.from_pretrained(
            "checkpoints/2026-04-23_FSQ_Transformer_Qwen3-4B/fsq_transformer.pt"
        )
        tok.eval()
        return tok, tok.codebook_size, 4, "FSQ Transformer (625 codes)"
    elif name == "totem":
        from tempo.tokenizer.totem import TOTEMTokenizer
        tok = TOTEMTokenizer.from_pretrained("checkpoints/totem_clean_local.pt")
        tok.eval()
        return tok, 256, 4, "TOTEM (256 codes)"
    raise ValueError(f"Unknown: {name}")


def tokenize_signal(tokenizer, signal):
    """Tokenize a single signal, return code IDs."""
    sig_t = torch.tensor(signal, dtype=torch.float32)
    sig_t = (sig_t - sig_t.mean()) / max(sig_t.std().item(), 1e-8)
    with torch.no_grad():
        codes = tokenizer.tokenize(sig_t.unsqueeze(0))
    return codes.squeeze(0).numpy(), sig_t.numpy()


def load_har_signals(max_per_class=50):
    import pandas as pd
    df = pd.read_csv("src/data/har_cot/har_cot_test_cot.csv", nrows=2000)
    signals, labels = [], []
    for _, row in df.iterrows():
        label = str(row["label"]).strip()
        if label == "label":
            continue
        try:
            x = json.loads(row["x_axis"])
        except (json.JSONDecodeError, TypeError):
            continue
        signals.append(x)
        labels.append(label)
    by_class = defaultdict(list)
    for sig, lab in zip(signals, labels):
        by_class[lab].append(sig)
    balanced_sigs, balanced_labs = [], []
    for lab, sigs in by_class.items():
        for sig in sigs[:max_per_class]:
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
    grid[grid == 0] = 0.5  # avoid log(0)
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
                   c=[colors[i]], label=label, s=20, alpha=0.7)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.set_title("t-SNE of Code Distributions by Class", fontsize=13, fontweight="bold")
    ax.set_xlabel("Dim-1")
    ax.set_ylabel("Dim-2")


def plot_code_waveforms(tokenizer, n_codes, n_show=25, ax=None):
    """Decode each code to its waveform pattern."""
    waveforms = {}
    for code_id in range(n_codes):
        try:
            codes_tensor = torch.tensor([code_id], dtype=torch.long)
            with torch.no_grad():
                decoded = tokenizer.decode_codes(codes_tensor.unsqueeze(0))
            waveforms[code_id] = decoded.squeeze().numpy()
        except Exception:
            continue

    if not waveforms:
        ax.text(0.5, 0.5, "Could not decode codes", transform=ax.transAxes,
                ha="center", va="center")
        ax.axis("off")
        return

    sorted_codes = sorted(waveforms.keys(),
                          key=lambda c: np.std(waveforms[c]), reverse=True)
    n_show = min(n_show, len(sorted_codes))
    n_cols = 5
    n_rows = int(np.ceil(n_show / n_cols))

    for i, code_id in enumerate(sorted_codes[:n_show]):
        sub_ax = ax.inset_axes([
            (i % n_cols) / n_cols + 0.02,
            1 - (i // n_cols + 1) / n_rows + 0.02,
            1 / n_cols - 0.04,
            1 / n_rows - 0.04,
        ])
        wf = waveforms[code_id]
        sub_ax.plot(wf, linewidth=1.5, color=plt.cm.tab20(code_id % 20))
        sub_ax.set_title(f"#{code_id}", fontsize=6)
        sub_ax.set_xticks([])
        sub_ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Code Archetypes (highest variance)", fontsize=13, fontweight="bold")
    ax.axis("off")


def plot_tokenized_signals(tokenizer, signals, labels, patch_size, n_codes, ax,
                           n_examples=6):
    """Plot example signals with each patch colored by its token ID."""
    # Pick diverse examples
    unique_labels = sorted(set(labels))
    examples = []
    for lab in unique_labels:
        for sig, l in zip(signals, labels):
            if l == lab and len(examples) < n_examples:
                examples.append((sig, lab))
                break
        if len(examples) >= n_examples:
            break

    # Fill remaining slots
    while len(examples) < n_examples:
        idx = len(examples)
        if idx < len(signals):
            examples.append((signals[idx], labels[idx]))
        else:
            break

    cmap = plt.cm.hsv
    n_examples = len(examples)

    for i, (signal, label) in enumerate(examples):
        codes, sig_norm = tokenize_signal(tokenizer, signal)
        n_pts = len(sig_norm)
        n_patches = len(codes)

        # Y offset for stacking
        y_offset = (n_examples - 1 - i) * 3.5

        # Plot each patch as a colored segment
        for p in range(n_patches):
            start = p * patch_size
            end = min(start + patch_size, n_pts)
            if start >= n_pts:
                break

            x_seg = np.arange(start, end)
            y_seg = sig_norm[start:end] + y_offset

            code_id = codes[p]
            color = cmap(code_id / max(n_codes, 1))

            ax.plot(x_seg, y_seg, color=color, linewidth=1.5, solid_capstyle="round")

            # Label the code ID at the center of each patch
            if p % 4 == 0:  # label every 4th patch to avoid clutter
                cx = (start + end) / 2
                cy = y_offset + sig_norm[start:end].mean()
                ax.text(cx, cy + 1.2, str(code_id), fontsize=4,
                        ha="center", va="bottom", color=color, fontweight="bold")

        # Activity label
        ax.text(-5, y_offset, label, fontsize=8, ha="right", va="center",
                fontweight="bold")

    ax.set_xlim(-30, n_pts + 5)
    ax.set_ylim(-2, n_examples * 3.5 + 1)
    ax.set_xlabel("Sample index")
    ax.set_yticks([])
    ax.set_title("Signal Tokenization (color = token ID)", fontsize=13, fontweight="bold")


def analyze_one_tokenizer(name, signals, labels):
    """Generate full analysis figure for one tokenizer."""
    import os
    os.makedirs("results", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"{'='*60}")

    tokenizer, n_codes, patch_size, display_name = load_tokenizer(name)
    print(f"  {n_codes} codes, patch_size={patch_size}")

    print("  Tokenizing...")
    all_codes = []
    all_labels = []
    for sig, lab in zip(signals, labels):
        codes, _ = tokenize_signal(tokenizer, sig)
        all_codes.append(codes)
        all_labels.append(lab)
    print(f"  {len(all_codes)} signals, avg {np.mean([len(c) for c in all_codes]):.0f} codes/signal")

    flat = np.concatenate(all_codes)
    n_unique = len(set(flat))
    print(f"  {n_unique}/{n_codes} unique codes used ({100*n_unique/n_codes:.0f}%)")

    # Figure: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(display_name, fontsize=18, fontweight="bold", y=0.98)

    plot_usage_heatmap(all_codes, n_codes, axes[0, 0])
    plot_tsne_by_class(all_codes, all_labels, n_codes, axes[0, 1])
    plot_code_waveforms(tokenizer, n_codes, n_show=25, ax=axes[1, 0])
    plot_tokenized_signals(tokenizer, signals, labels, patch_size, n_codes,
                           axes[1, 1], n_examples=8)

    plt.tight_layout()
    out_path = f"results/tokenizer_{name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved to {out_path}")
    plt.close()


def main():
    print("Loading HAR signals...")
    signals, labels = load_har_signals(max_per_class=50)
    print(f"  {len(signals)} signals, {len(set(labels))} classes")

    for tok_name in ["fsq_transformer", "totem"]:
        try:
            analyze_one_tokenizer(tok_name, signals, labels)
        except Exception as e:
            print(f"  {tok_name} FAILED: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
