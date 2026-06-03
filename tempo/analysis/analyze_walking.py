"""Compare walking vs walking_up vs walking_down tokenization.

Shows all 3 accelerometer axes for each activity, with color-coded tokens.
"""

import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

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


def load_walking_samples(n_per_class=5):
    """Load walking/walking_up/walking_down with all 3 axes."""
    import pandas as pd
    df = pd.read_csv("src/data/har_cot/har_cot_test_cot.csv", nrows=3000)

    by_class = defaultdict(list)
    for _, row in df.iterrows():
        label = str(row["label"]).strip()
        if label not in ("walking", "walking_up", "walking_down"):
            continue
        try:
            x = json.loads(row["x_axis"])
            y = json.loads(row["y_axis"])
            z = json.loads(row["z_axis"])
        except (json.JSONDecodeError, TypeError):
            continue
        by_class[label].append({"x": x, "y": y, "z": z})

    samples = {}
    for label in ["walking", "walking_up", "walking_down"]:
        samples[label] = by_class[label][:n_per_class]

    return samples


def plot_tokenized_channel(ax, tokenizer, signal, n_codes, patch_size, title=""):
    """Plot a single channel with color-coded tokens."""
    codes, sig_norm = tokenize(tokenizer, signal)
    n_pts = len(sig_norm)

    cmap = plt.cm.hsv
    for p in range(len(codes)):
        start = p * patch_size
        end = min(start + patch_size + 1, n_pts)  # +1 for continuity
        if start >= n_pts:
            break
        x_seg = np.arange(start, min(end, n_pts))
        y_seg = sig_norm[start:min(end, n_pts)]
        color = cmap(codes[p] / max(n_codes, 1))
        ax.plot(x_seg, y_seg, color=color, linewidth=2, solid_capstyle="round")

    ax.set_ylabel(title, fontsize=9, fontweight="bold")
    ax.set_xlim(0, n_pts)
    return codes


def main():
    import os
    os.makedirs("results", exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    n_codes = tokenizer.codebook_size
    patch_size = tokenizer.config.patch_size

    print("Loading walking samples...")
    samples = load_walking_samples(n_per_class=4)
    for label, sigs in samples.items():
        print(f"  {label}: {len(sigs)} samples")

    activities = ["walking", "walking_up", "walking_down"]
    n_examples = min(len(samples[a]) for a in activities)
    axes_names = ["x-axis", "y-axis", "z-axis"]
    axes_keys = ["x", "y", "z"]

    # Layout: columns = activities, rows = examples * 3 channels
    fig, axes = plt.subplots(
        n_examples * 3, len(activities),
        figsize=(18, 3 * n_examples * 3),
        sharex=True,
    )

    fig.suptitle(
        "Walking Activity Comparison — FSQ Transformer Tokenization\n"
        "(each color = one of 625 discrete tokens, 4-point patches at 50 Hz)",
        fontsize=16, fontweight="bold", y=1.01,
    )

    # Column headers
    for col, activity in enumerate(activities):
        axes[0, col].set_title(
            activity.replace("_", " ").title(),
            fontsize=14, fontweight="bold", pad=10,
        )

    # Collect code distributions for comparison
    code_dists = {a: {k: [] for k in axes_keys} for a in activities}

    for col, activity in enumerate(activities):
        for ex_idx in range(n_examples):
            sample = samples[activity][ex_idx]
            for ch_idx, (ch_key, ch_name) in enumerate(zip(axes_keys, axes_names)):
                row = ex_idx * 3 + ch_idx
                signal = sample[ch_key]
                codes = plot_tokenized_channel(
                    axes[row, col], tokenizer, signal, n_codes, patch_size,
                    title=f"#{ex_idx+1} {ch_name}" if col == 0 else "",
                )
                code_dists[activity][ch_key].extend(codes.tolist())

                # Only show x-axis label on bottom row
                if row < n_examples * 3 - 1:
                    axes[row, col].set_xticklabels([])
                else:
                    axes[row, col].set_xlabel("Sample (50 Hz)")

                # Light grid
                axes[row, col].grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = "results/walking_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path}")

    # --- Code distribution comparison ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle(
        "Token Distribution by Walking Activity (per axis)",
        fontsize=16, fontweight="bold",
    )

    for ch_idx, (ch_key, ch_name) in enumerate(zip(axes_keys, axes_names)):
        ax = axes2[ch_idx]
        for activity in activities:
            codes = code_dists[activity][ch_key]
            hist, _ = np.histogram(codes, bins=np.arange(n_codes + 1))
            hist = hist / max(hist.sum(), 1)
            ax.bar(
                np.arange(n_codes), hist, alpha=0.5,
                label=activity.replace("_", " "),
                width=1.0,
            )
        ax.set_title(ch_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Token ID")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9)
        ax.set_xlim(0, n_codes)

    plt.tight_layout()
    out_path2 = "results/walking_token_distributions.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path2}")

    # --- Print top discriminative codes ---
    print("\n=== Most Discriminative Codes (x-axis) ===")
    from collections import Counter
    for ch_key, ch_name in zip(axes_keys, axes_names):
        print(f"\n{ch_name}:")
        counters = {}
        for a in activities:
            c = Counter(code_dists[a][ch_key])
            total = sum(c.values())
            counters[a] = {k: v / total for k, v in c.items()}

        # Find codes with biggest difference between walking types
        all_code_ids = set()
        for c in counters.values():
            all_code_ids.update(c.keys())

        diffs = []
        for code_id in all_code_ids:
            freqs = [counters[a].get(code_id, 0) for a in activities]
            diff = max(freqs) - min(freqs)
            diffs.append((diff, code_id, {a: round(counters[a].get(code_id, 0), 4) for a in activities}))

        diffs.sort(reverse=True)
        for diff, code_id, freqs in diffs[:5]:
            print(f"  Code {code_id}: diff={diff:.4f}  {freqs}")

    plt.show()


if __name__ == "__main__":
    main()
