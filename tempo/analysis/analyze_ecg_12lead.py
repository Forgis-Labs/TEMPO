"""Analyze FSQ Transformer tokenization on 12-lead ECG (PTB-XL).

Shows: heatmap, t-SNE by diagnosis, tokenized 12-lead examples.
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
import wfdb

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


def load_ecg_12lead(max_per_class=20, max_records=300):
    """Load PTB-XL 12-lead ECG with diagnostic labels."""
    meta_path = os.path.join(PTBXL_DIR, "ptbxl_database.csv")
    meta = pd.read_csv(meta_path) if os.path.exists(meta_path) else None

    signals_12lead, labels = [], []
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
                sig = record.p_signal  # (n_samples, 12)
                if sig is None or sig.shape[1] < 12:
                    continue

                ecg_id = int(fname.split("_")[0])
                label = "Unknown"
                if meta is not None and ecg_id in meta["ecg_id"].values:
                    row = meta[meta["ecg_id"] == ecg_id].iloc[0]
                    scp = str(row.get("scp_codes", "{}"))
                    if "NORM" in scp:
                        label = "Normal"
                    elif "MI" in scp or "IMI" in scp:
                        label = "MI"
                    elif "STTC" in scp or "STD" in scp:
                        label = "ST-T Change"
                    elif "HYP" in scp or "LVH" in scp:
                        label = "Hypertrophy"
                    elif "CD" in scp or "LBBB" in scp or "RBBB" in scp:
                        label = "Conduction"
                    else:
                        label = "Other"

                # Take first 2s at 500Hz = 1000 pts, all 12 leads
                all_leads = []
                valid = True
                for lead_idx in range(12):
                    lead = sig[:1000, lead_idx].astype(np.float32)
                    if np.isnan(lead).any() or lead.std() < 1e-10:
                        valid = False
                        break
                    all_leads.append(lead)
                if not valid or len(all_leads) != 12:
                    continue

                signals_12lead.append(all_leads)
                labels.append(label)
                record_count += 1
            except Exception:
                continue
        if record_count >= max_records:
            break

    # Balance
    by_class = defaultdict(list)
    for sig, lab in zip(signals_12lead, labels):
        by_class[lab].append(sig)

    balanced_sigs, balanced_labs = [], []
    for lab in sorted(by_class.keys()):
        for sig in by_class[lab][:max_per_class]:
            balanced_sigs.append(sig)
            balanced_labs.append(lab)

    return balanced_sigs, balanced_labs


def main():
    os.makedirs("results", exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    n_codes = tokenizer.codebook_size
    patch_size = tokenizer.config.patch_size

    print("Loading 12-lead ECG...")
    signals, labels = load_ecg_12lead(max_per_class=20, max_records=300)
    print(f"  {len(signals)} records, classes: {sorted(set(labels))}")
    for lab in sorted(set(labels)):
        print(f"    {lab}: {sum(1 for l in labels if l == lab)}")

    # Tokenize all 12 leads per record, concatenate code distributions
    print("Tokenizing 12 leads per record...")
    all_codes_concat = []  # flattened across all leads for heatmap
    all_code_dists = []    # per-record 12-lead code frequency vector for t-SNE
    all_labels = []
    per_lead_codes = defaultdict(list)  # lead_idx -> list of code arrays

    for sig_12, lab in zip(signals, labels):
        record_codes = []
        freq = np.zeros(n_codes)
        for lead_idx in range(12):
            codes, _ = tokenize(tokenizer, sig_12[lead_idx])
            record_codes.append(codes)
            per_lead_codes[lead_idx].append(codes)
            for c in codes:
                if c < n_codes:
                    freq[c] += 1
        all_codes_concat.append(np.concatenate(record_codes))
        freq = freq / max(freq.sum(), 1)
        all_code_dists.append(freq)
        all_labels.append(lab)

    flat = np.concatenate(all_codes_concat)
    n_unique = len(set(flat))
    print(f"  {n_unique}/{n_codes} codes used ({100*n_unique/n_codes:.0f}%)")

    # === Figure 1: Heatmap + t-SNE + tokenized example ===
    fig = plt.figure(figsize=(22, 7))
    fig.suptitle("FSQ Transformer (625 codes) — 12-Lead ECG Analysis",
                 fontsize=16, fontweight="bold")

    # Heatmap
    ax1 = fig.add_subplot(131)
    counts = Counter(flat)
    grid_size = int(np.ceil(np.sqrt(n_codes)))
    grid = np.zeros((grid_size, grid_size))
    for code_id in range(n_codes):
        r, c = divmod(code_id, grid_size)
        grid[r, c] = counts.get(code_id, 0)
    grid[grid == 0] = 0.5
    im = ax1.imshow(grid, cmap="Reds",
                    norm=LogNorm(vmin=max(1, grid[grid > 0].min()), vmax=grid.max()))
    ax1.set_title("Token Usage (all 12 leads)", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax1, shrink=0.8)

    # t-SNE of 12-lead code distributions
    ax2 = fig.add_subplot(132)
    X = np.array(all_code_dists)
    labels_arr = np.array(all_labels)
    tsne = TSNE(n_components=2, perplexity=min(30, len(X) - 1),
                random_state=42, max_iter=1000)
    embedding = tsne.fit_transform(X)
    unique_labels = sorted(set(all_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        mask = labels_arr == label
        ax2.scatter(embedding[mask, 0], embedding[mask, 1],
                    c=[colors[i]], label=label, s=25, alpha=0.7)
    ax2.legend(fontsize=8)
    ax2.set_title("t-SNE by Diagnosis (12-lead combined)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Dim-1")
    ax2.set_ylabel("Dim-2")

    # Tokenized 12-lead example (one per class)
    ax3 = fig.add_subplot(133)
    cmap = plt.cm.hsv
    by_class = defaultdict(list)
    for idx, lab in enumerate(all_labels):
        by_class[lab].append(idx)

    # Pick one example from each class
    example_indices = []
    for lab in sorted(by_class.keys()):
        if by_class[lab]:
            example_indices.append((by_class[lab][0], lab))

    n_examples = len(example_indices)
    max_pts = 200  # Show ~0.4s for visibility

    for ex_i, (rec_idx, label) in enumerate(example_indices):
        sig_12 = signals[rec_idx]
        # Show Lead II only for the overview
        codes, sig_norm = tokenize(tokenizer, sig_12[1][:max_pts])
        y_offset = (n_examples - 1 - ex_i) * 4.0

        for p in range(len(codes)):
            start = p * patch_size
            end = min(start + patch_size + 1, len(sig_norm))
            if start >= len(sig_norm):
                break
            x_seg = np.arange(start, min(end, len(sig_norm)))
            y_seg = sig_norm[start:min(end, len(sig_norm))] + y_offset
            color = cmap(codes[p] / max(n_codes, 1))
            ax3.plot(x_seg, y_seg, color=color, linewidth=1.5)

        ax3.text(-5, y_offset, label, fontsize=7, ha="right", va="center", fontweight="bold")

    ax3.set_xlim(-50, max_pts + 5)
    ax3.set_title("Tokenized Lead II (one per class)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Sample (500 Hz)")
    ax3.set_yticks([])

    plt.tight_layout()
    fig.savefig("results/ecg_12lead_overview.png", dpi=150, bbox_inches="tight")
    print("  Saved results/ecg_12lead_overview.png")

    # === Figure 2: Full 12-lead tokenization for 2 patients (Normal vs MI) ===
    fig2, axes2 = plt.subplots(12, 2, figsize=(16, 24), sharex=True)
    fig2.suptitle("12-Lead ECG Tokenization: Normal vs Myocardial Infarction",
                  fontsize=16, fontweight="bold", y=1.01)

    # Find one Normal and one MI
    normal_idx = next((i for i, l in enumerate(all_labels) if l == "Normal"), None)
    mi_idx = next((i for i, l in enumerate(all_labels) if l == "MI"), None)

    if normal_idx is None or mi_idx is None:
        print("  Could not find both Normal and MI examples")
    else:
        max_pts = 500  # 1s at 500Hz
        for col, (rec_idx, title) in enumerate([(normal_idx, "Normal"), (mi_idx, "MI")]):
            axes2[0, col].set_title(title, fontsize=14, fontweight="bold")
            sig_12 = signals[rec_idx]
            for lead_idx in range(12):
                ax = axes2[lead_idx, col]
                codes, sig_norm = tokenize(tokenizer, sig_12[lead_idx][:max_pts])

                for p in range(len(codes)):
                    start = p * patch_size
                    end = min(start + patch_size + 1, len(sig_norm))
                    if start >= len(sig_norm):
                        break
                    x_seg = np.arange(start, min(end, len(sig_norm)))
                    y_seg = sig_norm[start:min(end, len(sig_norm))]
                    color = cmap(codes[p] / max(n_codes, 1))
                    ax.plot(x_seg, y_seg, color=color, linewidth=1.2)

                ax.set_ylabel(LEAD_NAMES[lead_idx], fontsize=9, fontweight="bold")
                ax.grid(True, alpha=0.2)
                if lead_idx < 11:
                    ax.set_xticklabels([])

            axes2[11, col].set_xlabel("Sample (500 Hz)")

    plt.tight_layout()
    fig2.savefig("results/ecg_12lead_normal_vs_mi.png", dpi=150, bbox_inches="tight")
    print("  Saved results/ecg_12lead_normal_vs_mi.png")

    # === Figure 3: Per-lead t-SNE (do different leads separate differently?) ===
    fig3, axes3 = plt.subplots(3, 4, figsize=(20, 14))
    fig3.suptitle("t-SNE by Diagnosis — Per Lead",
                  fontsize=16, fontweight="bold")

    for lead_idx in range(12):
        ax = axes3[lead_idx // 4, lead_idx % 4]
        # Build per-lead frequency vectors
        lead_freq_vecs = []
        for codes_arr in per_lead_codes[lead_idx]:
            freq = np.zeros(n_codes)
            for c in codes_arr:
                if c < n_codes:
                    freq[c] += 1
            freq = freq / max(freq.sum(), 1)
            lead_freq_vecs.append(freq)

        X_lead = np.array(lead_freq_vecs)
        if len(X_lead) < 5:
            continue

        tsne_lead = TSNE(n_components=2, perplexity=min(20, len(X_lead) - 1),
                         random_state=42, max_iter=800)
        emb = tsne_lead.fit_transform(X_lead)

        for i, label in enumerate(unique_labels):
            mask = labels_arr == label
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=[colors[i]], label=label if lead_idx == 0 else "", s=15, alpha=0.6)

        ax.set_title(f"Lead {LEAD_NAMES[lead_idx]}", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # Add legend to first subplot
    axes3[0, 0].legend(fontsize=7, loc="best")

    plt.tight_layout()
    fig3.savefig("results/ecg_12lead_per_lead_tsne.png", dpi=150, bbox_inches="tight")
    print("  Saved results/ecg_12lead_per_lead_tsne.png")
    plt.show()


if __name__ == "__main__":
    main()
