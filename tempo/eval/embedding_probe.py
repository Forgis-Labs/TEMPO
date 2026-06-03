"""Linear probing of TS embeddings — measures alignment quality.

Extracts TS embeddings from a TEMPO checkpoint, trains a logistic
regression per UCR dataset, and reports accuracy. Compares two checkpoints
to show the effect of imputation pre-training on embedding quality.

Usage:
    # Single checkpoint
    python -m tempo.eval.embedding_probe \
        --checkpoint path/to/phase0_best.pt \
        --tokenizer-ckpt path/to/fsq_transformer_rope_625_best.pt \
        --output-dir results/embedding_probe

    # Compare two checkpoints (v3 vs v4)
    python -m tempo.eval.embedding_probe \
        --checkpoint path/to/v3_phase0.pt \
        --checkpoint-b path/to/v4_phase0.pt \
        --labels "Without Imputation" "With Imputation" \
        --tokenizer-ckpt path/to/fsq_transformer_rope_625_best.pt \
        --output-dir results/embedding_probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# Embedding extraction

def extract_embeddings(
    model,
    signals: list[np.ndarray],
    device: str = "cpu",
) -> np.ndarray:
    """Extract TS embeddings from a TEMPO model for a list of signals.

    Returns (N, D) array where D is the LLM hidden dimension.
    Each signal is tokenized → codes → look up in the TS embedding table
    → mean-pool across code positions.
    """
    model.eval()
    embeddings = []

    # Get the embedding layer
    embed_layer = model.llm.get_input_embeddings()
    tokenizer = model.tokenizer

    with torch.no_grad():
        for sig in signals:
            sig_t = torch.tensor(sig, dtype=torch.float32)
            codes = model.tokenize_ts(sig_t)

            # Convert codes to token IDs
            code_tokens = [f"<ts_{c}>" for c in codes]
            token_ids = tokenizer.convert_tokens_to_ids(code_tokens)
            token_ids_t = torch.tensor(token_ids, dtype=torch.long).to(device)

            # Get embeddings and mean-pool
            embs = embed_layer(token_ids_t)  # (n_codes, hidden_dim)
            pooled = embs.mean(dim=0).float().cpu().numpy()
            embeddings.append(pooled)

    return np.array(embeddings)


# UCR dataset loading

# Datasets with enough samples and clear class structure
DEFAULT_DATASETS = [
    "ArrowHead", "CBF", "ECG200", "FaceFour", "GunPoint",
    "Trace", "Wafer", "SyntheticControl", "TwoPatterns",
    "FaceAll", "StarLightCurves", "Chinatown", "ItalyPowerDemand",
    "SmoothSubspace", "UMD",
]


def load_ucr_datasets(dataset_names: list[str] | None = None) -> dict:
    """Load UCR datasets, return dict of name -> (X_train, y_train, X_test, y_test)."""
    from aeon.datasets import load_classification

    if dataset_names is None:
        dataset_names = DEFAULT_DATASETS

    datasets = {}
    for name in dataset_names:
        try:
            X_train, y_train = load_classification(name, split="train")
            X_test, y_test = load_classification(name, split="test")

            # Flatten to 1D signals (univariate)
            train_signals = [X_train[i, 0].astype(np.float64) for i in range(len(X_train))]
            test_signals = [X_test[i, 0].astype(np.float64) for i in range(len(X_test))]

            datasets[name] = {
                "train_signals": train_signals,
                "train_labels": y_train,
                "test_signals": test_signals,
                "test_labels": y_test,
                "n_classes": len(set(y_train)),
            }
            print(f"  {name}: train={len(train_signals)}, test={len(test_signals)}, "
                  f"classes={datasets[name]['n_classes']}")
        except Exception as e:
            print(f"  {name}: SKIP ({e})")

    return datasets


# Linear probing

def probe_dataset(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
) -> dict:
    """Train logistic regression on embeddings, return accuracy."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_emb)
    X_test = scaler.transform(test_emb)

    clf = LogisticRegression(
        max_iter=1000, solver="lbfgs", multi_class="multinomial",
        C=1.0, random_state=42,
    )
    clf.fit(X_train, train_labels)

    train_acc = accuracy_score(train_labels, clf.predict(X_train))
    test_acc = accuracy_score(test_labels, clf.predict(X_test))

    return {"train_acc": round(train_acc, 4), "test_acc": round(test_acc, 4)}


def run_probe(
    model,
    datasets: dict,
    device: str = "cpu",
    label: str = "model",
) -> dict:
    """Run linear probing across all datasets for one model."""
    results = {}
    for name, data in datasets.items():
        print(f"  [{label}] Probing {name}...", end=" ")

        train_emb = extract_embeddings(model, data["train_signals"], device)
        test_emb = extract_embeddings(model, data["test_signals"], device)

        res = probe_dataset(train_emb, data["train_labels"], test_emb, data["test_labels"])
        results[name] = res
        print(f"test_acc={res['test_acc']:.1%}")

    avg_acc = np.mean([r["test_acc"] for r in results.values()])
    results["_average"] = {"test_acc": round(float(avg_acc), 4)}
    print(f"  [{label}] Average: {avg_acc:.1%}")

    return results


# Visualization

def plot_comparison(
    results_a: dict,
    results_b: dict,
    label_a: str,
    label_b: str,
    output_path: str,
):
    """Bar chart comparing probe accuracy across datasets."""
    import matplotlib.pyplot as plt

    datasets = [k for k in results_a if k != "_average"]
    acc_a = [results_a[k]["test_acc"] * 100 for k in datasets]
    acc_b = [results_b[k]["test_acc"] * 100 for k in datasets]

    # Sort by improvement
    improvement = [b - a for a, b in zip(acc_a, acc_b)]
    order = np.argsort(improvement)[::-1]
    datasets = [datasets[i] for i in order]
    acc_a = [acc_a[i] for i in order]
    acc_b = [acc_b[i] for i in order]

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars_a = ax.bar(x - width / 2, acc_a, width, label=label_a, color="#2196F3", alpha=0.8)
    bars_b = ax.bar(x + width / 2, acc_b, width, label=label_b, color="#FF5722", alpha=0.8)

    ax.set_ylabel("Linear Probe Accuracy (%)")
    ax.set_title("Embedding Quality: Linear Probe on UCR Datasets")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0, 105)

    # Add average line
    avg_a = results_a["_average"]["test_acc"] * 100
    avg_b = results_b["_average"]["test_acc"] * 100
    ax.axhline(avg_a, color="#2196F3", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(avg_b, color="#FF5722", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(len(datasets) - 0.5, avg_a + 1, f"avg={avg_a:.1f}%", color="#2196F3", fontsize=8)
    ax.text(len(datasets) - 0.5, avg_b + 1, f"avg={avg_b:.1f}%", color="#FF5722", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved comparison plot: {output_path}")


def plot_umap_embeddings(
    model,
    datasets: dict,
    device: str = "cpu",
    output_path: str = "fig_umap_embeddings.png",
    max_per_dataset: int = 100,
    label: str = "model",
):
    """UMAP of TS embeddings colored by dataset, shaped by class."""
    import matplotlib.pyplot as plt
    from umap import UMAP

    all_embs = []
    all_datasets = []
    all_labels = []

    for name, data in list(datasets.items())[:8]:  # max 8 datasets for clarity
        sigs = data["test_signals"][:max_per_dataset]
        labs = data["test_labels"][:max_per_dataset]
        embs = extract_embeddings(model, sigs, device)
        all_embs.append(embs)
        all_datasets.extend([name] * len(embs))
        all_labels.extend(labs)

    all_embs = np.concatenate(all_embs)

    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(all_embs)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_ds = sorted(set(all_datasets))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_ds)))

    for ds, color in zip(unique_ds, colors):
        mask = [d == ds for d in all_datasets]
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color], label=ds, alpha=0.6, s=15)

    ax.legend(fontsize=8, markerscale=2, loc="best")
    ax.set_title(f"UMAP of TS Embeddings ({label})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved UMAP plot: {output_path}")


# Main

def main():
    parser = argparse.ArgumentParser(description="Linear probing of TS embeddings")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint A path")
    parser.add_argument("--checkpoint-b", default=None, help="Checkpoint B path (for comparison)")
    parser.add_argument("--labels", nargs=2, default=["Baseline", "With Imputation"],
                        help="Labels for A and B")
    parser.add_argument("--tokenizer-ckpt", required=True, help="Tokenizer checkpoint for model A")
    parser.add_argument("--tokenizer-type", default="fsq_transformer_rope")
    parser.add_argument("--tokenizer-ckpt-b", default=None, help="Tokenizer checkpoint for model B (if different)")
    parser.add_argument("--tokenizer-type-b", default=None, help="Tokenizer type for model B (if different)")
    parser.add_argument("--llm-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output-dir", default="results/embedding_probe")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="UCR dataset names (default: 15 standard datasets)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EMBEDDING PROBE — Linear Classification on UCR")
    print("=" * 60)

    # Load UCR datasets
    print("\nLoading UCR datasets...")
    datasets = load_ucr_datasets(args.datasets)
    if not datasets:
        print("ERROR: No datasets loaded")
        return

    # Load model A
    from tempo import TEMPO
    print(f"\nLoading checkpoint A: {args.checkpoint}")
    model_a = TEMPO.from_pretrained(
        args.checkpoint, llm_id=args.llm_id,
        tokenizer_type=args.tokenizer_type,
        fsq_ckpt=args.tokenizer_ckpt,
        use_dora=True, device=args.device,
    )

    print(f"\nProbing {args.labels[0]}...")
    results_a = run_probe(model_a, datasets, args.device, args.labels[0])

    # UMAP for model A
    plot_umap_embeddings(
        model_a, datasets, args.device,
        str(output_dir / f"fig_umap_{args.labels[0].replace(' ', '_').lower()}.png"),
        label=args.labels[0],
    )

    # Compare if checkpoint B provided
    if args.checkpoint_b:
        tok_type_b = args.tokenizer_type_b or args.tokenizer_type
        tok_ckpt_b = args.tokenizer_ckpt_b or args.tokenizer_ckpt
        # For TOTEM, pass as totem_ckpt instead of fsq_ckpt
        b_kwargs = {"use_dora": True, "device": args.device}
        if tok_type_b == "totem":
            b_kwargs["totem_ckpt"] = tok_ckpt_b
        else:
            b_kwargs["fsq_ckpt"] = tok_ckpt_b

        print(f"\nLoading checkpoint B: {args.checkpoint_b} (tokenizer: {tok_type_b})")
        model_b = TEMPO.from_pretrained(
            args.checkpoint_b, llm_id=args.llm_id,
            tokenizer_type=tok_type_b,
            **b_kwargs,
        )

        print(f"\nProbing {args.labels[1]}...")
        results_b = run_probe(model_b, datasets, args.device, args.labels[1])

        # UMAP for model B
        plot_umap_embeddings(
            model_b, datasets, args.device,
            str(output_dir / f"fig_umap_{args.labels[1].replace(' ', '_').lower()}.png"),
            label=args.labels[1],
        )

        # Comparison plot
        plot_comparison(
            results_a, results_b,
            args.labels[0], args.labels[1],
            str(output_dir / "fig_probe_comparison.png"),
        )

        # Summary table
        print(f"\n{'=' * 60}")
        print(f"{'Dataset':<25} {args.labels[0]:<15} {args.labels[1]:<15} {'Delta':>8}")
        print("-" * 63)
        for name in sorted(results_a):
            if name == "_average":
                continue
            a = results_a[name]["test_acc"] * 100
            b = results_b[name]["test_acc"] * 100
            delta = b - a
            print(f"  {name:<23} {a:>6.1f}%       {b:>6.1f}%       {delta:>+6.1f}%")
        a_avg = results_a["_average"]["test_acc"] * 100
        b_avg = results_b["_average"]["test_acc"] * 100
        print("-" * 63)
        print(f"  {'AVERAGE':<23} {a_avg:>6.1f}%       {b_avg:>6.1f}%       {b_avg - a_avg:>+6.1f}%")

        all_results = {
            args.labels[0]: results_a,
            args.labels[1]: results_b,
        }
    else:
        all_results = {args.labels[0]: results_a}

    # Save results
    with open(output_dir / "probe_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {output_dir}/")


if __name__ == "__main__":
    main()
