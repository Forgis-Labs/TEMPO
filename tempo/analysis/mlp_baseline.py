"""MLP baseline on FSQ codes — proves "why LLM".

Trains a simple MLP classifier on code frequency histograms.
If it matches the LLM on classification, the LLM's value is in
reasoning/explanation, not classification accuracy.

Usage:
    python -m tempo.analysis.mlp_baseline
"""

import sys
import json
import numpy as np
import torch
from collections import defaultdict
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, ".")


def load_tokenizer():
    from tempo.tokenizer.fsq_transformer import FSQTransformerTokenizer
    tok = FSQTransformerTokenizer.from_pretrained(
        "checkpoints/2026-04-23_FSQ_Transformer_Qwen3-4B/fsq_transformer.pt"
    )
    tok.eval()
    return tok


def tokenize_signal(tokenizer, signal):
    sig_t = torch.tensor(signal, dtype=torch.float32)
    sig_t = (sig_t - sig_t.mean()) / max(sig_t.std().item(), 1e-8)
    with torch.no_grad():
        codes = tokenizer.tokenize(sig_t.unsqueeze(0))
    return codes.squeeze(0).numpy()


def codes_to_histogram(codes, n_codes):
    """Convert code sequence to frequency histogram feature vector."""
    hist = np.zeros(n_codes)
    for c in codes:
        if c < n_codes:
            hist[c] += 1
    return hist / max(hist.sum(), 1)


def load_har_data(tokenizer, n_codes, max_per_split=None):
    """Load and tokenize HAR data for MLP training."""
    import pandas as pd

    splits = {}
    for split_name, csv_name in [
        ("train", "src/data/har_cot/har_cot_train_cot.csv"),
        ("val", "src/data/har_cot/har_cot_val_cot.csv"),
        ("test", "src/data/har_cot/har_cot_test_cot.csv"),
    ]:
        print(f"  Loading {split_name}...")
        nrows = max_per_split * 3 if max_per_split else None  # oversample for bad rows
        df = pd.read_csv(csv_name, nrows=nrows)

        features, labels = [], []
        for _, row in df.iterrows():
            if max_per_split and len(features) >= max_per_split:
                break
            label = str(row["label"]).strip()
            if label == "label":
                continue
            try:
                x = json.loads(row["x_axis"])
                y = json.loads(row["y_axis"])
                z = json.loads(row["z_axis"])
            except (json.JSONDecodeError, TypeError):
                continue

            # Tokenize each axis and combine histograms
            codes_x = tokenize_signal(tokenizer, x)
            codes_y = tokenize_signal(tokenizer, y)
            codes_z = tokenize_signal(tokenizer, z)

            hist = np.concatenate([
                codes_to_histogram(codes_x, n_codes),
                codes_to_histogram(codes_y, n_codes),
                codes_to_histogram(codes_z, n_codes),
            ])
            features.append(hist)

            # 6-class: merge walking variants
            if label in ("walking_up", "walking_down"):
                label = "walking"
            labels.append(label)

        splits[split_name] = (np.array(features), np.array(labels, dtype=str))
        print(f"    {split_name}: {len(features)} samples, {len(set(labels))} classes")

    return splits


def main():
    print("=" * 60)
    print("MLP Baseline on FSQ Codes — HAR 6-class")
    print("=" * 60)

    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer()
    n_codes = tokenizer.codebook_size
    print(f"  {n_codes} codes")

    print("\nLoading and tokenizing HAR data...")
    splits = load_har_data(tokenizer, n_codes, max_per_split=5000)

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    # Encode labels as integers (sklearn bug with np.str_)
    le = LabelEncoder()
    le.fit(list(set(y_train)))
    y_train = le.transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)

    print(f"\nFeature dim: {X_train.shape[1]} (3 axes × {n_codes} code bins)")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Classes: {list(le.classes_)}")

    # Train MLP
    print("\nTraining MLP...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False,
    )
    mlp.fit(X_train, y_train)
    print(f"  Converged in {mlp.n_iter_} iterations")

    # Evaluate
    y_pred_val = mlp.predict(X_val)
    y_pred_test = mlp.predict(X_test)

    val_acc = accuracy_score(y_val, y_pred_val)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1w = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
    test_f1m = f1_score(y_test, y_pred_test, average="macro", zero_division=0)

    print(f"\n{'=' * 60}")
    print(f"MLP BASELINE RESULTS (HAR 6-class)")
    print(f"{'=' * 60}")
    print(f"  Val Accuracy:      {val_acc:.1%}")
    print(f"  Test Accuracy:     {test_acc:.1%}")
    print(f"  Test F1 (weighted): {test_f1w:.3f}")
    print(f"  Test F1 (macro):   {test_f1m:.3f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=le.classes_, zero_division=0))

    # Cross-validation for error bars
    print("Running 5-fold cross-validation...")
    X_all = np.concatenate([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    cv_scores = cross_val_score(
        MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42),
        X_all, y_all, cv=5, scoring="accuracy",
    )
    print(f"  CV Accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

    print(f"\n{'=' * 60}")
    print(f"COMPARISON")
    print(f"{'=' * 60}")
    print(f"  MLP on FSQ codes:      {test_acc:.1%} (no LLM, no reasoning)")
    print(f"  TEMPO 4B (old):       86.5% (LLM + CoT reasoning)")
    print(f"  TEMPO 1.7B (aligned): TBD")
    print(f"  OpenTSLM Flamingo:     65 F1 (delta=0%, ignores signal)")
    print(f"  GPT-4o:                20% (text serialization)")


if __name__ == "__main__":
    main()
