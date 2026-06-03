"""Train FSQ Transformer tokenizer.

Transformer encoder gives global context before FSQ quantization,
producing semantically meaningful codes. Temporal smoothness loss
encourages adjacent codes to be similar.

Usage:
    python -m tempo.tokenizer.train_fsq_transformer
    python -m tempo.tokenizer.train_fsq_transformer --levels 5 5 5 5 --d_model 192 --epochs 100
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

from .fsq_transformer import FSQTransformerTokenizer, FSQTransformerConfig

INPUT_LENGTH = 256


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


class TSDataset(Dataset):
    def __init__(self, signals):
        self.signals = signals

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        x = np.asarray(self.signals[idx], dtype=np.float32).flatten()
        if len(x) != INPUT_LENGTH:
            x = np.interp(np.linspace(0, len(x) - 1, INPUT_LENGTH),
                          np.arange(len(x)), x)
        std = x.std()
        if std > 1e-8:
            x = (x - x.mean()) / std
        return torch.tensor(x, dtype=torch.float32)


def collect_signals(max_total=500000):
    """Collect diverse time series — reuse from train_fsq_tokenizer."""
    # Import the collection function from the old training script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    try:
        from train_fsq_tokenizer import collect_signals as _collect
        return _collect(max_total)
    except ImportError:
        print("WARNING: could not import collect_signals from train_fsq_tokenizer")
        print("Falling back to synthetic data only")
        rng = np.random.RandomState(42)
        signals = []
        for _ in range(max_total):
            kind = rng.choice(["sine", "chirp", "ar", "noise", "damped", "modulated"])
            t = np.linspace(0, 1, INPUT_LENGTH)
            if kind == "sine":
                sig = np.sin(2 * np.pi * rng.uniform(1, 20) * t + rng.uniform(0, 6.28))
                sig += rng.randn(INPUT_LENGTH) * 0.1
            elif kind == "chirp":
                f0, f1 = rng.uniform(1, 5), rng.uniform(10, 30)
                sig = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / 2) * t)
            elif kind == "ar":
                sig = np.zeros(INPUT_LENGTH)
                a = rng.uniform(0.5, 0.99)
                for i in range(1, INPUT_LENGTH):
                    sig[i] = a * sig[i - 1] + rng.randn() * 0.3
            elif kind == "noise":
                sig = rng.randn(INPUT_LENGTH)
            elif kind == "damped":
                sig = np.exp(-rng.uniform(1, 5) * t) * np.sin(2 * np.pi * rng.uniform(3, 15) * t)
            elif kind == "modulated":
                fc, fm = rng.uniform(5, 20), rng.uniform(0.5, 3)
                sig = np.sin(2 * np.pi * fc * t) * (1 + 0.5 * np.sin(2 * np.pi * fm * t))
            if np.std(sig) > 1e-12:
                signals.append(sig.astype(np.float64))
        return signals


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = FSQTransformerConfig(
        levels=args.levels,
        patch_size=args.patch_size,
        input_length=INPUT_LENGTH,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_layers,
        n_decoder_layers=args.n_layers,
        d_ff=args.d_model * 4,
        dropout=args.dropout,
        temporal_weight=args.temporal_weight,
    )

    print("=" * 60)
    print(f"FSQ Transformer Tokenizer")
    print(f"  Codes: {config.codebook_size} (levels={config.levels})")
    print(f"  Patch size: {config.patch_size} → {config.n_patches} patches per {INPUT_LENGTH}-pt signal")
    print(f"  Transformer: d={config.d_model}, heads={config.n_heads}, "
          f"layers={config.n_encoder_layers}+{config.n_decoder_layers}")
    print(f"  Temporal weight: {config.temporal_weight}")
    print("=" * 60)

    # Data
    signals = collect_signals(args.max_signals)
    n_val = max(500, len(signals) // 10)
    train_ds = TSDataset(signals[n_val:])
    val_ds = TSDataset(signals[:n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Model
    model = FSQTransformerTokenizer(config).to(device)
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
        for batch in train_loader:
            x = batch.to(device)
            recon, indices, z_pre = model(x)

            # Reconstruction losses
            mse = F.mse_loss(recon, x)
            spec = spectral_loss(recon, x)

            # Temporal smoothness on pre-quantization latents
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

        # Validate
        model.eval()
        vl, vl_mse, unique_codes = 0, 0, set()
        self_transitions, total_transitions = 0, 0
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

                # Track self-transitions
                for seq in flat.cpu().numpy():
                    for i in range(1, len(seq)):
                        total_transitions += 1
                        if seq[i] == seq[i - 1]:
                            self_transitions += 1

        vl /= len(val_loader)
        vl_mse /= len(val_loader)
        self_trans_pct = self_transitions / total_transitions * 100 if total_transitions > 0 else 0

        scheduler.step()

        print(f"  Epoch {epoch:3d}: loss={tl:.4f} (mse={tl_mse:.4f} spec={tl_spec:.4f} temp={tl_temp:.4f}) "
              f"val={vl:.4f} codes={len(unique_codes):,}/{config.codebook_size:,} "
              f"self_trans={self_trans_pct:.1f}%")
        sys.stdout.flush()

        # Save best
        if vl < best_val - 1e-6:
            best_val = vl
            patience_left = args.patience
            save_path = args.output
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "levels": config.levels,
                    "patch_size": config.patch_size,
                    "input_length": INPUT_LENGTH,
                    "compression": config.patch_size,  # patch_size = compression ratio
                    "d_model": config.d_model,
                    "n_heads": config.n_heads,
                    "n_encoder_layers": config.n_encoder_layers,
                    "n_decoder_layers": config.n_decoder_layers,
                    "d_ff": config.d_ff,
                    "dropout": config.dropout,
                    "temporal_weight": config.temporal_weight,
                },
                "epoch": epoch,
                "val_loss": vl,
                "unique_codes": len(unique_codes),
                "self_transition_pct": self_trans_pct,
            }, save_path)
            print(f"    -> Saved best (codes={len(unique_codes)}, self_trans={self_trans_pct:.1f}%)")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"Done. Best val={best_val:.4f}")


def main():
    p = argparse.ArgumentParser(description="Train FSQ Transformer tokenizer")
    p.add_argument("--levels", type=int, nargs="+", default=[7, 7, 7, 7])
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
    p.add_argument("--output", type=str, default="fsq_transformer_best.pt")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
