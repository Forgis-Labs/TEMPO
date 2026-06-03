"""Inference CLI for TEMPO.

Usage:
    # Analyze a signal from a numpy file:
    python -m tempo --checkpoint results/best_model.pt --totem_ckpt totem.pt \
        --signal data/sample.npy --question "What is the trend?"

    # Forecast future values:
    python -m tempo --checkpoint results/best_model.pt --totem_ckpt totem.pt \
        --signal data/sample.npy --forecast --max_codes 64

    # Interactive mode:
    python -m tempo --checkpoint results/best_model.pt --totem_ckpt totem.pt --interactive
"""

import argparse
import sys

import numpy as np
import torch

from tempo import TEMPO


def main():
    p = argparse.ArgumentParser(
        prog="python -m tempo",
        description="Run TEMPO inference on time series signals.",
    )

    # Model
    p.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    p.add_argument("--totem_ckpt", default=None, help="Path to TOTEM checkpoint")
    p.add_argument("--llm_id", default="Qwen/Qwen3-4B", help="Base LLM model ID")
    p.add_argument("--device", default="cuda", help="Device (cuda or cpu)")

    # Input
    p.add_argument("--signal", default=None,
                   help="Path to signal file (.npy, .csv, or .pt)")
    p.add_argument("--question", default="Describe the patterns in this signal.",
                   help="Question to ask about the signal")
    p.add_argument("--context", default="",
                   help="Additional context for the analysis")

    # Mode
    p.add_argument("--forecast", action="store_true",
                   help="Generate forecast instead of analysis")
    p.add_argument("--max_codes", type=int, default=64,
                   help="Max forecast codes to generate")
    p.add_argument("--max_new_tokens", type=int, default=400,
                   help="Max tokens for analysis generation")
    p.add_argument("--interactive", action="store_true",
                   help="Interactive mode: ask multiple questions")

    args = p.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = TEMPO.from_pretrained(
        args.checkpoint,
        totem_ckpt=args.totem_ckpt,
        llm_id=args.llm_id,
        device=args.device,
    )
    print("Model loaded.\n")

    if args.interactive:
        _interactive_mode(model, args)
    elif args.signal:
        signal = _load_signal(args.signal)
        if args.forecast:
            _run_forecast(model, signal, args)
        else:
            _run_analysis(model, signal, args)
    else:
        p.error("Provide --signal or use --interactive mode")


def _load_signal(path: str) -> torch.Tensor:
    """Load a signal from file (.npy, .csv, .pt)."""
    if path.endswith(".npy"):
        data = np.load(path)
        return torch.tensor(data, dtype=torch.float32).flatten()
    elif path.endswith(".pt"):
        return torch.load(path, map_location="cpu").float().flatten()
    elif path.endswith(".csv"):
        data = np.loadtxt(path, delimiter=",")
        return torch.tensor(data, dtype=torch.float32).flatten()
    else:
        # Try numpy
        data = np.load(path)
        return torch.tensor(data, dtype=torch.float32).flatten()


def _run_analysis(model, signal: torch.Tensor, args):
    """Run analysis on a signal."""
    print(f"Signal: {len(signal)} points, "
          f"mean={signal.mean():.4f}, std={signal.std():.4f}")
    print(f"Question: {args.question}\n")

    answer = model.analyze(
        signal,
        question=args.question,
        context=args.context,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Answer:\n{answer}")


def _run_forecast(model, signal: torch.Tensor, args):
    """Run forecast on a signal."""
    print(f"Signal: {len(signal)} points")
    print(f"Generating {args.max_codes} forecast codes...\n")

    result = model.forecast(signal, max_codes=args.max_codes)
    print(f"Forecast codes ({len(result['codes'])}): {result['codes']}")
    if "values" in result and result["values"] is not None:
        print(f"Decoded values: {result['values'][:16].tolist()}")
    print(f"\nRaw output: {result['raw_text'][:200]}")


def _interactive_mode(model, args):
    """Interactive loop: load signal once, ask multiple questions."""
    signal = None

    print("TEMPO interactive mode. Commands:")
    print("  load <path>   — Load a signal file")
    print("  ask <question> — Analyze current signal")
    print("  forecast      — Forecast from current signal")
    print("  quit          — Exit\n")

    while True:
        try:
            line = input("tempo> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line.startswith("quit") or line.startswith("exit"):
            break
        elif line.startswith("load "):
            path = line[5:].strip()
            try:
                signal = _load_signal(path)
                print(f"Loaded: {len(signal)} points, "
                      f"mean={signal.mean():.4f}, std={signal.std():.4f}")
            except Exception as e:
                print(f"Error loading: {e}")
        elif line.startswith("ask "):
            if signal is None:
                print("No signal loaded. Use: load <path>")
                continue
            question = line[4:].strip()
            answer = model.analyze(signal, question=question,
                                   max_new_tokens=args.max_new_tokens)
            print(f"\n{answer}\n")
        elif line == "forecast":
            if signal is None:
                print("No signal loaded. Use: load <path>")
                continue
            result = model.forecast(signal, max_codes=args.max_codes)
            print(f"Codes: {result['codes']}")
            if "values" in result and result["values"] is not None:
                print(f"Values: {result['values'][:16].tolist()}")
        else:
            # Treat bare text as a question
            if signal is not None:
                answer = model.analyze(signal, question=line,
                                       max_new_tokens=args.max_new_tokens)
                print(f"\n{answer}\n")
            else:
                print("No signal loaded. Use: load <path>")


if __name__ == "__main__":
    main()
