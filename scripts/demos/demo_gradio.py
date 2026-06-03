"""Quick Gradio demo for TEMPO — test and compare checkpoints locally.

Usage:
    uv run --with gradio python tempo/demo_gradio.py \
        --checkpoint checkpoints/tempo/phase0_rope_imputation_10epochs/phase0_best.pt \
        --checkpoint-b checkpoints/phase1_v2_best.pt \
        --labels "Phase 0 (alignment only)" "Phase 1 (trained)" \
        --tokenizer-ckpt tempo/fsq_transformer_rope_625_best.pt

    # Then open http://localhost:7860
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def load_model(checkpoint, tokenizer_ckpt, tokenizer_type, llm_id, device):
    from tempo import TEMPO
    kwargs = {"use_dora": True, "device": device}
    if tokenizer_type == "totem":
        kwargs["totem_ckpt"] = tokenizer_ckpt
    else:
        kwargs["fsq_ckpt"] = tokenizer_ckpt
    model = TEMPO.from_pretrained(
        checkpoint,
        llm_id=llm_id,
        tokenizer_type=tokenizer_type,
        **kwargs,
    )
    return model


def generate_signal(signal_type: str, length: int = 256) -> np.ndarray:
    t = np.linspace(0, 4 * np.pi, length)
    if signal_type == "Sine wave":
        return np.sin(t)
    elif signal_type == "Trend up":
        return np.linspace(0, 3, length) + np.random.randn(length) * 0.2
    elif signal_type == "Trend down":
        return np.linspace(3, 0, length) + np.random.randn(length) * 0.2
    elif signal_type == "Spike":
        s = np.random.randn(length) * 0.3
        s[length // 2] = 5.0
        return s
    elif signal_type == "Step function":
        s = np.zeros(length)
        s[length // 2:] = 2.0
        return s + np.random.randn(length) * 0.1
    elif signal_type == "Noisy periodic":
        return np.sin(t) + np.sin(3 * t) * 0.5 + np.random.randn(length) * 0.3
    elif signal_type == "Random walk":
        return np.cumsum(np.random.randn(length) * 0.1)
    elif signal_type == "Flat":
        return np.ones(length) * 0.5 + np.random.randn(length) * 0.05
    else:
        return np.random.randn(length)


def get_signal(signal_type: str, csv_input: str) -> np.ndarray | None:
    if csv_input and csv_input.strip():
        try:
            values = [float(x.strip()) for x in csv_input.split(",") if x.strip()]
            return np.array(values, dtype=np.float32)
        except ValueError:
            return None
    return generate_signal(signal_type).astype(np.float32)


def make_plot(signal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(signal, linewidth=0.8)
    ax.set_title(f"Input signal ({len(signal)} pts, mean={signal.mean():.2f}, std={signal.std():.2f})")
    ax.set_xlabel("Time")
    plt.tight_layout()
    return fig


def run_model(model, signal, question, max_tokens):
    try:
        signal_t = torch.tensor(signal, dtype=torch.float32)
        output = model.analyze(
            signal_t,
            question=question + " /no_think",
            max_new_tokens=max_tokens,
        )
        return output
    except Exception as e:
        return f"Error: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Checkpoint A (e.g., phase 0)")
    parser.add_argument("--checkpoint-b", default=None, help="Checkpoint B (e.g., phase 1)")
    parser.add_argument("--labels", nargs=2, default=["Phase 0", "Phase 1"])
    parser.add_argument("--tokenizer-ckpt", required=True)
    parser.add_argument("--tokenizer-type", default="fsq_transformer_rope")
    parser.add_argument("--tokenizer-ckpt-b", default=None)
    parser.add_argument("--tokenizer-type-b", default=None)
    parser.add_argument("--llm-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print(f"Loading model A: {args.checkpoint}")
    model_a = load_model(args.checkpoint, args.tokenizer_ckpt,
                         args.tokenizer_type, args.llm_id, args.device)
    print(f"  {args.labels[0]} loaded")

    model_b = None
    if args.checkpoint_b:
        tok_type_b = args.tokenizer_type_b or args.tokenizer_type
        tok_ckpt_b = args.tokenizer_ckpt_b or args.tokenizer_ckpt
        print(f"Loading model B: {args.checkpoint_b} (tokenizer: {tok_type_b})")
        model_b = load_model(args.checkpoint_b, tok_ckpt_b,
                             tok_type_b, args.llm_id, args.device)
        print(f"  {args.labels[1]} loaded")

    import gradio as gr

    if model_b:
        # --- Two-model comparison mode ---
        def analyze_both(question, signal_type, csv_input, max_tokens):
            signal = get_signal(signal_type, csv_input)
            if signal is None:
                return "Error: bad CSV", "Error: bad CSV", None
            if len(signal) < 8:
                return "Signal too short", "Signal too short", None

            fig = make_plot(signal)
            out_a = run_model(model_a, signal, question, max_tokens)
            out_b = run_model(model_b, signal, question, max_tokens)
            return out_a, out_b, fig

        with gr.Blocks(title="TEMPO A/B Comparison") as demo:
            gr.Markdown("# TEMPO — Phase 0 vs Phase 1 Comparison")
            gr.Markdown(f"**A:** `{args.labels[0]}` — `{args.checkpoint}`  \n"
                        f"**B:** `{args.labels[1]}` — `{args.checkpoint_b}`  \n"
                        f"**Device:** `{args.device}`")

            with gr.Row():
                with gr.Column(scale=1):
                    signal_type = gr.Dropdown(
                        choices=["Sine wave", "Trend up", "Trend down", "Spike",
                                 "Step function", "Noisy periodic", "Random walk",
                                 "Flat", "Random noise"],
                        value="Sine wave",
                        label="Signal type",
                    )
                    csv_input = gr.Textbox(
                        label="Custom signal (CSV)",
                        placeholder="0.1, 0.3, 0.5, ...",
                        lines=2,
                    )
                    question = gr.Textbox(
                        label="Question",
                        value="Describe the key characteristics of this time series.",
                        lines=2,
                    )
                    max_tokens = gr.Slider(50, 500, value=200, step=50, label="Max tokens")
                    run_btn = gr.Button("Run both models", variant="primary")

                with gr.Column(scale=1):
                    plot = gr.Plot(label="Signal")

            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"### {args.labels[0]}")
                    output_a = gr.Textbox(label=args.labels[0], lines=10)
                with gr.Column():
                    gr.Markdown(f"### {args.labels[1]}")
                    output_b = gr.Textbox(label=args.labels[1], lines=10)

            gr.Examples(
                examples=[
                    ["Describe the key characteristics of this time series.", "Sine wave", ""],
                    ["Is there an anomaly in this signal?", "Spike", ""],
                    ["What is the overall trend?", "Trend up", ""],
                    ["How volatile is this signal?", "Random walk", ""],
                    ["Does this time series exhibit periodicity?", "Noisy periodic", ""],
                    ["Is this signal stable or changing?", "Step function", ""],
                ],
                inputs=[question, signal_type, csv_input],
            )

            run_btn.click(
                fn=analyze_both,
                inputs=[question, signal_type, csv_input, max_tokens],
                outputs=[output_a, output_b, plot],
            )

    else:
        # --- Single model mode ---
        def analyze(question, signal_type, csv_input, max_tokens):
            signal = get_signal(signal_type, csv_input)
            if signal is None:
                return "Error: bad CSV", None
            if len(signal) < 8:
                return "Signal too short", None

            fig = make_plot(signal)
            out = run_model(model_a, signal, question, max_tokens)
            return out, fig

        with gr.Blocks(title="TEMPO Demo") as demo:
            gr.Markdown("# TEMPO — Time Series Reasoning Demo")
            gr.Markdown(f"**Checkpoint:** `{args.checkpoint}`  \n"
                        f"**Device:** `{args.device}`")

            with gr.Row():
                with gr.Column(scale=1):
                    signal_type = gr.Dropdown(
                        choices=["Sine wave", "Trend up", "Trend down", "Spike",
                                 "Step function", "Noisy periodic", "Random walk",
                                 "Flat", "Random noise"],
                        value="Sine wave",
                        label="Signal type",
                    )
                    csv_input = gr.Textbox(
                        label="Custom signal (CSV)",
                        placeholder="0.1, 0.3, 0.5, ...",
                        lines=2,
                    )
                    question = gr.Textbox(
                        label="Question",
                        value="Describe the key characteristics of this time series.",
                        lines=2,
                    )
                    max_tokens = gr.Slider(50, 500, value=200, step=50, label="Max tokens")
                    run_btn = gr.Button("Analyze", variant="primary")

                with gr.Column(scale=1):
                    plot = gr.Plot(label="Signal")
                    output = gr.Textbox(label="Model output", lines=10)

            gr.Examples(
                examples=[
                    ["Describe the key characteristics of this time series.", "Sine wave", ""],
                    ["Is there an anomaly in this signal?", "Spike", ""],
                    ["What is the overall trend?", "Trend up", ""],
                    ["How volatile is this signal?", "Random walk", ""],
                    ["Does this time series exhibit periodicity?", "Noisy periodic", ""],
                    ["Is this signal stable or changing?", "Step function", ""],
                ],
                inputs=[question, signal_type, csv_input],
            )

            run_btn.click(
                fn=analyze,
                inputs=[question, signal_type, csv_input, max_tokens],
                outputs=[output, plot],
            )

    demo.launch(server_port=args.port, share=False)


if __name__ == "__main__":
    main()
