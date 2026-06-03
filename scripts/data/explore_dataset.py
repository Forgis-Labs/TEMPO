"""Interactive dataset explorer — view signals and associated text.

Browse phase 0/1 parquet datasets sample by sample. For each sample:
  - Plots the raw time series signal
  - Shows pre_prompt, post_prompt (question), answer, task, domain
  - Displays TS codes and stats
  - Filter by task type or domain

Usage:
    cd tempo
    uv run python explore_dataset.py                                          # default: phase0 val
    uv run python explore_dataset.py --parquet data/pretokenized_rope/phase0/train.parquet
    uv run python explore_dataset.py --parquet data/pretokenized_rope_v2/phase0/train.parquet
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# pip install gradio if needed
try:
    import gradio as gr
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gradio"])
    import gradio as gr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq


def load_parquet(path):
    table = pq.read_table(path)
    data = table.to_pydict()
    n = table.num_rows
    cols = table.column_names
    return data, n, cols


def plot_signal(signal_data, channel_labels=None):
    """Plot time series signal(s). One subplot per channel if multiple."""
    if not signal_data or (isinstance(signal_data, list) and len(signal_data) == 0):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No time series data", ha="center", va="center",
                fontsize=14, color="gray")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        return fig

    if isinstance(signal_data, list) and len(signal_data) > 0 and isinstance(signal_data[0], list):
        # Multi-channel: one subplot per channel
        n_ch = len(signal_data)
        fig, axes = plt.subplots(n_ch, 1, figsize=(10, 2.5 * n_ch), sharex=True)
        if n_ch == 1:
            axes = [axes]
        colors = ["steelblue", "coral", "seagreen", "mediumpurple",
                  "darkorange", "crimson", "teal", "olive"]
        for ch_idx, (ax, ch) in enumerate(zip(axes, signal_data)):
            vals = np.array(ch, dtype=np.float32)
            color = colors[ch_idx % len(colors)]
            label = (channel_labels[ch_idx] if channel_labels and ch_idx < len(channel_labels)
                     else f"Channel {ch_idx}")
            ax.plot(vals, linewidth=0.8, color=color)
            ax.set_ylabel(label, fontsize=9)
            ax.grid(alpha=0.2)
            # Show stats
            ax.text(0.99, 0.95, f"mean={vals.mean():.2f} std={vals.std():.2f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        axes[-1].set_xlabel("Time step", fontsize=9)
        plt.tight_layout()
        return fig
    elif isinstance(signal_data, list):
        # Single channel
        fig, ax = plt.subplots(figsize=(10, 3))
        vals = np.array(signal_data, dtype=np.float32)
        ax.plot(vals, linewidth=0.8, color="steelblue")
        ax.set_xlabel("Time step", fontsize=9)
        ax.set_ylabel("Value", fontsize=9)
        ax.grid(alpha=0.2)
        ax.text(0.99, 0.95, f"mean={vals.mean():.2f} std={vals.std():.2f} len={len(vals)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        plt.tight_layout()
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, f"Unknown format: {type(signal_data)}",
                ha="center", va="center")
        plt.tight_layout()
        return fig


def plot_codes(codes_data):
    """Plot TS codes as a bar/stem chart."""
    fig, ax = plt.subplots(figsize=(10, 2))

    if not codes_data:
        ax.text(0.5, 0.5, "No codes", ha="center", va="center",
                fontsize=12, color="gray")
        plt.tight_layout()
        return fig

    # Flatten if nested
    if isinstance(codes_data, list) and len(codes_data) > 0 and isinstance(codes_data[0], list):
        flat = [c for ch in codes_data for c in ch]
    else:
        flat = codes_data

    ax.bar(range(len(flat)), flat, width=1.0, color="coral", alpha=0.7, edgecolor="none")
    ax.set_xlabel("Code position", fontsize=8)
    ax.set_ylabel("Code ID", fontsize=8)
    ax.set_title(f"{len(flat)} codes, range [{min(flat)}, {max(flat)}]", fontsize=9)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    return fig


def parse_imputation_codes(pre_prompt):
    """Parse codes and mask positions from an imputation pre_prompt.

    Returns (code_values, is_masked) lists, or (None, None) if not imputation.
    """
    import re
    if "[MASKED]" not in pre_prompt:
        return None, None

    code_values = []
    is_masked = []

    # Find all <ts_N> tokens and [MASKED] markers in order
    pattern = r'(<ts_(\d+)>|\[MASKED\])'
    for match in re.finditer(pattern, pre_prompt):
        token = match.group(0)
        if token == "[MASKED]":
            code_values.append(-1)  # placeholder
            is_masked.append(True)
        else:
            code_values.append(int(match.group(2)))
            is_masked.append(False)

    return code_values, is_masked


def parse_answer_codes(answer):
    """Extract code values from an answer like '<ts_358><ts_237>'."""
    import re
    return [int(m.group(1)) for m in re.finditer(r'<ts_(\d+)>', answer)]


def plot_imputation(pre_prompt, answer):
    """Plot imputation sample: visible codes + masked positions + ground truth."""
    code_values, is_masked = parse_imputation_codes(pre_prompt)
    if code_values is None:
        return None

    gt_codes = parse_answer_codes(answer)

    fig, ax = plt.subplots(figsize=(10, 3))

    n = len(code_values)
    colors = []
    display_values = []

    gt_idx = 0
    for i in range(n):
        if is_masked[i]:
            colors.append("red")
            if gt_idx < len(gt_codes):
                display_values.append(gt_codes[gt_idx])
                gt_idx += 1
            else:
                display_values.append(0)
        else:
            colors.append("steelblue")
            display_values.append(code_values[i])

    # Plot visible codes
    ax.bar(range(n), display_values, width=1.0, color=colors, alpha=0.7, edgecolor="none")

    # Highlight masked region
    mask_positions = [i for i, m in enumerate(is_masked) if m]
    if mask_positions:
        ax.axvspan(mask_positions[0] - 0.5, mask_positions[-1] + 0.5,
                   alpha=0.15, color="red", label=f"Masked ({len(mask_positions)} codes)")

    ax.set_xlabel("Code position", fontsize=9)
    ax.set_ylabel("Code ID", fontsize=9)
    n_visible = sum(1 for m in is_masked if not m)
    ax.set_title(f"Imputation: {n_visible} visible (blue) + {len(mask_positions)} masked (red = ground truth)",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    return fig


def build_app(data, n, cols):
    """Build Gradio interface."""

    # Get unique tasks and domains for filtering
    tasks = sorted(set(data.get("task", ["?"] * n)))
    domains = sorted(set(data.get("domain", ["?"] * n)))

    # Build index by task
    task_indices = {}
    for i in range(n):
        t = data.get("task", ["?"] * n)[i]
        if t not in task_indices:
            task_indices[t] = []
        task_indices[t].append(i)

    def get_sample(idx, task_filter):
        idx = int(idx)

        # Apply task filter
        if task_filter and task_filter != "ALL":
            valid = task_indices.get(task_filter, [])
            if not valid:
                return None, None, "No samples for this task", "", "", "", "", ""
            idx = valid[idx % len(valid)]

        if idx < 0 or idx >= n:
            return None, None, "Index out of range", "", "", "", "", ""

        # Extract fields
        ts = data.get("time_series", [None] * n)[idx]
        codes = data.get("ts_codes", [None] * n)[idx]
        pre = data.get("pre_prompt", [""] * n)[idx] or ""
        post = data.get("post_prompt", [""] * n)[idx] or ""
        answer = data.get("answer", [""] * n)[idx] or ""
        task = data.get("task", ["?"] * n)[idx] or "?"
        domain = data.get("domain", ["?"] * n)[idx] or "?"
        source = data.get("source", ["?"] * n)[idx] or "?"

        # Stats
        ts_mean = data.get("ts_mean", [0] * n)[idx]
        ts_std = data.get("ts_std", [0] * n)[idx]

        # Signal plot
        ts_text = data.get("time_series_text", [None] * n)[idx]

        # Check if this is an imputation sample
        is_imputation = task == "code_imputation" or "[MASKED]" in pre

        if is_imputation:
            # For imputation: show masked code visualization instead of signal
            sig_fig = plot_imputation(pre, answer)
            if sig_fig is None:
                sig_fig = plot_signal(ts, channel_labels=ts_text)
            codes_fig = plot_codes(codes)
        else:
            sig_fig = plot_signal(ts, channel_labels=ts_text)
            codes_fig = plot_codes(codes)

        # Info string
        info = f"**Sample {idx}** / {n-1}\n\n"
        info += f"**Task:** {task}  |  **Domain:** {domain}  |  **Source:** {source}\n\n"
        if ts_mean or ts_std:
            info += f"**TS Stats:** mean={ts_mean:.4f}, std={ts_std:.4f}\n\n"
        if codes:
            flat = [c for ch in codes for c in ch] if isinstance(codes[0], list) else codes
            info += f"**Codes:** {len(flat)} tokens, range [{min(flat)}, {max(flat)}]\n\n"

        return sig_fig, codes_fig, info, pre, post, answer, task, domain

    with gr.Blocks(title="Dataset Explorer", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Phase 0 Dataset Explorer")
        gr.Markdown(f"**{n} samples** | Columns: {', '.join(cols)}")

        with gr.Row():
            idx_slider = gr.Slider(0, n - 1, value=0, step=1, label="Sample index")
            task_filter = gr.Dropdown(
                choices=["ALL"] + tasks, value="ALL", label="Filter by task"
            )

        with gr.Row():
            prev_btn = gr.Button("< Prev")
            next_btn = gr.Button("Next >")
            random_btn = gr.Button("Random")

        info_md = gr.Markdown()

        with gr.Row():
            sig_plot = gr.Plot(label="Time Series Signal")
            codes_plot = gr.Plot(label="TS Codes")

        with gr.Row():
            with gr.Column():
                pre_text = gr.Textbox(label="Pre-prompt (context)", lines=3)
                post_text = gr.Textbox(label="Post-prompt (question)", lines=2)
            with gr.Column():
                ans_text = gr.Textbox(label="Answer", lines=4)
                task_text = gr.Textbox(label="Task")
                domain_text = gr.Textbox(label="Domain")

        def update(idx, tf):
            sig, codes, info, pre, post, ans, task, dom = get_sample(idx, tf)
            return sig, codes, info, pre, post, ans, task, dom

        outputs = [sig_plot, codes_plot, info_md, pre_text, post_text,
                   ans_text, task_text, domain_text]

        idx_slider.change(update, [idx_slider, task_filter], outputs)
        task_filter.change(update, [idx_slider, task_filter], outputs)

        def go_prev(idx, tf):
            return max(0, int(idx) - 1)
        def go_next(idx, tf):
            return min(n - 1, int(idx) + 1)
        def go_random(idx, tf):
            import random
            return random.randint(0, n - 1)

        prev_btn.click(go_prev, [idx_slider, task_filter], idx_slider)
        next_btn.click(go_next, [idx_slider, task_filter], idx_slider)
        random_btn.click(go_random, [idx_slider, task_filter], idx_slider)

        # Load first sample on start
        app.load(update, [idx_slider, task_filter], outputs)

    return app


def main():
    parser = argparse.ArgumentParser(description="Explore training dataset")
    parser.add_argument("--parquet", default=None,
                        help="Path to parquet file. Default: auto-detect.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create public Gradio link")
    args = parser.parse_args()

    # Auto-detect parquet
    if args.parquet is None:
        candidates = [
            "data/pretokenized_rope/phase0/validation.parquet",
            "data/pretokenized_rope/phase0/train.parquet",
            "data/pretokenized_rope_v2/phase0/validation.parquet",
            "../data/phase0_val_sample.parquet",
        ]
        for c in candidates:
            if os.path.exists(c):
                args.parquet = c
                break
        if args.parquet is None:
            print("No parquet found. Specify with --parquet <path>")
            print("Or download: aws s3 cp s3://.../tempo/phase0/validation.parquet data/phase0_val.parquet")
            sys.exit(1)

    print(f"Loading {args.parquet}...")
    data, n, cols = load_parquet(args.parquet)
    print(f"  {n} samples, columns: {cols}")

    app = build_app(data, n, cols)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
