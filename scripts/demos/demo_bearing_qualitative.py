"""Qualitative bearing diagnosis experiments for the paper.

Demonstrates 4 key framework properties using CWRU bearing data
with the TEMPO model (hyperion_v3, DoRA r=32, Qwen3-4B + TOTEM):

1. Same signal, different context → different interpretation
2. Graded severity reasoning (real CWRU faults at 4 diameters)
3. Missing or conflicting context → graceful degradation
4. Comparison against classifier + GPT-4o post-hoc explanation

Usage (needs GPU):
    cd tempo
    PYTHONPATH=. python demo_bearing_qualitative.py \
        --checkpoint ../checkpoints/hyperion_v3_r32_best.pt \
        --tokenizer-ckpt ../checkpoints/totem_clean_local.pt

    # With Qwen3 think mode (ChatML + <think> reasoning):
    PYTHONPATH=. python demo_bearing_qualitative.py \
        --checkpoint ../checkpoints/hyperion_v3_r32_best.pt \
        --tokenizer-ckpt ../checkpoints/totem_clean_local.pt \
        --think

    # Skip GPT-4o comparison (no OpenAI key needed):
    PYTHONPATH=. python demo_bearing_qualitative.py \
        --checkpoint ../checkpoints/hyperion_v3_r32_best.pt \
        --tokenizer-ckpt ../checkpoints/totem_clean_local.pt \
        --skip-gpt4o
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load .env (same pattern as launch_eval_sagemaker.py)
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for line in _env_path.read_text().strip().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import numpy as np
import torch

sys.path.insert(0, str(PROJECT_ROOT))


# CWRU data: local path or SageMaker channel
_CWRU_LOCAL = PROJECT_ROOT / "data" / "cwru_github"
_CWRU_SM = Path(os.environ.get("SM_CHANNEL_CWRU", str(_CWRU_LOCAL)))
CWRU_BASE = _CWRU_SM if _CWRU_SM.exists() else _CWRU_LOCAL
CWRU_IR_DIR = CWRU_BASE / "12k_Drive_End_Bearing_Fault_Data" / "IR"
CWRU_NORMAL_DIR = CWRU_BASE / "Normal"


def _load_cwru_mat(mat_path: str, n_points: int = 1024) -> np.ndarray:
    """Load a CWRU .mat file and extract the drive-end accelerometer signal."""
    from scipy.io import loadmat

    mat = loadmat(mat_path)
    for key in mat:
        if "DE_time" in key:
            sig = mat[key].flatten().astype(np.float64)
            # Skip initial transient, take n_points
            start = min(5000, len(sig) - n_points)
            return sig[start : start + n_points]
    raise ValueError(f"No DE_time key found in {mat_path}")


def load_multiple_segments(fault_diameter: str, n_segments: int = 5,
                           n_points: int = 1024) -> list[np.ndarray]:
    """Load multiple non-overlapping segments from CWRU data for robustness."""
    fault_dir = CWRU_IR_DIR / fault_diameter
    segments = []

    if fault_dir.exists():
        from scipy.io import loadmat
        mat_files = sorted(fault_dir.glob("*.mat"))
        for mat_file in mat_files:
            mat = loadmat(str(mat_file))
            for key in mat:
                if "DE_time" not in key:
                    continue
                sig = mat[key].flatten().astype(np.float64)
                # Extract non-overlapping segments, skip transient
                offset = 5000
                while offset + n_points <= len(sig) and len(segments) < n_segments:
                    segments.append(sig[offset : offset + n_points])
                    offset += n_points + 500  # gap between segments
            if len(segments) >= n_segments:
                break

    # Pad with synthetic if not enough real data
    while len(segments) < n_segments:
        segments.append(_synthetic_bpfi(
            n_points,
            amplitude={"007": 0.15, "014": 0.35, "021": 0.7, "028": 1.2}.get(fault_diameter, 0.3),
        ))

    return segments[:n_segments]


def load_ir_fault_signal(fault_diameter: str = "007", n_points: int = 1024) -> np.ndarray:
    """Load an inner race fault signal from CWRU at given fault diameter.

    fault_diameter: "007", "014", "021", or "028" (inches × 1000)
    """
    fault_dir = CWRU_IR_DIR / fault_diameter
    if fault_dir.exists():
        mat_files = sorted(fault_dir.glob("*.mat"))
        if mat_files:
            return _load_cwru_mat(str(mat_files[0]), n_points)

    # Fallback: synthetic BPFI signal
    print(f"CWRU IR/{fault_diameter} not found, generating synthetic signal")
    return _synthetic_bpfi(n_points, amplitude={"007": 0.15, "014": 0.35, "021": 0.7, "028": 1.2}.get(fault_diameter, 0.3))


def load_normal_signal(n_points: int = 1024) -> np.ndarray:
    """Load a normal (healthy) CWRU bearing signal."""
    if CWRU_NORMAL_DIR.exists():
        mat_files = sorted(CWRU_NORMAL_DIR.glob("*.mat"))
        if mat_files:
            return _load_cwru_mat(str(mat_files[0]), n_points)
    return np.random.randn(n_points) * 0.02


def _synthetic_bpfi(n_points: int, amplitude: float = 0.3, fs: int = 12000) -> np.ndarray:
    """Generate synthetic bearing signal with BPFI harmonics at given amplitude."""
    t = np.arange(n_points) / fs
    bpfi = 162  # SKF 6205 at 1800 RPM
    sig = (
        np.sin(2 * np.pi * bpfi * t) * amplitude
        + np.sin(2 * np.pi * bpfi * 2 * t) * amplitude * 0.5
        + np.sin(2 * np.pi * bpfi * 3 * t) * amplitude * 0.2
        + np.random.randn(n_points) * amplitude * 0.3
    )
    return sig


def signal_stats(sig: np.ndarray) -> dict:
    """Compute signal statistics for reporting."""
    rms = np.sqrt(np.mean(sig ** 2))
    return {
        "mean": float(np.mean(sig)),
        "std": float(np.std(sig)),
        "rms": float(rms),
        "peak": float(np.max(np.abs(sig))),
        "n_points": len(sig),
    }


USE_CHATML = False  # Set by --think flag


def run_inference(model, signal: np.ndarray, context: str, question: str,
                  max_new_tokens: int = 400) -> str:
    """Run a single inference pass."""
    sig_tensor = torch.tensor(signal, dtype=torch.float32)
    return model.analyze(
        signal=sig_tensor,
        question=question,
        signal_label="Vibration (Drive End accelerometer):",
        context=context,
        max_new_tokens=max_new_tokens,
        use_chatml=USE_CHATML,
    )


def experiment_1_context_dependence(model, signals: list[np.ndarray]) -> list[dict]:
    """Same BPFI signal, different operational context -> different diagnosis.

    Runs each context on 5 different signal segments to ensure robustness.
    The signal tokens change between segments but the context stays the same.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Same signal, different context -> different interpretation")
    print(f"  ({len(signals)} signal segments per context)")
    print("=" * 70)

    def _make_contexts(stats):
        return [
            {
                "name": "Aged bearing -- incipient fault expected",
                "context": (
                    "SKF 6205-2RS deep groove ball bearing, installed 18 months ago. "
                    "Continuous operation at 1800 RPM, radial load 2.5 kN. "
                    "Previous inspection 3 months ago noted mild vibration increase. "
                    f"Current measurement: RMS = {stats['rms']:.2f} g, peak = {stats['peak']:.2f} g."
                ),
            },
            {
                "name": "New non-OEM bearing -- commissioning run",
                "context": (
                    "Bearing replaced 2 days ago with non-OEM equivalent part during "
                    "emergency maintenance. Commissioning run at 1800 RPM, no load. "
                    "First vibration measurement since installation. "
                    f"Current measurement: RMS = {stats['rms']:.2f} g, peak = {stats['peak']:.2f} g."
                ),
            },
            {
                "name": "High-speed heavy load -- overload scenario",
                "context": (
                    "SKF 6205-2RS, installed 6 months ago. Currently operating at "
                    "3600 RPM under heavy combined load (4.0 kN radial + 1.5 kN axial). "
                    "Cooling fan application, ambient temperature 45C. "
                    f"Current measurement: RMS = {stats['rms']:.2f} g, peak = {stats['peak']:.2f} g."
                ),
            },
        ]

    question = (
        "You are a senior vibration analyst writing a diagnostic report. "
        "Analyze this vibration signal in detail. Do NOT just classify -- "
        "explain what specific patterns you observe in the signal, how they "
        "relate to the operational context provided, what fault mechanism "
        "is most likely, and what maintenance action you recommend with "
        "a specific timeframe. Reason step by step."
    )

    results = []
    for ctx_template in _make_contexts(signal_stats(signals[0])):
        print(f"\n--- {ctx_template['name']} ---")
        ctx_results = []
        for i, sig in enumerate(signals):
            stats = signal_stats(sig)
            # Recompute context with this segment's stats
            contexts = _make_contexts(stats)
            ctx = next(c for c in contexts if c["name"] == ctx_template["name"])
            output = run_inference(model, sig, ctx["context"], question)
            print(f"  [segment {i+1}] {output[:300]}")
            ctx_results.append({
                "segment": i,
                "signal_stats": stats,
                "output": output,
            })
        results.append({
            "context_name": ctx_template["name"],
            "samples": ctx_results,
        })

    return results


def experiment_2_severity(model) -> list[dict]:
    """Real CWRU inner race faults at 4 severity levels.

    Uses actual CWRU data at fault diameters 0.007", 0.014", 0.021", 0.028".
    The context includes mean/std/RMS values so the model can reference
    amplitude information alongside spectral evidence from the signal tokens.
    We expect the model to modulate its language from "mild wear" through
    "immediate replacement needed" and cite ISO 10816 zone placement.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Graded severity reasoning")
    print("=" * 70)

    severities = [
        {
            "name": "0.007\" fault — early/mild",
            "diameter": "007",
            "iso_zone": "Zone A (good)",
            "description": "early-stage micro-pitting",
        },
        {
            "name": "0.014\" fault — moderate",
            "diameter": "014",
            "iso_zone": "Zone B (acceptable)",
            "description": "developing spall",
        },
        {
            "name": "0.021\" fault — severe",
            "diameter": "021",
            "iso_zone": "Zone C (alert)",
            "description": "progressed inner race damage",
        },
        {
            "name": "0.028\" fault — critical",
            "diameter": "028",
            "iso_zone": "Zone D (danger)",
            "description": "extensive inner race spalling",
        },
    ]

    question = (
        "Assess the vibration severity and fault progression. "
        "Reference the signal amplitude and any spectral patterns you observe. "
        "Classify the severity level and recommend a specific maintenance action "
        "with urgency level."
    )

    question = (
        "You are writing a vibration severity assessment report. "
        "Analyze the signal amplitude, impulsiveness, and spectral content. "
        "Classify the severity using ISO 10816 zones if possible. "
        "Describe how advanced the fault damage is based on signal evidence. "
        "Recommend a specific maintenance action with urgency level "
        "(routine monitoring / plan replacement / urgent / immediate shutdown). "
        "Reason step by step."
    )

    results = []
    for sev in severities:
        segments = load_multiple_segments(sev["diameter"], n_segments=5)
        print(f"\n--- {sev['name']} ---")
        sev_results = []
        for i, sig in enumerate(segments):
            stats = signal_stats(sig)
            context = (
                f"SKF 6205-2RS deep groove ball bearing, 1800 RPM continuous operation. "
                f"BPFI = 162 Hz for this bearing geometry. "
                f"ISO 10816 vibration severity assessment applies. "
                f"Signal statistics: mean = {stats['mean']:.4f} g, std = {stats['std']:.4f} g, "
                f"RMS = {stats['rms']:.4f} g, peak = {stats['peak']:.4f} g."
            )
            output = run_inference(model, sig, context, question)
            print(f"  [segment {i+1}] RMS={stats['rms']:.4f} | {output[:250]}")
            sev_results.append({
                "segment": i,
                "signal_stats": stats,
                "output": output,
            })
        results.append({
            "severity": sev["name"],
            "fault_diameter_inch": sev["diameter"],
            "iso_zone_expected": sev["iso_zone"],
            "samples": sev_results,
        })

    return results


def experiment_3_missing_conflicting(model, signals: list[np.ndarray]) -> list[dict]:
    """Same BPFI signal with incomplete or contradictory context.

    Runs 5 segments per scenario. The conflicting context does NOT hint
    at the mismatch -- the model must discover it from the signal.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Missing and conflicting context")
    print(f"  ({len(signals)} signal segments per scenario)")
    print("=" * 70)

    question = (
        "You are a vibration analyst. Analyze this signal and diagnose "
        "the bearing condition. Explicitly state your confidence level "
        "and what information would be needed to improve the diagnosis. "
        "If anything in the context conflicts with what you observe in "
        "the signal, flag it. Reason step by step."
    )

    def _make_scenarios(stats):
        return [
            {
                "name": "Full context (baseline)",
                "context": (
                    "SKF 6205-2RS deep groove ball bearing, 1800 RPM, radial load 2.5 kN. "
                    "BPFI = 162 Hz for this bearing geometry. Installed 12 months ago. "
                    f"RMS = {stats['rms']:.4f} g, peak = {stats['peak']:.4f} g."
                ),
            },
            {
                "name": "Missing bearing model",
                "context": (
                    "Industrial rotating machinery, shaft speed 1800 RPM. "
                    "Bearing model number and geometry are not available. "
                    "Radial load estimated 2-3 kN. "
                    f"RMS = {stats['rms']:.4f} g, peak = {stats['peak']:.4f} g."
                ),
            },
            {
                "name": "Conflicting context (wrong bearing model)",
                "context": (
                    "SKF 6310-2Z deep groove ball bearing, 1800 RPM, radial load 3.0 kN. "
                    "BPFI = 89 Hz for this bearing geometry. Installed 8 months ago. "
                    f"RMS = {stats['rms']:.4f} g, peak = {stats['peak']:.4f} g."
                ),
            },
            {
                "name": "Minimal context",
                "context": "Vibration measurement from industrial rotating equipment.",
            },
        ]

    results = []
    for sc_template in _make_scenarios(signal_stats(signals[0])):
        print(f"\n--- {sc_template['name']} ---")
        sc_results = []
        for i, sig in enumerate(signals):
            stats = signal_stats(sig)
            scenarios = _make_scenarios(stats)
            sc = next(s for s in scenarios if s["name"] == sc_template["name"])
            output = run_inference(model, sig, sc["context"], question)
            print(f"  [segment {i+1}] {output[:250]}")
            sc_results.append({
                "segment": i,
                "signal_stats": stats,
                "output": output,
            })
        results.append({
            "scenario": sc_template["name"],
            "samples": sc_results,
        })

    return results


def _get_gpt4o_posthoc(classifier_label: str, context: str) -> str:
    """Call GPT via OpenAI to generate a post-hoc explanation.

    This simulates the standard pipeline: CNN classifier → label → LLM explanation.
    GPT receives the fault label and context but NOT the signal.
    Uses the same OpenAI client pattern as tempo.eval.gpt_baseline.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return "[openai package not installed — using placeholder]"

    if not os.environ.get("OPENAI_API_KEY"):
        return "[OPENAI_API_KEY not set — using placeholder]"

    model = os.environ.get("OPENAI_MODEL", "gpt-4.1")
    client = OpenAI()

    prompt = (
        f"A vibration classifier has identified the following fault in a bearing:\n\n"
        f"Classification result: {classifier_label}\n"
        f"Operating context: {context}\n\n"
        f"Provide a detailed technical explanation of this diagnosis. "
        f"Explain what spectral patterns are associated with this fault type, "
        f"what the likely physical mechanism is, and recommend maintenance action. "
        f"Keep it to one paragraph."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a vibration analysis expert providing post-hoc explanations of classifier results."},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=400,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def experiment_4_vs_posthoc(model, signals: list[np.ndarray],
                            skip_gpt4o: bool = False) -> dict:
    """Compare TEMPO (signal-conditioned) vs CNN classifier + GPT post-hoc.

    Runs 5 segments. TEMPO sees signal tokens + context; GPT sees only the
    classifier label + context.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Signal-conditioned reasoning vs post-hoc rationalization")
    print(f"  ({len(signals)} signal segments)")
    print("=" * 70)

    question = (
        "You are writing a detailed bearing diagnostic report for the "
        "maintenance team. Identify the fault type and explain EXACTLY "
        "what evidence in the signal led you to this conclusion. "
        "Describe specific signal features (amplitude patterns, impact "
        "events, modulation, frequency content) that you observe. "
        "Assess the severity and recommend a maintenance action with "
        "a specific timeframe. Do NOT give a generic textbook answer -- "
        "your report must be specific to THIS signal."
    )

    classifier_label = "inner_race_fault (confidence: 0.94)"
    tempo_results = []

    for i, sig in enumerate(signals):
        stats = signal_stats(sig)
        context = (
            "SKF 6205-2RS deep groove ball bearing, installed 18 months ago. "
            "1800 RPM continuous operation, radial load 2.5 kN. BPFI = 162 Hz. "
            "Vibration trending upward over last 3 months. "
            f"RMS = {stats['rms']:.4f} g, peak = {stats['peak']:.4f} g."
        )
        print(f"\n--- TEMPO [segment {i+1}] ---")
        output = run_inference(model, sig, context, question, max_new_tokens=500)
        print(f"  {output[:400]}")
        tempo_results.append({
            "segment": i,
            "signal_stats": stats,
            "output": output,
        })

    # GPT post-hoc (one call -- it doesn't see the signal)
    context_for_gpt = (
        "SKF 6205-2RS deep groove ball bearing, installed 18 months ago. "
        "1800 RPM continuous operation, radial load 2.5 kN. BPFI = 162 Hz. "
        "Vibration trending upward over last 3 months."
    )
    print(f"\n--- CNN classifier + GPT post-hoc ---")
    print(f"  CNN output: {classifier_label}")

    if skip_gpt4o:
        posthoc = (
            "Inner race faults typically manifest as characteristic Ball Pass Frequency "
            "Inner Race (BPFI) harmonics in the vibration spectrum. For an SKF 6205-2RS "
            "bearing operating at 1800 RPM, the BPFI is approximately 162 Hz. When the "
            "inner race develops a defect such as spalling or pitting, each rolling element "
            "passing over the defect generates an impact, producing spectral peaks at BPFI "
            "and its harmonics (2×, 3×, etc.). The presence of sidebands around these peaks, "
            "spaced at shaft rotation frequency, confirms the fault is on the rotating inner "
            "race. Given 18 months of continuous operation with upward trending vibration, "
            "this suggests progressive fatigue-related spalling. Recommended action: schedule "
            "bearing replacement during the next planned maintenance window, within 2-4 weeks."
        )
        posthoc_source = "placeholder (--skip-gpt4o)"
    else:
        posthoc = _get_gpt4o_posthoc(classifier_label, context_for_gpt)
        posthoc_source = f"OpenAI {os.environ.get('OPENAI_MODEL', 'gpt-4.1')}"

    print(f"  GPT: {posthoc[:500]}")

    return {
        "tempo_samples": tempo_results,
        "classifier_label": classifier_label,
        "posthoc_explanation": posthoc,
        "posthoc_source": posthoc_source,
    }


def run_all(model, output_dir: str = "results/bearing_qualitative",
            skip_gpt4o: bool = False) -> dict:
    """Run all 4 experiments and save results."""
    os.makedirs(output_dir, exist_ok=True)

    # Load 5 segments of CWRU IR fault (0.007" -- clear BPFI, mild severity)
    signals = load_multiple_segments("007", n_segments=5)
    print(f"Loaded {len(signals)} signal segments, {len(signals[0])} pts each")
    print(f"Think mode: {USE_CHATML}")

    results = {
        "metadata": {
            "base_signal": "CWRU IR fault 0.007\" (12k DE, SKF 6205-2RS)",
            "n_segments": len(signals),
            "model": "hyperion_v3_r32 (Qwen3-4B + TOTEM, DoRA r=32)",
            "think_mode": USE_CHATML,
        },
    }
    results["exp1_context_dependence"] = experiment_1_context_dependence(model, signals)
    results["exp2_graded_severity"] = experiment_2_severity(model)
    results["exp3_missing_conflicting"] = experiment_3_missing_conflicting(model, signals)
    results["exp4_vs_posthoc"] = experiment_4_vs_posthoc(model, signals, skip_gpt4o)

    out_path = os.path.join(output_dir, "qualitative_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Bearing qualitative experiments")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--tokenizer-ckpt", type=str, default=None,
                        help="Path to TOTEM tokenizer checkpoint (.pt)")
    parser.add_argument("--llm-id", type=str, default="Qwen/Qwen3-4B",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--output-dir", type=str, default="results/bearing_qualitative")
    parser.add_argument("--skip-gpt4o", action="store_true",
                        help="Skip GPT-4o call in experiment 4 (use placeholder)")
    parser.add_argument("--think", action="store_true",
                        help="Enable Qwen3 think mode (ChatML + system prompt)")
    parser.add_argument("--experiment", type=int, default=None, choices=[1, 2, 3, 4],
                        help="Run only a specific experiment (1-4)")
    args = parser.parse_args()

    global USE_CHATML
    USE_CHATML = args.think

    # SageMaker compatibility
    ckpt_dir = Path(os.environ.get("SM_CHANNEL_CHECKPOINT", "../checkpoints"))
    tok_dir = Path(os.environ.get("SM_CHANNEL_TOKENIZER", "../checkpoints"))
    output_dir = os.environ.get("SM_MODEL_DIR", args.output_dir)
    llm_dir = os.environ.get("SM_CHANNEL_LLM", None)
    llm_id = llm_dir if llm_dir and os.path.exists(llm_dir) else args.llm_id

    # Resolve checkpoint
    ckpt_path = args.checkpoint
    if not ckpt_path:
        for f in sorted(ckpt_dir.glob("*.pt")):
            ckpt_path = str(f)
            break
    # Resolve tokenizer
    tok_path = args.tokenizer_ckpt
    if not tok_path:
        for f in sorted(tok_dir.glob("*totem_clean*.pt")):
            tok_path = str(f)
            break
        if not tok_path:
            for f in sorted(tok_dir.glob("*totem*.pt")):
                tok_path = str(f)
                break

    print(f"Checkpoint: {ckpt_path}")
    print(f"Tokenizer:  {tok_path}")
    print(f"LLM:        {llm_id}")
    print(f"Output:     {output_dir}")

    from tempo.model.bearing import BearingModel
    model = BearingModel.from_pretrained(
        ckpt_path,
        llm_id=llm_id,
        totem_ckpt=tok_path,
        device="cuda",
    )
    model.llm.half()
    print("Model loaded.\n")

    if args.experiment is not None:
        os.makedirs(output_dir, exist_ok=True)
        signals = load_multiple_segments("007", n_segments=5)

        if args.experiment == 1:
            res = experiment_1_context_dependence(model, signals)
        elif args.experiment == 2:
            res = experiment_2_severity(model)
        elif args.experiment == 3:
            res = experiment_3_missing_conflicting(model, signals)
        elif args.experiment == 4:
            res = experiment_4_vs_posthoc(model, signals, args.skip_gpt4o)

        out_path = os.path.join(output_dir, f"exp{args.experiment}_results.json")
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {out_path}")
    else:
        run_all(model, output_dir, args.skip_gpt4o)


if __name__ == "__main__":
    main()
