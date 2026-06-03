"""UCR Time Series Classification Archive loader.

Loads any of the 112 UCR datasets, converts to TEMPO's CoT format
(like bearing/HAR), and returns a HuggingFace Dataset.

The UCR archive is the standard benchmark for time series classification.
This module wraps it with:
  - HF Dataset format (consistent with the rest of TEMPO)
  - CoT prompt templates (teaches the model to reason, not just classify)
  - Automatic TOTEM tokenization metadata (signal length, domain type)

Requires: pip install aeon

Usage:
    from tempo.data.ucr import load_ucr_dataset

    # Load a single dataset
    dataset = load_ucr_dataset("ArrowHead")
    print(dataset["train"][0])

    # Load with CoT prompts
    dataset = load_ucr_dataset("ArrowHead", cot=True)

    # List all available datasets
    from tempo.data.ucr import list_ucr_datasets
    print(list_ucr_datasets())
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict


# UCR dataset metadata (type, description for prompts)

UCR_DOMAIN_MAP = {
    "Image": "image outline shape",
    "Sensor": "sensor measurement",
    "Motion": "motion capture",
    "Spectro": "spectroscopy",
    "Simulated": "simulated signal",
    "Device": "device monitoring",
    "ECG": "electrocardiogram",
    "Spectrum": "spectral measurement",
    "Hemodynamics": "hemodynamic monitoring",
    "EPG": "electrical penetration graph",
    "Traffic": "traffic flow",
    "EOG": "electrooculogram",
}

# Rich semantic metadata for datasets where class names and context
# are known. The LLM uses this to understand WHAT it's classifying
# and WHY the signal patterns matter — something DTW/ROCKET can't do.
UCR_SEMANTIC_METADATA: dict[str, dict] = {
    "ArrowHead": {
        "description": "angle-based outline shapes of archaeological arrowhead artifacts, classified by manufacturing tradition",
        "classes": {"0": "Avonlea", "1": "Clovis", "2": "Mix"},
        "domain": "archaeology",
        "class_features": {
            "Avonlea": "distinctive side-notches near the base producing sharp angular changes, with a concave base shape and peak in the latter portion of the outline",
            "Clovis": "lanceolate profile with a smooth outline and a characteristic concave fluted base, narrower amplitude range than Avonlea",
            "Mix": "combines features of both traditions with peak positioned early in the outline and intermediate amplitude variations",
        },
        "discriminative_features": "peak position in the outline (late vs early), presence of angular notch patterns, base shape (concave vs fluted), amplitude range",
    },
    "ECG200": {
        "description": "single-lead ECG heartbeat recordings (one P-QRS-T cycle), distinguishing normal rhythm from myocardial infarction",
        "classes": {"-1": "Normal", "1": "Myocardial Infarction"},
        "domain": "cardiology",
        "class_features": {
            "Normal": "well-defined single R-peak, isoelectric ST segment at baseline, upright T-wave, fewer significant peaks, stronger positive trend (tau ~0.19)",
            "Myocardial Infarction": "ST-segment elevation or T-wave inversion, more complex morphology with additional peaks (mean ~3.1 peaks vs 1.9), weaker trend",
        },
        "discriminative_features": "number of significant peaks (1-2 vs 3+), ST-segment elevation, T-wave polarity, overall trend strength",
    },
    "ECG5000": {
        "description": "ECG heartbeat morphology classification into 5 clinical categories from the BIDMC Congestive Heart Failure Database",
        "classes": {"1": "Normal", "2": "R-on-T PVC", "3": "PVC", "4": "Supraventricular", "5": "Unclassifiable"},
        "domain": "cardiology",
        "class_features": {
            "Normal": "standard P-QRS-T morphology with narrow QRS complex, ~2.2 peaks, strongest upward trend, valley early in the beat",
            "R-on-T PVC": "premature wide complex overlapping the preceding T-wave, fewer peaks (~1.2), valley positioned late",
            "PVC": "wide bizarre QRS complex (>120ms), no preceding P-wave, T-wave deflects opposite to QRS, ~1.1 peaks, peak at 90% of cycle",
            "Supraventricular": "narrow QRS but abnormal P-wave or short PR interval, ~1.2 peaks, peak very late in cycle",
            "Unclassifiable": "ambiguous morphology that doesn't fit other categories, ~2.0 peaks, moderate trend",
        },
        "discriminative_features": "QRS width, peak count, peak position within cycle, valley position, trend direction and strength",
    },
    "GunPoint": {
        "description": "X-axis centroid tracking of a person's right hand during motion capture, distinguishing drawing a gun from pointing a finger",
        "classes": {"1": "Gun-Draw", "2": "Point"},
        "domain": "motion capture",
        "class_features": {
            "Gun-Draw": "motion arc with a subtle secondary dip in the return phase (hand moves toward hip to holster), weaker downward trend, valley positioned differently",
            "Point": "smooth direct return after pointing motion, stronger downward trend (tau ~-0.08 vs -0.01), no secondary dip in the return phase",
        },
        "discriminative_features": "subtle shape difference in the return phase (secondary dip vs smooth return), valley position, trend strength -- the original shapelet dataset",
    },
    "Beef": {
        "description": "mid-infrared absorbance spectra of beef tissue samples, identifying tissue type by molecular composition",
        "classes": {"1": "Pure Beef", "2": "Heart", "3": "Kidney", "4": "Liver", "5": "Tripe"},
        "domain": "food science",
        "class_features": {
            "Pure Beef": "spectral profile of skeletal muscle with characteristic protein absorption, widest amplitude range (~5.1), more peaks from complex protein bands",
            "Heart": "cardiac muscle spectrum with different myoglobin content, more spectral peaks (~3.5), peak position at ~70% of spectrum",
            "Kidney": "distinct fat/protein ratio spectrum, fewer peaks (~2.5) than heart, intermediate amplitude range",
            "Liver": "high iron and protein content producing distinct absorption bands, strongest downward trend (tau ~-0.27), multiple peaks",
            "Tripe": "high collagen absorption from stomach lining, weakest downward trend, intermediate peak count (~2.8)",
        },
        "discriminative_features": "absorption peak heights and positions across mid-IR range, trend direction (related to baseline slope), peak count, amplitude range reflecting tissue composition",
    },
    "Coffee": {
        "description": "FTIR spectra of coffee beans, distinguishing varieties by chemical composition (lipid content, chlorogenic acid, caffeine)",
        "classes": {"0": "Arabica", "1": "Robusta"},
        "domain": "food science",
        "class_features": {
            "Arabica": "higher lipid content producing stronger C-H stretching bands, slightly fewer spectral peaks (~4.6), nearly flat overall trend",
            "Robusta": "higher caffeine content affecting absorption bands, slightly more spectral peaks (~5.0), weak positive trend, peak position shifted to ~77% of spectrum",
        },
        "discriminative_features": "subtle differences in specific band intensities (carbohydrate region 1000-1800, lipid region 2800-3000), peak count, peak position -- a difficult dataset requiring fine spectral discrimination",
    },
    "CBF": {
        "description": "synthetic benchmark with three fundamental waveform shapes embedded in noise, each defined over a random interval",
        "classes": {"1": "Cylinder", "2": "Bell", "3": "Funnel"},
        "domain": "synthetic benchmark",
        "class_features": {
            "Cylinder": "flat rectangular pulse (constant amplitude plateau with abrupt start and end), no overall trend (tau ~0.01), many small oscillatory peaks from noise",
            "Bell": "symmetric rise to a peak then fall back (Gaussian-like bump), positive trend in first half (tau ~0.24), most peaks from noise oscillations",
            "Funnel": "starts at peak amplitude and decays to zero (exponential decay), negative trend (tau ~-0.23), widest amplitude range, peak position early in signal",
        },
        "discriminative_features": "trend direction is the primary discriminator (flat vs positive vs negative), shape morphology (plateau vs peak vs decay), amplitude range, peak position",
    },
    "Wafer": {
        "description": "in-line process control sensor readings during semiconductor wafer fabrication, detecting manufacturing anomalies",
        "classes": {"-1": "Normal", "1": "Abnormal"},
        "domain": "semiconductor manufacturing",
        "class_features": {
            "Normal": "smooth expected process signature with ramp-up, plateau, and ramp-down phases, fewer peaks (~1.8), narrower amplitude range",
            "Abnormal": "deviations from expected trajectory such as unexpected spikes, delayed transitions, or abnormal plateaus, more peaks (~2.5), wider amplitude range, stronger downward trend",
        },
        "discriminative_features": "peak count (local anomalies), amplitude range, trend strength, presence of unexpected transitions in the process curve",
    },
    "Lightning2": {
        "description": "satellite sensor optical transient data capturing lightning flash characteristics",
        "classes": {"0": "Intra-cloud", "1": "Cloud-to-ground"},
        "domain": "meteorology",
        "class_features": {
            "Intra-cloud": "multiple lower-amplitude pulses spread over longer duration as discharge propagates through cloud, more peaks (~9.9), narrower amplitude range",
            "Cloud-to-ground": "single dominant sharp peak from the bright return stroke, fewer peaks (~7.2), wider amplitude range (~10.6 vs 8.2), stronger upward trend",
        },
        "discriminative_features": "peak structure (multiple diffuse vs single dominant), amplitude range, peak count, overall trend strength",
    },
    "Earthquakes": {
        "description": "hourly-averaged seismograph recordings from the Northern California Earthquake Data Center",
        "classes": {"0": "Non-earthquake", "1": "Earthquake"},
        "domain": "seismology",
        "class_features": {
            "Non-earthquake": "random background noise or slow low-amplitude variations, fewer oscillation peaks (~66), main peak early in signal (15%), wider amplitude range",
            "Earthquake": "characteristic impulsive onset followed by larger amplitude waves and gradual decay, more oscillation peaks (~80), main peak late in signal (94%), narrower amplitude range from averaging",
        },
        "discriminative_features": "peak position is the primary discriminator (early vs late -- 15% vs 94%), peak count (more oscillations in earthquake coda), amplitude envelope shape",
    },
    "StarLightCurves": {
        "description": "phase-folded light curves of periodic variable stars from astronomical surveys",
        "classes": {"1": "Eclipsing Binary", "2": "Cepheid", "3": "RR Lyrae"},
        "domain": "astronomy",
        "class_features": {
            "Eclipsing Binary": "two sharp symmetric dips per period (primary and secondary eclipses) with flat maxima between them, strong negative trend (tau ~-0.28), narrowest amplitude range (~3.1), single dominant peak",
            "Cepheid": "asymmetric sawtooth waveform with rapid brightness increase and slower decline, weakest trend (tau ~-0.08), widest amplitude range (~4.2), two peaks per cycle",
            "RR Lyrae": "sharply asymmetric profile (fast rise, slower fall) with possible Blazhko bump, moderate trend (tau ~-0.14), intermediate amplitude (~3.4), single peak",
        },
        "discriminative_features": "trend strength separates all three classes, amplitude range (widest for Cepheid), peak count (2 dips for EB, 1 for pulsators), waveform symmetry",
    },
    "Trace": {
        "description": "simulated nuclear power plant instrumentation transients representing different reactor parameter responses",
        "classes": {"1": "Normal", "2": "Anomaly Type 1", "3": "Anomaly Type 2", "4": "Anomaly Type 3"},
        "domain": "nuclear engineering",
        "class_features": {
            "Normal": "smooth step response with clear peak, one significant peak, weak trend (tau ~0.11), widest amplitude range (~5.8)",
            "Anomaly Type 1": "no significant peaks, weak trend (tau ~0.10), narrow amplitude range (~2.9) -- subdued or missing transient response",
            "Anomaly Type 2": "one significant peak with strong upward trend (tau ~0.56), peak positioned late at ~72% of signal, narrow range (~2.9)",
            "Anomaly Type 3": "no significant peaks but strong upward trend (tau ~0.56), peak at ~63% of signal, narrowest range (~2.5) -- gradual drift without transient",
        },
        "discriminative_features": "peak count (1 vs 0), trend strength (weak ~0.1 vs strong ~0.56), amplitude range (5.8 vs 2.5-2.9), peak position",
    },
    "Ham": {
        "description": "near-infrared spectroscopy of ham samples from two geographic origins with different pig breeds and diets",
        "classes": {"1": "Spanish", "2": "French"},
        "domain": "food science",
        "class_features": {
            "Spanish": "Iberian pig spectrum with acorn-fed fatty acid profile, more spectral peaks (~8.4), slightly weaker trend, narrower amplitude range",
            "French": "different fat and protein composition, fewer spectral peaks (~7.5), slightly stronger trend, wider amplitude range (~5.7 vs 5.4)",
        },
        "discriminative_features": "peak count in fat and protein absorption bands (1700-1800nm, 1400-1500nm), amplitude range, subtle spectral shape differences -- a difficult dataset",
    },
    "Herring": {
        "description": "boundary contour curvature of sagittal otolith (ear bone) outlines from Atlantic herring populations",
        "classes": {"1": "North Sea", "2": "Thames"},
        "domain": "marine biology",
        "class_features": {
            "North Sea": "autumn/winter spawning stock otolith shape with specific rostrum and excisura profiles, slightly fewer peaks (~2.1)",
            "Thames": "spring spawning stock with different aspect ratio and rostrum shape, slightly more peaks (~2.2)",
        },
        "discriminative_features": "localized shape differences at rostrum tip, antirostrum, and excisura -- very subtle, requires fine-grained shape discrimination",
    },
    "Wine": {
        "description": "FTIR-ATR spectroscopy of wine samples, distinguishing grape varieties by polyphenol and acid profiles",
        "classes": {"1": "Variety A", "2": "Variety B"},
        "domain": "food science",
        "class_features": {
            "Variety A": "Cabernet Sauvignon-like tannin and anthocyanin spectral profile",
            "Variety B": "Shiraz-like spectral profile with different polyphenol bands",
        },
        "discriminative_features": "subtle absorption differences in polyphenol bands (~1050-1150), organic acid bands (~1400-1700), and O-H stretching (~3000-3500) -- very difficult, near-identical summary statistics",
    },
    "Yoga": {
        "description": "centroid-to-contour distance profiles of body silhouettes during yoga poses, identifying the performer",
        "classes": {"1": "Male", "2": "Female"},
        "domain": "motion capture",
        "class_features": {
            "Male": "broader shoulder-to-hip ratio in silhouette producing larger amplitude peaks at shoulder angles, wider amplitude range (~3.9)",
            "Female": "different body proportions and flexibility producing distinct limb extension profiles, narrower amplitude range (~3.7), slightly stronger downward trend",
        },
        "discriminative_features": "relative peak amplitudes corresponding to limb extensions, body proportion ratios in the silhouette outline, amplitude range",
    },
    "Car": {
        "description": "side-view car silhouette outlines converted to distance-from-centroid profiles, identifying body type",
        "classes": {"1": "Type A", "2": "Type B", "3": "Type C", "4": "Type D"},
        "domain": "automotive engineering",
        "class_features": {
            "Type A": "silhouette profile with weak positive trend, few peaks (~1.1)",
            "Type B": "nearly flat trend, few peaks (~1.2)",
            "Type C": "negative trend (sloped roofline), more peaks (~1.8), widest amplitude range from distinct body proportions",
            "Type D": "nearly flat trend, few peaks (~1.2)",
        },
        "discriminative_features": "roofline profile slope (trend direction), number of prominent shape features (peaks), hood-to-trunk ratio, overall height-to-length aspect ratio",
    },
    "Plane": {
        "description": "aircraft silhouette outline profiles, identifying aircraft type by wing, fuselage, and tail configuration",
        "classes": {str(i): f"Aircraft {i}" for i in range(1, 8)},
        "domain": "aerospace",
        "class_features": {
            "Aircraft 1": "outline with ~2.9 peaks, peak at end of profile, longer period features",
            "Aircraft 2": "~3.0 peaks with shorter period oscillations (~37 samples), peak at end",
            "Aircraft 3": "~3.0 peaks, slight negative trend, peak at end, long period features",
            "Aircraft 4": "most peaks (~4.5) from complex wing/engine configuration, widest amplitude, short period (~32 samples)",
            "Aircraft 5": "~3.0 peaks with peak at center (51%) rather than edge -- distinctive symmetric profile",
            "Aircraft 6": "~3.9 peaks, peak at end, longer period features",
            "Aircraft 7": "~4.0 peaks with peak at start (0%) -- reversed profile, very short period (~22 samples) from compact design",
        },
        "discriminative_features": "peak position (center vs edges), peak count (wing/engine complexity), period of outline oscillation (wing sweep), amplitude (fuselage size), trend direction",
    },
    "FaceFour": {
        "description": "face profile outlines from 4 individuals, biometric identification from nose, forehead, and chin shapes",
        "classes": {"1": "Person A", "2": "Person B", "3": "Person C", "4": "Person D"},
        "domain": "biometrics",
        "class_features": {
            "Person A": "~7.2 profile peaks, moderate amplitude (~5.2), weak positive trend, peak at end of profile",
            "Person B": "most profile peaks (~11.4) suggesting more facial feature detail, wider amplitude (~5.6), slight negative trend",
            "Person C": "widest amplitude range (~7.0) indicating most prominent facial features, ~9 peaks, peak at start of profile -- distinctive reversed orientation",
            "Person D": "~8.4 peaks, moderate amplitude (~6.0), weak positive trend, peak at end",
        },
        "discriminative_features": "peak count (facial feature complexity), amplitude range (feature prominence), peak position (start vs end -- profile orientation), forehead slope and nose bridge profile",
    },
    "TwoLeadECG": {
        "description": "ECG heartbeats from two different leads capturing different cardiac axis projections",
        "classes": {"1": "Morphology A", "2": "Morphology B"},
        "domain": "cardiology",
        "class_features": {
            "Morphology A": "lateral axis projection (Lead I-like) with stronger positive trend (tau ~0.27), peak later in cycle (~82%)",
            "Morphology B": "inferior axis projection (Lead II-like) with weaker positive trend (tau ~0.22), peak earlier in cycle (~79%)",
        },
        "discriminative_features": "P-wave polarity, QRS axis projection, T-wave amplitude and direction, trend strength, peak timing within the cardiac cycle",
    },
    "Strawberry": {
        "description": "mid-infrared spectra of fruit purees, distinguishing authentic strawberry from adulterated samples",
        "classes": {"1": "Authentic", "2": "Adulterated"},
        "domain": "food science",
        "class_features": {
            "Authentic": "strawberry-specific sugar ratios and anthocyanin peaks, fewer spectral peaks (~1.1), stronger downward trend (tau ~-0.52)",
            "Adulterated": "shifted sugar profile from cheaper purees (apple/plum), more spectral peaks (~1.7) from mixed composition, slightly weaker downward trend (tau ~-0.50)",
        },
        "discriminative_features": "peak count (pure vs mixed composition), trend strength, specific absorption band ratios in sugar region (~1000-1100) and ester carbonyl (~1740)",
    },
}


# CoT prompt templates for classification

CLASSIFICATION_QUESTIONS = [
    "Classify this time series.",
    "What class does this signal belong to?",
    "Identify the category of this time series.",
    "Analyze this signal and determine its class.",
    "Based on the pattern, what type is this?",
    "Examine this time series and classify it.",
    "What does this signal represent?",
    "Determine the classification of this data.",
]

CLASSIFICATION_COT_TEMPLATES = [
    "Looking at the signal pattern, I observe {observation}. "
    "The key features suggest this belongs to class {label}. "
    "Answer: {label}",

    "The time series shows {observation}. "
    "Based on these characteristics, the classification is {label}. "
    "Answer: {label}",

    "After analyzing the signal, the pattern indicates {observation}. "
    "This is consistent with class {label}. "
    "Answer: {label}",

    "The data exhibits {observation}. "
    "These features are characteristic of {label}. "
    "Answer: {label}",

    "Examining the waveform: {observation}. "
    "The signal matches the profile of class {label}. "
    "Answer: {label}",
]

OBSERVATION_TEMPLATES = [
    "distinctive temporal patterns with specific amplitude variations",
    "a characteristic shape profile that distinguishes it from other classes",
    "unique frequency components and temporal dynamics",
    "specific morphological features in the waveform",
    "a recognizable pattern structure typical of this category",
]


def _describe_signal(signal: np.ndarray, dataset: str = "", label: str = "") -> str:
    """Generate a grounded observation from actual signal analysis.

    Combines statistical analysis of this specific signal with domain knowledge
    about what features distinguish the classes in this dataset.
    """
    from .analysis import (
        analyze_trend, analyze_periodicity,
        analyze_anomalies, analyze_turning_points,
    )

    n = len(signal)
    parts = []

    # Trend / shape
    trend = analyze_trend(signal)
    parts.append(trend["details"])

    # Periodicity
    period = analyze_periodicity(signal)
    if period["is_periodic"]:
        pct = 100.0 * period["period"] / n
        parts.append(f"a periodic component with period ~{period['period']} samples "
                     f"({pct:.0f}% of signal length)")
    else:
        parts.append("no dominant periodic component")

    # Turning points (shape complexity)
    tp = analyze_turning_points(signal)
    if tp["count"] > 0:
        parts.append(f"{tp['n_peaks']} significant peaks and {tp['n_valleys']} valleys")
    else:
        parts.append("a smooth shape with no significant peaks or valleys")

    # Amplitude
    sig_range = float(signal.max() - signal.min())
    parts.append(f"amplitude range of {sig_range:.2f}")

    # Peak position (discriminative for many datasets)
    peak_pos = float(np.argmax(signal)) / n * 100
    parts.append(f"main peak at {peak_pos:.0f}% of the signal")

    # Anomalies (only if present)
    anom = analyze_anomalies(signal)
    if anom["has_anomaly"]:
        parts.append(f"{anom['n_outliers']} outlier points")

    # Add class-specific reasoning if available
    meta = UCR_SEMANTIC_METADATA.get(dataset, {})
    class_features = meta.get("class_features", {})
    if label and label in class_features:
        parts.append(f"These features are consistent with {label}: {class_features[label]}")

    # Format
    formatted = []
    for p in parts:
        if p and p[0].islower():
            p = p[0].upper() + p[1:]
        formatted.append(p)
    return ". ".join(formatted)


def _get_curated_cot(dataset: str, label: str) -> list[str] | None:
    """Get curated per-class CoT templates if available.

    Returns a list of 10 reasoning variations, or None if not available
    for this dataset/class (falls back to signal analysis).
    """
    from .ucr_cot_templates import UCR_COT_TEMPLATES
    from .ucr_cot_remaining import REMAINING_TEMPLATES

    # Check both template sources
    for source in (UCR_COT_TEMPLATES, REMAINING_TEMPLATES):
        if dataset in source and label in source[dataset]:
            return source[dataset][label]
    return None


# Context prompt templates (domain-aware)

CONTEXT_TEMPLATES = [
    "You are classifying a {domain} time series from the {dataset} dataset. "
    "The signal has {length} time points and belongs to one of {n_classes} classes.",

    "Analyze this {domain} signal from {dataset}. "
    "There are {n_classes} possible classes. The signal is {length} points long.",

    "This is a {domain} time series ({dataset}, {length} points). "
    "Classify it into one of {n_classes} categories.",

    "You are given a {length}-point {domain} signal from the {dataset} benchmark. "
    "Determine which of the {n_classes} classes it belongs to.",
]


# Data loading

def list_ucr_datasets() -> list[str]:
    """List all available UCR dataset names."""
    try:
        from aeon.datasets.tsc_datasets import univariate_equal_length
        return sorted(univariate_equal_length)
    except ImportError:
        raise ImportError("Install aeon: pip install aeon")


def load_ucr_dataset(
    name: str,
    cot: bool = True,
    seed: int = 42,
) -> DatasetDict:
    """Load a UCR dataset as a HuggingFace DatasetDict.

    Args:
        name: UCR dataset name (e.g., "ArrowHead", "ECG200").
        cot: If True, generate CoT classification prompts.
             If False, return raw time series + labels only.
        seed: RNG seed for template selection.

    Returns:
        DatasetDict with 'train' and 'test' splits. Each sample has:
          - time_series: list of lists (1 channel)
          - pre_prompt: context text
          - post_prompt: question
          - answer: CoT reasoning + "Answer: {label}"
          - task: "classification"
          - label: class label string
          - domain: dataset domain type
    """
    try:
        from aeon.datasets import load_classification
    except ImportError:
        raise ImportError("Install aeon: pip install aeon")

    rng = random.Random(seed)

    # Load train and test
    X_train, y_train = load_classification(name, split="train")
    X_test, y_test = load_classification(name, split="test")

    # Get dataset metadata
    n_classes = len(set(y_train))
    length = X_train.shape[-1]
    domain = _get_domain(name)

    # Build samples
    train_samples = _build_samples(X_train, y_train, name, domain, n_classes, length, cot, rng)
    test_samples = _build_samples(X_test, y_test, name, domain, n_classes, length, cot, rng)

    return DatasetDict({
        "train": Dataset.from_dict(_samples_to_dict(train_samples)),
        "test": Dataset.from_dict(_samples_to_dict(test_samples)),
    })


def _build_samples(
    X: np.ndarray,
    y: np.ndarray,
    dataset: str,
    domain: str,
    n_classes: int,
    length: int,
    cot: bool,
    rng: random.Random,
) -> list[dict]:
    """Convert raw arrays to TEMPO sample dicts."""
    samples = []
    classes = sorted(set(y))

    # Use semantic class names if available, else numeric
    meta = UCR_SEMANTIC_METADATA.get(dataset, {})
    semantic_classes = meta.get("classes", {})
    description = meta.get("description", f"{domain} data from the {dataset} dataset")

    def class_name(c: str) -> str:
        return semantic_classes.get(c, c)

    for i in range(len(X)):
        # X shape is (n_samples, n_channels, length) — take first channel
        signal = X[i][0].astype(np.float64).tolist()
        raw_label = str(y[i])
        label = class_name(raw_label)

        if cot:
            if semantic_classes:
                disc_features = meta.get("discriminative_features", "")
                context = f"You are analyzing {description}. " \
                          f"The signal has {length} time points."
                if disc_features:
                    context += f" Key distinguishing features: {disc_features}."
            else:
                context = rng.choice(CONTEXT_TEMPLATES).format(
                    domain=domain, dataset=dataset,
                    length=length, n_classes=n_classes,
                )
            question = rng.choice(CLASSIFICATION_QUESTIONS)

            # Use curated per-class CoT templates if available
            curated = _get_curated_cot(dataset, label)
            if curated:
                answer = rng.choice(curated)
            else:
                # Fallback: signal analysis + generic template
                observation = _describe_signal(
                    np.asarray(signal, dtype=np.float64),
                    dataset=dataset, label=label,
                )
                answer = rng.choice(CLASSIFICATION_COT_TEMPLATES).format(
                    observation=observation, label=label,
                )
        else:
            context = f"Classify this {domain} time series from {dataset}."
            question = "What class is this?"
            answer = f"Answer: {label}"

        # Possible classes with semantic names
        if semantic_classes:
            classes_str = ", ".join(class_name(str(c)) for c in classes)
        else:
            classes_str = ", ".join(str(c) for c in classes)
        post = f"{question} Possible types: {classes_str}."

        samples.append({
            "time_series": [signal],
            "time_series_text": [f"{domain.capitalize()} signal:"],
            "pre_prompt": context,
            "post_prompt": post,
            "answer": answer,
            "task": "classification",
            "label": label,
            "domain": domain,
            "dataset": dataset,
        })

    return samples


def _samples_to_dict(samples: list[dict]) -> dict[str, list]:
    """Convert list of dicts to dict of lists (HF format)."""
    if not samples:
        return {}
    keys = samples[0].keys()
    return {k: [s[k] for s in samples] for k in keys}


def _get_domain(name: str) -> str:
    """Infer domain description from dataset name."""
    # Check known mappings from UCR metadata
    name_lower = name.lower()
    if "ecg" in name_lower:
        return "electrocardiogram"
    if "eog" in name_lower:
        return "electrooculogram"
    if "gun" in name_lower or "hand" in name_lower:
        return "motion capture"
    if "arrow" in name_lower or "fish" in name_lower or "beetle" in name_lower:
        return "image outline shape"
    if "chlorine" in name_lower or "power" in name_lower:
        return "sensor measurement"
    if "beef" in name_lower or "coffee" in name_lower or "wine" in name_lower:
        return "spectroscopy"
    if "wafer" in name_lower or "computer" in name_lower:
        return "device monitoring"
    if "cbf" in name_lower or "synthetic" in name_lower:
        return "simulated signal"
    return "time series"
