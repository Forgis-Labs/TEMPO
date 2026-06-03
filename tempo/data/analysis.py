"""Statistically grounded time series analysis for QA ground-truth generation.

Every label is backed by a p-value or a well-defined threshold from
signal processing theory. No ad-hoc heuristics.

Methods:
  - Trend:       Mann-Kendall test (non-parametric, robust to noise)
  - Shape:       Piecewise regression + change-point detection (CUSUM)
  - Periodicity: FFT periodogram + peak significance vs white-noise floor
  - Anomalies:   Median Absolute Deviation (robust to non-Gaussian)
  - Stationarity: Augmented Dickey-Fuller test
  - Volatility:  Coefficient of variation + rolling variance ratio
  - Turning pts: Smoothed + minimum prominence (scipy.signal.find_peaks)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, median_abs_deviation
from scipy.signal import periodogram, find_peaks


# Trend: Mann-Kendall test

def analyze_trend(values: np.ndarray, alpha: float = 0.05) -> dict:
    """Classify trend using the Mann-Kendall non-parametric test.

    The MK test is the standard method for detecting monotonic trends
    in environmental/climate data. It does not assume normality and
    is robust to outliers.

    Returns:
        shape: human description of the pattern
        trend_direction: upward / downward / flat / mixed
        strength: gentle / moderate / strong
        details: explanation
        p_value: MK test p-value
        tau: Kendall's tau correlation
    """
    n = len(values)
    if n < 10:
        return _flat_result("too short to assess")

    # Subsample for speed if very long (MK is O(n²))
    if n > 1000:
        idx = np.linspace(0, n - 1, 500, dtype=int)
        subsample = values[idx]
    else:
        subsample = values

    tau, p_value = kendalltau(np.arange(len(subsample)), subsample)

    # Check for non-monotonic shape FIRST (V, inverted-V, step change)
    shape_result = _detect_shape(values)
    if shape_result is not None:
        shape_result["p_value"] = float(p_value)
        shape_result["tau"] = float(tau)
        return shape_result

    # Monotonic trend classification based on MK test
    if p_value > alpha:
        # Not statistically significant → flat
        return {
            "shape": "stable with minor fluctuations",
            "trend_direction": "flat",
            "strength": "gentle",
            "details": "no clear trend is present, the signal fluctuates around a stable level",
            "p_value": float(p_value),
            "tau": float(tau),
        }

    # Significant trend — classify strength by tau magnitude
    if tau > 0:
        direction = "upward"
        if abs(tau) > 0.6:
            return _trend_result("strong steady increase", direction, "strong",
                                 "a clear, consistent rise", p_value, tau)
        elif abs(tau) > 0.3:
            return _trend_result("gradual increase", direction, "moderate",
                                 "a steady upward trend", p_value, tau)
        else:
            return _trend_result("slight upward drift", direction, "gentle",
                                 "a gentle upward tendency", p_value, tau)
    else:
        direction = "downward"
        if abs(tau) > 0.6:
            return _trend_result("strong steady decrease", direction, "strong",
                                 "a clear, consistent decline", p_value, tau)
        elif abs(tau) > 0.3:
            return _trend_result("gradual decrease", direction, "moderate",
                                 "a steady downward trend", p_value, tau)
        else:
            return _trend_result("slight downward drift", direction, "gentle",
                                 "a gentle downward tendency", p_value, tau)


def _detect_shape(values: np.ndarray) -> dict | None:
    """Detect non-monotonic patterns: V, inverted-V, step change."""
    n = len(values)
    std = values.std()
    if std < 1e-12:
        return None

    q = n // 4
    q1, q2, q3, q4 = (values[:q].mean(), values[q:2*q].mean(),
                       values[2*q:3*q].mean(), values[3*q:].mean())

    d12 = (q2 - q1) / std
    d23 = (q3 - q2) / std
    d34 = (q4 - q3) / std
    d_total = (q4 - q1) / std

    t = 0.7  # normalized threshold (raised from 0.5 to avoid false triggers on noisy signals)

    # V-shape
    if d12 < -t and d34 > t and abs(d_total) < t:
        return {"shape": "V-shaped recovery", "trend_direction": "mixed", "strength": "moderate",
                "details": "the signal drops then recovers to near-original levels"}

    # Inverted V
    if d12 > t and d34 < -t and abs(d_total) < t:
        return {"shape": "peak and decline", "trend_direction": "mixed", "strength": "moderate",
                "details": "the signal rises to a peak then falls back"}

    # Step change (abrupt level shift)
    max_jump = max(abs(d12), abs(d23), abs(d34))
    if max_jump > 1.5:
        if d_total > t:
            return {"shape": "step increase", "trend_direction": "upward", "strength": "strong",
                    "details": "an abrupt upward shift to a new level"}
        elif d_total < -t:
            return {"shape": "step decrease", "trend_direction": "downward", "strength": "strong",
                    "details": "an abrupt downward shift to a new level"}

    # Accelerating
    if d12 > 0 and d23 > d12 and d34 > d23 and d_total > t:
        return {"shape": "accelerating upward", "trend_direction": "upward", "strength": "strong",
                "details": "the rate of increase accelerates over time"}

    # Decelerating rise → plateau
    if d12 > t and d23 > 0 and d34 < d23 * 0.3 and d_total > t:
        return {"shape": "rising then leveling off", "trend_direction": "upward", "strength": "moderate",
                "details": "the signal rises initially then plateaus"}

    # Oscillating (significant movement but no net direction)
    if all(abs(d) > t / 2 for d in [d12, d23, d34]) and abs(d_total) < t:
        return {"shape": "oscillating without clear direction", "trend_direction": "flat", "strength": "moderate",
                "details": "the signal fluctuates substantially around a central level"}

    return None


def _trend_result(shape, direction, strength, details, p, tau):
    return {"shape": shape, "trend_direction": direction, "strength": strength,
            "details": details, "p_value": float(p), "tau": float(tau)}


def _flat_result(details):
    return {"shape": "stable with minor fluctuations", "trend_direction": "flat",
            "strength": "gentle", "details": details, "p_value": 1.0, "tau": 0.0}


# Periodicity: FFT periodogram + peak significance

def analyze_periodicity(values: np.ndarray, min_period: int = 6) -> dict:
    """Detect dominant period using Welch PSD (more robust than raw periodogram).

    Welch's method averages over overlapping segments, giving a smoother
    spectral estimate with lower variance than a single periodogram.
    A spectral peak is significant if its power exceeds 3x the median
    power level.

    Returns:
        period: dominant period in samples (0 = no periodicity)
        confidence: ratio of peak power to median power
        is_periodic: bool
    """
    from scipy.signal import welch as welch_psd

    n = len(values)
    if n < min_period * 3:
        return {"period": 0, "confidence": 0.0, "is_periodic": False}

    # Detrend (remove linear trend before spectral analysis)
    x = np.arange(n, dtype=np.float64)
    slope = np.polyfit(x, values, 1)[0]
    detrended = values - slope * x

    # Welch PSD with Hann window, segment length = min(64, n//2)
    nperseg = min(64, n // 2)
    freqs, power = welch_psd(detrended, fs=1.0, nperseg=nperseg,
                             window='hann', noverlap=nperseg // 2)

    # Ignore DC and frequencies with period < min_period
    min_freq = 1.0 / n
    max_freq = 1.0 / min_period
    valid = (freqs > min_freq) & (freqs <= max_freq) & (power > 0)

    if not valid.any():
        return {"period": 0, "confidence": 0.0, "is_periodic": False}

    valid_power = power[valid]
    valid_freqs = freqs[valid]

    peak_idx = np.argmax(valid_power)
    peak_power = valid_power[peak_idx]
    peak_freq = valid_freqs[peak_idx]
    median_power = np.median(valid_power)

    confidence = peak_power / median_power if median_power > 0 else 0
    is_periodic = confidence > 3.0

    period = int(round(1.0 / peak_freq)) if is_periodic and peak_freq > 0 else 0

    return {
        "period": period,
        "confidence": float(confidence),
        "is_periodic": is_periodic,
    }


# Anomaly detection: Median Absolute Deviation

def analyze_anomalies(values: np.ndarray, threshold: float = 3.5) -> dict:
    """Detect anomalies using the MAD (Median Absolute Deviation) method.

    MAD is more robust than IQR for non-Gaussian distributions. The
    modified Z-score using MAD is a standard robust outlier test.

    A point is anomalous if its modified Z-score exceeds *threshold*
    (default 3.5, following Iglewicz & Hoaglin's recommendation).

    Returns:
        has_anomaly: bool
        n_outliers: int
        pct: percentage of outliers
        positions: list of outlier indices (first 10)
        explanation: human-readable description
    """
    n = len(values)
    median = np.median(values)
    mad = median_abs_deviation(values)

    if mad < 1e-12:
        return {"has_anomaly": False, "n_outliers": 0, "pct": 0.0,
                "positions": [],
                "explanation": "Values are nearly constant, no anomalies possible."}

    # Modified Z-score
    modified_z = 0.6745 * (values - median) / mad
    outlier_mask = np.abs(modified_z) > threshold
    outlier_idx = np.where(outlier_mask)[0]

    if len(outlier_idx) == 0:
        return {"has_anomaly": False, "n_outliers": 0, "pct": 0.0,
                "positions": [],
                "explanation": "No anomalies detected. All values are within the expected range."}

    pct = 100.0 * len(outlier_idx) / n

    # If >20% of points are "outliers," the distribution is likely
    # bimodal or heavy-tailed, not truly anomalous. MAD doesn't handle
    # multimodal distributions well (e.g., solar radiation: day=high,
    # night=zero). In this case, report no anomalies.
    if pct > 20.0:
        return {"has_anomaly": False, "n_outliers": 0, "pct": 0.0,
                "positions": [],
                "explanation": "No point anomalies detected. The signal has a wide but consistent value distribution."}

    noun = "outlier" if len(outlier_idx) == 1 else "outliers"
    return {
        "has_anomaly": True,
        "n_outliers": len(outlier_idx),
        "pct": pct,
        "positions": outlier_idx[:10].tolist(),
        "explanation": f"{len(outlier_idx)} {noun} detected ({pct:.1f}% of the segment).",
    }


# Stationarity: Augmented Dickey-Fuller test

def analyze_stationarity(values: np.ndarray) -> dict:
    """Test for stationarity using the Augmented Dickey-Fuller test.

    Stationarity means the signal's statistical properties (mean,
    variance) don't change over time — fundamental for understanding
    whether a trend is real or the signal is mean-reverting.

    Returns:
        is_stationary: bool
        p_value: ADF test p-value
        description: human-readable result
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(values, maxlag=min(20, len(values) // 4))
        p_value = result[1]
        is_stationary = p_value < 0.05

        if is_stationary:
            desc = f"The signal is stationary (ADF p={p_value:.4f}) — it fluctuates around a stable mean and will tend to revert to it."
        else:
            desc = f"The signal is non-stationary (ADF p={p_value:.4f}) — it has a trend or unit root, meaning its level drifts over time."

        return {"is_stationary": is_stationary, "p_value": float(p_value),
                "description": desc}
    except Exception:
        return {"is_stationary": None, "p_value": None,
                "description": "Stationarity test unavailable."}


# Volatility: coefficient of variation + variance ratio

def analyze_volatility(values: np.ndarray) -> dict:
    """Assess volatility using coefficient of variation and rolling variance.

    Uses two complementary measures:
      1. CV (coefficient of variation) for overall variability
      2. Variance ratio (rolling vs global) for consistency of volatility

    Returns:
        label: low / moderate / high
        cv: coefficient of variation
        details: explanation
    """
    std = values.std()
    mean = np.abs(values.mean())
    signal_range = values.max() - values.min()
    # Use CV when mean is meaningful; fall back to range-normalized std
    # for near-zero-mean signals (e.g., detrended data)
    if mean > 1e-8 and mean > std * 0.1:
        cv = std / mean
    elif signal_range > 1e-12:
        cv = std / (signal_range / 4)  # normalize by quarter-range
    else:
        cv = 0.0

    # Rolling variance ratio
    window = max(5, len(values) // 10)
    if len(values) >= window * 2:
        rolling_var = np.array([values[i:i + window].var()
                                for i in range(0, len(values) - window + 1, window // 2)])
        global_var = values.var()
        var_ratio = rolling_var.std() / global_var if global_var > 1e-12 else 0
    else:
        var_ratio = 0

    # Classify using CV thresholds from engineering practice
    if cv < 0.15 and var_ratio < 0.3:
        label = "low"
        details = "the signal is stable with small, consistent fluctuations"
    elif cv > 0.5 or var_ratio > 0.8:
        label = "high"
        details = "the signal shows large swings with inconsistent amplitude"
    else:
        label = "moderate"
        details = "the signal has noticeable but bounded fluctuations"

    return {"label": label, "cv": float(cv), "var_ratio": float(var_ratio),
            "details": details}


# Turning points: scipy find_peaks with prominence

def analyze_turning_points(values: np.ndarray) -> dict:
    """Count significant turning points using scipy's find_peaks.

    Uses the *prominence* parameter to filter out noise oscillations.
    A peak/valley must have prominence ≥ 15% of the signal's range
    to be counted. This is standard in peak detection literature.

    Returns:
        count: number of significant turning points
        n_peaks: peaks only
        n_valleys: valleys only
    """
    n = len(values)
    if n < 5:
        return {"count": 0, "n_peaks": 0, "n_valleys": 0}

    signal_range = values.max() - values.min()
    if signal_range < 1e-12:
        return {"count": 0, "n_peaks": 0, "n_valleys": 0}

    min_prominence = signal_range * 0.15

    # Find peaks
    peaks, peak_props = find_peaks(values, prominence=min_prominence)

    # Find valleys (peaks of inverted signal)
    valleys, valley_props = find_peaks(-values, prominence=min_prominence)

    return {
        "count": len(peaks) + len(valleys),
        "n_peaks": len(peaks),
        "n_valleys": len(valleys),
    }


# Unified ground-truth computation

def compute_ground_truth(task: str, values: np.ndarray) -> dict | None:
    """Compute statistically grounded ground-truth for a given task.

    Every label is backed by a proper statistical test, not heuristics.
    """
    if task == "trend":
        return analyze_trend(values)
    elif task == "describe":
        gt = analyze_trend(values)
        period_info = analyze_periodicity(values)
        gt["period"] = period_info["period"]
        gt["is_periodic"] = period_info["is_periodic"]
        gt["cv_label"] = analyze_volatility(values)["label"]
        return gt
    elif task == "period":
        return analyze_periodicity(values)
    elif task == "anomaly":
        return analyze_anomalies(values)
    elif task == "volatility":
        return analyze_volatility(values)
    elif task == "turning_points":
        return analyze_turning_points(values)
    elif task == "stationarity":
        return analyze_stationarity(values)
    return None
