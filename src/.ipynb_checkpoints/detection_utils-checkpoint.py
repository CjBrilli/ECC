"""
detection_utils.py

Detection helpers for DSN phase-scintillation event analysis.

This module is based directly on the existing working workflow:

Block 3:
- attach daily elongation to phase windows
- build quiet baseline vs elongation
- compute expected phase and phase_ratio

Block 4:
- detect CIR-like long-duration structures from smoothed phase_ratio

Block 5:
- remove CIR-scale background
- detect transient / CME-like events from clean_signal

The goal is to preserve the existing science, not redesign it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# ============================================================
# CONFIG DATACLASSES
# ============================================================

@dataclass
class BaselineConfig:
    bin_width_deg: float = 2.0
    max_elong_deg: float = 50.0


@dataclass
class CIRConfig:
    step_min: int = 10
    window_hours: int = 12
    thresh_on: float = 1.4
    thresh_off: float = 1.2
    min_duration_hr: float = 24.0


@dataclass
class TransientConfig:
    step_min: int = 20
    window_hours: int = 12
    threshold: float = 3.0
    min_duration_hr: float = 0.25
    max_duration_hr: float = 24.0


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_percentile(x: pd.Series | np.ndarray, q: float) -> float:
    """
    Safe percentile helper that returns NaN for empty input.
    """
    arr = pd.Series(x).dropna().values
    if len(arr) == 0:
        return np.nan
    return float(np.percentile(arr, q))


def attach_daily_elongation_to_windows(
    windows_df: pd.DataFrame,
    horizons_daily: pd.DataFrame,
    mid_col: str = "mid",
) -> pd.DataFrame:
    """
    Attach daily elongation to each window using window midpoint floored to day.

    Parameters
    ----------
    windows_df : pd.DataFrame
        Must contain `mid_col`.
    horizons_daily : pd.DataFrame
        Must contain columns ['day', 'elongation_deg'].
    mid_col : str
        Name of midpoint datetime column in windows_df.

    Returns
    -------
    pd.DataFrame
        Copy of windows_df with:
        - day
        - elongation_deg
    """
    if mid_col not in windows_df.columns:
        raise ValueError(f"windows_df must contain '{mid_col}'")

    required = {"day", "elongation_deg"}
    if not required.issubset(horizons_daily.columns):
        raise ValueError(f"horizons_daily must contain columns {required}")

    out = windows_df.copy()

    cols_to_drop = ["elongation_deg", "elongation_deg_x", "elongation_deg_y", "day"]
    out = out.drop(columns=[c for c in cols_to_drop if c in out.columns], errors="ignore")

    geom_daily = horizons_daily.copy()
    geom_daily["day"] = pd.to_datetime(geom_daily["day"])

    out["day"] = pd.to_datetime(out[mid_col]).dt.floor("D")

    out = out.merge(
        geom_daily[["day", "elongation_deg"]],
        on="day",
        how="left"
    )

    return out


# ============================================================
# BLOCK 3 — BASELINE VS ELONGATION
# ============================================================

def build_phase_baseline_vs_elongation(
    windows_df: pd.DataFrame,
    config: Optional[BaselineConfig] = None,
    phase_col: str = "phase_rms_rad",
    elong_col: str = "elongation_deg",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build quiet baseline vs elongation and compute expected phase / phase_ratio.

    This follows the working Block 3 logic:
    - drop invalid rows
    - bin by elongation
    - compute median / p25 / p75
    - define hybrid baseline = 0.5*p25 + 0.5*median
    - interpolate expected phase
    - compute phase_ratio = observed / expected

    Parameters
    ----------
    windows_df : pd.DataFrame
        Must contain phase and elongation columns.
    config : BaselineConfig, optional
        Baseline settings.
    phase_col : str
        Name of phase RMS column.
    elong_col : str
        Name of elongation column.

    Returns
    -------
    windows_out : pd.DataFrame
        Copy of windows_df with:
        - phase_expected
        - phase_ratio
    binned : pd.DataFrame
        Binned baseline table with elong_med, phase_med, phase_p25, phase_p75, n,
        and phase_baseline.
    """
    if config is None:
        config = BaselineConfig()

    required = {phase_col, elong_col}
    if not required.issubset(windows_df.columns):
        raise ValueError(f"windows_df must contain columns {required}")

    windows_out = windows_df.copy()

    w = windows_out.dropna(subset=[phase_col, elong_col]).copy()
    w = w[w[phase_col] > 0].copy()

    if w.empty:
        raise ValueError("No valid windows remain for baseline construction.")

    bins = np.arange(0, config.max_elong_deg + config.bin_width_deg, config.bin_width_deg)

    w["elong_bin"] = pd.cut(
        w[elong_col],
        bins=bins,
        include_lowest=True
    )

    binned = (
        w.groupby("elong_bin", observed=False)
        .agg(
            elong_med=(elong_col, "median"),
            phase_med=(phase_col, "median"),
            phase_p25=(phase_col, lambda x: safe_percentile(x, 25)),
            phase_p75=(phase_col, lambda x: safe_percentile(x, 75)),
            n=(phase_col, "count")
        )
        .reset_index(drop=True)
    )

    binned = binned.dropna(subset=["elong_med", "phase_med"]).copy()

    if binned.empty:
        raise ValueError("No valid elongation bins were produced for baseline fitting.")

    # Hybrid quiet baseline from your existing method
    binned["phase_baseline"] = 0.5 * binned["phase_p25"] + 0.5 * binned["phase_med"]

    interp_func = interp1d(
        binned["elong_med"],
        binned["phase_baseline"],
        bounds_error=False,
        fill_value="extrapolate"
    )

    w["phase_expected"] = interp_func(w[elong_col])
    w["phase_ratio"] = w[phase_col] / w["phase_expected"]

    windows_out["phase_expected"] = np.nan
    windows_out["phase_ratio"] = np.nan
    windows_out.loc[w.index, "phase_expected"] = w["phase_expected"]
    windows_out.loc[w.index, "phase_ratio"] = w["phase_ratio"]

    return windows_out, binned


# ============================================================
# BLOCK 4 — CIR DETECTION
# ============================================================

def detect_cir_regions(
    windows_df: pd.DataFrame,
    config: Optional[CIRConfig] = None,
    mid_col: str = "mid",
    ratio_col: str = "phase_ratio",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect CIR-like long-duration regions from smoothed phase_ratio.

    This follows the working Block 4 logic:
    - 12 hr rolling median smoothing
    - hysteresis thresholding
    - minimum duration filtering

    Parameters
    ----------
    windows_df : pd.DataFrame
        Must contain `mid_col` and `ratio_col`.
    config : CIRConfig, optional
        CIR settings.
    mid_col : str
        Midpoint datetime column.
    ratio_col : str
        Normalized phase ratio column.

    Returns
    -------
    windows_out : pd.DataFrame
        Copy of input with:
        - phase_smooth
    cir_df : pd.DataFrame
        Detected CIR regions with:
        - start, end, duration_hr, median_signal, peak_signal
    """
    if config is None:
        config = CIRConfig()

    required = {mid_col, ratio_col}
    if not required.issubset(windows_df.columns):
        raise ValueError(f"windows_df must contain columns {required}")

    windows_out = windows_df.copy()

    w = windows_out.dropna(subset=[ratio_col]).sort_values(mid_col).copy()
    if w.empty:
        raise ValueError("No valid windows remain for CIR detection.")

    smooth_n = max(1, int(config.window_hours * 60 / config.step_min))

    w["phase_smooth"] = (
        w[ratio_col]
        .rolling(window=smooth_n, center=True, min_periods=1)
        .median()
    )

    in_region = False
    regions: list[dict] = []
    current: Optional[dict] = None

    for _, row in w.iterrows():
        val = row["phase_smooth"]

        if not in_region:
            if val > config.thresh_on:
                in_region = True
                current = {
                    "start": row[mid_col],
                    "end": row[mid_col],
                    "values": [val],
                }
        else:
            if val > config.thresh_off:
                current["end"] = row[mid_col]
                current["values"].append(val)
            else:
                regions.append(current)
                in_region = False
                current = None

    if current is not None:
        regions.append(current)

    clean_regions = []
    for r in regions:
        duration = (r["end"] - r["start"]).total_seconds() / 3600.0

        if duration >= config.min_duration_hr:
            clean_regions.append({
                "start": r["start"],
                "end": r["end"],
                "duration_hr": duration,
                "median_signal": float(np.median(r["values"])),
                "peak_signal": float(np.max(r["values"])),
            })

    cir_df = pd.DataFrame(clean_regions)

    windows_out["phase_smooth"] = np.nan
    windows_out.loc[w.index, "phase_smooth"] = w["phase_smooth"]

    return windows_out, cir_df


# ============================================================
# BLOCK 5 — TRANSIENT / CME-LIKE DETECTION
# ============================================================

def detect_transient_events(
    windows_df: pd.DataFrame,
    config: Optional[TransientConfig] = None,
    mid_col: str = "mid",
    ratio_col: str = "phase_ratio",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect transient / CME-like events after CIR-scale background removal.

    This follows the working Block 5 logic:
    - rebuild CIR-scale background from rolling median of phase_ratio
    - clean_signal = phase_ratio / phase_smooth
    - threshold clean_signal
    - group contiguous events
    - filter by duration

    IMPORTANT:
    To reproduce your working result, use step_min=20 here.

    Parameters
    ----------
    windows_df : pd.DataFrame
        Must contain `mid_col` and `ratio_col`.
    config : TransientConfig, optional
        Transient settings.
    mid_col : str
        Midpoint datetime column.
    ratio_col : str
        Normalized phase ratio column.

    Returns
    -------
    windows_out : pd.DataFrame
        Copy of input with:
        - phase_smooth
        - clean_signal
        - event_flag
    events_df : pd.DataFrame
        Transient / CME-like event table.
    """
    if config is None:
        config = TransientConfig()

    required = {mid_col, ratio_col}
    if not required.issubset(windows_df.columns):
        raise ValueError(f"windows_df must contain columns {required}")

    windows_out = windows_df.copy()

    w = windows_out.dropna(subset=[ratio_col]).sort_values(mid_col).copy()
    if w.empty:
        raise ValueError("No valid windows remain for transient detection.")

    smooth_n = max(1, int(config.window_hours * 60 / config.step_min))

    w["phase_smooth"] = (
        w[ratio_col]
        .rolling(window=smooth_n, center=True, min_periods=1)
        .median()
    )

    w["clean_signal"] = w[ratio_col] / w["phase_smooth"]

    w = w.replace([np.inf, -np.inf], np.nan)
    w = w.dropna(subset=["clean_signal"]).copy()

    w["event_flag"] = w["clean_signal"] > config.threshold

    events: list[dict] = []
    current: Optional[dict] = None

    for _, row in w.iterrows():
        if row["event_flag"]:
            if current is None:
                current = {
                    "start": row[mid_col],
                    "end": row[mid_col],
                    "values": [row["clean_signal"]],
                    "raw_phase": [row[ratio_col]],
                }
            else:
                current["end"] = row[mid_col]
                current["values"].append(row["clean_signal"])
                current["raw_phase"].append(row[ratio_col])
        else:
            if current is not None:
                events.append(current)
                current = None

    if current is not None:
        events.append(current)

    final_events = []
    for e in events:
        duration_hr = (e["end"] - e["start"]).total_seconds() / 3600.0

        if config.min_duration_hr < duration_hr < config.max_duration_hr:
            final_events.append({
                "start": e["start"],
                "end": e["end"],
                "duration_hr": duration_hr,
                "peak_clean": float(np.max(e["values"])),
                "median_clean": float(np.median(e["values"])),
                "peak_phase": float(np.max(e["raw_phase"])),
            })

    events_df = pd.DataFrame(final_events)

    windows_out["phase_smooth"] = np.nan
    windows_out["clean_signal"] = np.nan
    windows_out["event_flag"] = False

    windows_out.loc[w.index, "phase_smooth"] = w["phase_smooth"]
    windows_out.loc[w.index, "clean_signal"] = w["clean_signal"]
    windows_out.loc[w.index, "event_flag"] = w["event_flag"]

    return windows_out, events_df