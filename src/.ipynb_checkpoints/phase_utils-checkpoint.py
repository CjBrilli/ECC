"""
phase_utils.py

Helpers for PSD-based phase scintillation from DSN Doppler residuals.

Current workflow:
- regularize Doppler cadence (typically 10 s)
- integrate Doppler to phase
- compute Welch PSD in windows
- integrate band-limited phase power to get phase RMS
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import detrend, welch

from .doppler_utils import resample_numeric_time_series


def prepare_phase_input(
    df: pd.DataFrame,
    time_col: str = "UTC_time",
    doppler_col: str = "doppler",
    dt_target_sec: float = 10.0,
) -> pd.DataFrame:
    """
    Prepare regularly sampled Doppler dataframe for phase analysis.

    Steps:
    - set datetime index
    - resample to fixed cadence
    - keep only rows with Doppler

    Returns
    -------
    pd.DataFrame
        Indexed by UTC_time
    """
    rule = f"{int(dt_target_sec)}s"

    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}")
    if doppler_col not in df.columns:
        raise ValueError(f"Missing Doppler column: {doppler_col}")

    out = resample_numeric_time_series(df, time_col=time_col, rule=rule)
    out = out.dropna(subset=[doppler_col]).copy()

    if out.empty:
        raise ValueError("No phase input samples remain after resampling and gap removal.")

    return out


def doppler_to_phase(
    doppler_hz: pd.Series | np.ndarray,
    dt_sec: float,
) -> np.ndarray:
    """
    Convert Doppler frequency residuals in Hz to accumulated phase in radians.

    phi(t) = 2*pi*integral(f_D dt)
    discrete approximation:
        phase = 2*pi*cumsum(doppler * dt)
    """
    doppler = np.asarray(doppler_hz, dtype=float)
    return 2.0 * np.pi * np.cumsum(doppler) * dt_sec


def add_phase_column(
    df_phase: pd.DataFrame,
    doppler_col: str = "doppler",
    dt_sec: float = 10.0,
    out_col: str = "phase_rad",
) -> pd.DataFrame:
    """
    Add integrated phase column to a regularly sampled dataframe.
    """
    out = df_phase.copy()
    if doppler_col not in out.columns:
        raise ValueError(f"Missing Doppler column: {doppler_col}")

    out[out_col] = doppler_to_phase(out[doppler_col].values, dt_sec=dt_sec)
    return out


def compute_band_limited_phase_rms(
    phase_seg_rad: np.ndarray,
    fs_hz: float,
    f_low_hz: float = 3e-4,
    f_high_hz: float = 3e-2,
    detrend_type: str = "linear",
    nperseg_max: int = 256,
) -> float:
    """
    Compute band-limited phase RMS from a phase segment.

    Steps:
    - detrend phase segment
    - compute Welch PSD
    - integrate power in chosen band
    - return sqrt(power)

    Returns
    -------
    float
        Phase RMS in radians
    """
    if len(phase_seg_rad) < 2:
        return np.nan

    phase_seg_rad = np.asarray(phase_seg_rad, dtype=float)
    phase_dt = detrend(phase_seg_rad, type=detrend_type)

    f, pxx = welch(
        phase_dt,
        fs=fs_hz,
        nperseg=min(nperseg_max, len(phase_dt)),
    )

    mask = (f >= f_low_hz) & (f <= f_high_hz)
    if mask.sum() == 0:
        return np.nan

    band_power = np.trapz(pxx[mask], f[mask])
    return float(np.sqrt(band_power))


def compute_phase_rms_windows(
    df: pd.DataFrame,
    time_col: str = "UTC_time",
    doppler_col: str = "doppler",
    dt_target_sec: float = 10.0,
    window_min: int = 20,
    step_min: int = 10,
    min_samples: int = 50,
    f_low_hz: float = 3e-4,
    f_high_hz: float = 3e-2,
    phase_col: str = "phase_rad",
) -> pd.DataFrame:
    """
    End-to-end computation of windowed PSD-based phase scintillation.

    Workflow mirrors the current notebook:
    - resample to ~10 s
    - convert Doppler -> phase
    - slide 20 min windows with 10 min step
    - detrend phase in each window
    - compute Welch PSD and integrate selected band

    Returns
    -------
    pd.DataFrame
        Columns:
        - start
        - end
        - mid
        - phase_rms_rad
        - n_samples
    """
    df_phase = prepare_phase_input(
        df,
        time_col=time_col,
        doppler_col=doppler_col,
        dt_target_sec=dt_target_sec,
    )

    df_phase = add_phase_column(
        df_phase,
        doppler_col=doppler_col,
        dt_sec=dt_target_sec,
        out_col=phase_col,
    )

    times = df_phase.index
    t0 = times.min()
    t_end = times.max()

    window_sec = int(window_min * 60)
    step_sec = int(step_min * 60)
    fs_hz = 1.0 / dt_target_sec

    results: list[dict[str, object]] = []

    while t0 + pd.Timedelta(seconds=window_sec) <= t_end:
        t1 = t0 + pd.Timedelta(seconds=window_sec)
        sub = df_phase[(df_phase.index >= t0) & (df_phase.index < t1)]

        if len(sub) < min_samples:
            t0 += pd.Timedelta(seconds=step_sec)
            continue

        phase_rms = compute_band_limited_phase_rms(
            sub[phase_col].values,
            fs_hz=fs_hz,
            f_low_hz=f_low_hz,
            f_high_hz=f_high_hz,
        )

        results.append(
            {
                "start": t0,
                "end": t1,
                "mid": t0 + (t1 - t0) / 2,
                "phase_rms_rad": phase_rms,
                "n_samples": int(len(sub)),
            }
        )

        t0 += pd.Timedelta(seconds=step_sec)

    windows_df = pd.DataFrame(results)

    if windows_df.empty:
        raise ValueError("No phase RMS windows were produced.")

    return windows_df.sort_values("mid").reset_index(drop=True)


def print_phase_summary(windows_df: pd.DataFrame) -> None:
    """
    Convenience notebook summary for phase-window output.
    """
    print("Windows created:", len(windows_df))
    if not windows_df.empty:
        print("Window time range:", windows_df["start"].min(), "→", windows_df["end"].max())
        if "phase_rms_rad" in windows_df.columns:
            valid = windows_df["phase_rms_rad"].dropna()
            if len(valid) > 0:
                print("Phase RMS range (rad):", valid.min(), "→", valid.max())