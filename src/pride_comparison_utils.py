"""
pride_comparison_utils.py

Utilities for comparing DSN Doppler-derived phase scintillation
with PRIDE/scintillation measurements in common 20-minute bins.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import pearsonr, spearmanr, kendalltau


# ============================================================
# LOADERS
# ============================================================

def load_dsn_doppler_file(path: str | Path) -> pd.DataFrame:
    """
    Load DSN Doppler file with UTC_time and doppler columns.
    """
    df = pd.read_csv(path, sep=r"\s+", header=0)

    if "UTC_time" not in df.columns:
        raise ValueError("DSN file missing UTC_time column")
    if "doppler" not in df.columns:
        raise ValueError("DSN file missing doppler column")

    df["UTC_time"] = pd.to_datetime(df["UTC_time"], errors="coerce")
    df["doppler"] = pd.to_numeric(df["doppler"], errors="coerce")

    df = (
        df.dropna(subset=["UTC_time", "doppler"])
        .sort_values("UTC_time")
        .set_index("UTC_time")
    )

    df = df[~df.index.duplicated(keep="first")]
    return df


def load_pride_scint_file(path: str | Path) -> pd.DataFrame:
    """
    Load PRIDE/scintillation file with UTC and Scint_rad columns.
    """
    df = pd.read_csv(path, sep=r"\s+", header=0)

    if "UTC" not in df.columns:
        raise ValueError("PRIDE file missing UTC column")
    if "Scint_rad" not in df.columns:
        raise ValueError("PRIDE file missing Scint_rad column")

    df["UTC"] = pd.to_datetime(df["UTC"], errors="coerce")
    df["Scint_rad"] = pd.to_numeric(df["Scint_rad"], errors="coerce")

    df = (
        df.dropna(subset=["UTC", "Scint_rad"])
        .sort_values("UTC")
        .set_index("UTC")
    )

    df = df[~df.index.duplicated(keep="first")]
    return df


# ============================================================
# OVERLAP HANDLING
# ============================================================

def get_common_days(
    dsn_df: pd.DataFrame,
    pride_df: pd.DataFrame,
) -> pd.DatetimeIndex:
    """
    Return UTC days present in both DSN and PRIDE files.
    """
    dsn_days = dsn_df.index.normalize().unique()
    pride_days = pride_df.index.normalize().unique()

    return pd.to_datetime(
        sorted(list(set(dsn_days) & set(pride_days)))
    )


def get_overlapping_day_data(
    dsn_df: pd.DataFrame,
    pride_df: pd.DataFrame,
    day,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Return overlapping DSN and PRIDE data for one UTC day.
    """
    day = pd.to_datetime(day).normalize()

    dsn_day = dsn_df[dsn_df.index.normalize() == day]
    pride_day = pride_df[pride_df.index.normalize() == day]

    if dsn_day.empty or pride_day.empty:
        return None, None

    start = max(dsn_day.index.min(), pride_day.index.min())
    end = min(dsn_day.index.max(), pride_day.index.max())

    if start >= end:
        return None, None

    dsn_overlap = dsn_day[(dsn_day.index >= start) & (dsn_day.index <= end)]
    pride_overlap = pride_day[(pride_day.index >= start) & (pride_day.index <= end)]

    return dsn_overlap, pride_overlap


# ============================================================
# DSN DOPPLER -> PHASE SCINTILLATION
# ============================================================

def compute_sigma_phi(
    t_seconds,
    doppler_hz,
    f_low_hz: float = 3e-3,
    f_high_hz: float = 0.1,
    detrend_poly_order: int = 4,
    min_samples: int = 16,
) -> float:
    """
    Compute Doppler-derived phase scintillation sigma_phi.

    Steps:
    1. Convert Doppler residuals to phase:
       phi(t) = 2*pi * integral doppler(t) dt
    2. Detrend phase with a polynomial
    3. Estimate PSD with Welch
    4. Integrate PSD over f_low_hz to f_high_hz

    Returns sigma_phi in radians.
    """
    t = np.asarray(t_seconds, dtype=float)
    y = np.asarray(doppler_hz, dtype=float)

    good = np.isfinite(t) & np.isfinite(y)
    t = t[good]
    y = y[good]

    if len(t) < min_samples:
        return np.nan

    dt = np.nanmedian(np.diff(t))

    if not np.isfinite(dt) or dt <= 0:
        return np.nan

    fs = 1.0 / dt

    phase = 2.0 * np.pi * np.cumsum(y * dt)

    x = t - t[0]

    if detrend_poly_order > 0 and len(phase) > detrend_poly_order + 2:
        coeff = np.polyfit(x, phase, detrend_poly_order)
        phase = phase - np.polyval(coeff, x)
    else:
        phase = phase - np.nanmean(phase)

    nperseg = min(2048, len(phase))

    f, pxx = welch(
        phase,
        fs=fs,
        nperseg=nperseg,
        window="hann",
        noverlap=nperseg // 2,
    )

    band = (f >= f_low_hz) & (f <= f_high_hz)

    if not np.any(band):
        return np.nan

    power = np.trapezoid(pxx[band], f[band])
    return float(np.sqrt(power))


def compute_sigma_phi_binned(
    dsn_day: pd.DataFrame,
    bin_freq: str = "20min",
    doppler_col: str = "doppler",
    f_low_hz: float = 3e-3,
    f_high_hz: float = 0.1,
    detrend_poly_order: int = 4,
    min_samples: int = 16,
) -> pd.DataFrame:
    """
    Compute DSN sigma_phi in fixed time bins.
    """
    rows = []

    for start, chunk in dsn_day[doppler_col].resample(bin_freq):
        chunk = chunk.dropna()

        if len(chunk) < min_samples:
            continue

        t_seconds = (chunk.index - chunk.index[0]).total_seconds().values

        sigma_phi = compute_sigma_phi(
            t_seconds,
            chunk.values,
            f_low_hz=f_low_hz,
            f_high_hz=f_high_hz,
            detrend_poly_order=detrend_poly_order,
            min_samples=min_samples,
        )

        rows.append({
            "time": start,
            "dsn_sigma_phi_rad": sigma_phi,
            "n_dsn_samples": len(chunk),
        })

    if not rows:
        return pd.DataFrame(
            columns=["time", "dsn_sigma_phi_rad", "n_dsn_samples"]
        ).set_index("time")

    return pd.DataFrame(rows).set_index("time")


# ============================================================
# ALIGN DSN AND PRIDE IN COMMON 20-MIN BINS
# ============================================================

def build_dsn_pride_binned_comparison(
    dsn_df: pd.DataFrame,
    pride_df: pd.DataFrame,
    common_days=None,
    bin_freq: str = "20min",
    f_low_hz: float = 3e-3,
    f_high_hz: float = 0.1,
    detrend_poly_order: int = 4,
    min_samples: int = 16,
) -> pd.DataFrame:
    """
    Build a common-bin comparison table:

    index: 20-min bin start
    columns:
      dsn_sigma_phi_rad
      pride_scint_rad
      n_dsn_samples
      day
    """
    if common_days is None:
        common_days = get_common_days(dsn_df, pride_df)

    all_bins = []

    for day in common_days:
        dsn_day, pride_day = get_overlapping_day_data(dsn_df, pride_df, day)

        if dsn_day is None:
            continue

        dsn_bins = compute_sigma_phi_binned(
            dsn_day,
            bin_freq=bin_freq,
            f_low_hz=f_low_hz,
            f_high_hz=f_high_hz,
            detrend_poly_order=detrend_poly_order,
            min_samples=min_samples,
        )

        if dsn_bins.empty:
            continue

        pride_bins = (
            pride_day["Scint_rad"]
            .resample(bin_freq)
            .mean()
            .rename("pride_scint_rad")
        )

        paired = dsn_bins.join(pride_bins, how="inner").dropna()

        if paired.empty:
            continue

        paired["day"] = pd.to_datetime(day).normalize()
        all_bins.append(paired)

    if not all_bins:
        return pd.DataFrame(
            columns=[
                "dsn_sigma_phi_rad",
                "n_dsn_samples",
                "pride_scint_rad",
                "day",
            ]
        )

    return pd.concat(all_bins).sort_index()


def build_daily_dsn_pride_summary(
    binned_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Average common-bin DSN and PRIDE measurements per day.
    """
    if binned_df.empty:
        return pd.DataFrame(
            columns=[
                "day",
                "dsn_sigma_phi_rad",
                "pride_scint_rad",
                "n_bins",
            ]
        )

    daily = (
        binned_df
        .groupby("day")
        .agg(
            dsn_sigma_phi_rad=("dsn_sigma_phi_rad", "mean"),
            pride_scint_rad=("pride_scint_rad", "mean"),
            n_bins=("dsn_sigma_phi_rad", "count"),
        )
        .reset_index()
    )

    return daily


# ============================================================
# STATISTICS
# ============================================================

def compute_signal_correlations(
    df: pd.DataFrame,
    x_col: str = "pride_scint_rad",
    y_col: str = "dsn_sigma_phi_rad",
) -> dict:
    """
    Compute Pearson, Spearman, and Kendall correlations.
    """
    sub = df[[x_col, y_col]].dropna()

    if len(sub) < 4:
        return {
            "n": len(sub),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "kendall_tau": np.nan,
            "kendall_p": np.nan,
        }

    x = sub[x_col].values
    y = sub[y_col].values

    r, p_r = pearsonr(x, y)
    rho, p_s = spearmanr(x, y)
    tau, p_k = kendalltau(x, y)

    return {
        "n": len(sub),
        "pearson_r": r,
        "pearson_p": p_r,
        "spearman_rho": rho,
        "spearman_p": p_s,
        "kendall_tau": tau,
        "kendall_p": p_k,
    }

# ============================================================
# CROSS-CORRELATION
# ============================================================

def normalised_xcorr(x, y):
    """
    Normalised cross-correlation for two equal-length finite series.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return None, None

    x = x - np.mean(x)
    y = y - np.mean(y)

    if np.std(x) == 0 or np.std(y) == 0:
        return None, None

    corr = np.correlate(x, y, mode="full")
    corr = corr / (np.std(x) * np.std(y) * len(x))

    lags = np.arange(-len(x) + 1, len(x))

    return lags, corr


def compute_xcorr_summary(
    binned_df: pd.DataFrame,
    day_col: str = "day",
    dsn_col: str = "dsn_sigma_phi_rad",
    pride_col: str = "pride_scint_rad",
    bin_minutes: int = 20,
    min_bins: int = 4,
) -> pd.DataFrame:
    """
    Compute per-day DSN/PRIDE cross-correlation summary.

    Days with fewer than min_bins are retained but marked as unused.
    """
    rows = []

    for day, sub in binned_df.groupby(day_col):
        sub = sub.sort_index()
        n_bins = len(sub)

        if n_bins < min_bins:
            rows.append({
                "day": pd.to_datetime(day),
                "n_bins": n_bins,
                "used_for_summary": False,
                "best_lag_bins": np.nan,
                "best_lag_minutes": np.nan,
                "best_corr": np.nan,
                "zero_lag_corr": np.nan,
            })
            continue

        lags, corr = normalised_xcorr(
            sub[dsn_col].values,
            sub[pride_col].values,
        )

        if lags is None:
            rows.append({
                "day": pd.to_datetime(day),
                "n_bins": n_bins,
                "used_for_summary": False,
                "best_lag_bins": np.nan,
                "best_lag_minutes": np.nan,
                "best_corr": np.nan,
                "zero_lag_corr": np.nan,
            })
            continue

        best_idx = np.nanargmax(corr)
        best_lag_bins = lags[best_idx]

        rows.append({
            "day": pd.to_datetime(day),
            "n_bins": n_bins,
            "used_for_summary": True,
            "best_lag_bins": best_lag_bins,
            "best_lag_minutes": best_lag_bins * bin_minutes,
            "best_corr": corr[best_idx],
            "zero_lag_corr": corr[lags == 0][0],
        })

    return pd.DataFrame(rows).sort_values("day").reset_index(drop=True)


def get_xcorr_for_day(
    binned_df: pd.DataFrame,
    day,
    bin_minutes: int = 20,
    dsn_col: str = "dsn_sigma_phi_rad",
    pride_col: str = "pride_scint_rad",
):
    """
    Return lag/correlation arrays for one day.
    """
    day = pd.to_datetime(day)
    sub = binned_df[binned_df["day"] == day].sort_index()

    lags, corr = normalised_xcorr(
        sub[dsn_col].values,
        sub[pride_col].values,
    )

    if lags is None:
        return None, None, None

    lags_minutes = lags * bin_minutes
    best_idx = np.nanargmax(corr)
    best_lag_minutes = lags_minutes[best_idx]

    return lags_minutes, corr, best_lag_minutes