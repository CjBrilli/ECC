"""
doppler_utils.py

Helpers for DSN Doppler residual processing.

This module covers:
- Hz -> mm/s conversion
- time resampling
- daily detrended Doppler RMS
- DSN solar scintillation model
- smoothing and decimal-year helpers
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import detrend


DEFAULT_F_CARRIER_HZ = 8.4e9
DEFAULT_C_MPS = 299792458.0


def hz_to_mm_s(
    doppler_hz: pd.Series | np.ndarray,
    f_carrier_hz: float = DEFAULT_F_CARRIER_HZ,
    c_mps: float = DEFAULT_C_MPS,
) -> pd.Series | np.ndarray:
    """
    Convert Doppler residuals from Hz to mm/s.

    v = (c / f_c) * df, then convert m/s to mm/s.
    """
    factor = (c_mps / f_carrier_hz) * 1000.0
    return doppler_hz * factor


def add_doppler_unit_columns(
    df: pd.DataFrame,
    f_carrier_hz: float = DEFAULT_F_CARRIER_HZ,
    c_mps: float = DEFAULT_C_MPS,
) -> pd.DataFrame:
    """
    Add convenience unit-converted columns to a DSN dataframe.

    Adds:
    - doppler_mm_s
    - tropo_mm_s (if tropo exists)
    """
    out = df.copy()
    out["doppler_mm_s"] = hz_to_mm_s(out["doppler"], f_carrier_hz, c_mps)

    if "tropo" in out.columns:
        out["tropo_mm_s"] = hz_to_mm_s(out["tropo"], f_carrier_hz, c_mps)

    return out


def resample_numeric_time_series(
    df: pd.DataFrame,
    time_col: str = "UTC_time",
    rule: str = "60s",
) -> pd.DataFrame:
    """
    Resample all numeric columns on a regular time grid.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `time_col`.
    time_col : str
        Name of datetime column.
    rule : str
        Pandas resampling rule, e.g. '60s' or '10s'.

    Returns
    -------
    pd.DataFrame
        Resampled dataframe indexed by datetime.
    """
    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}")

    out = (
        df.set_index(time_col)
        .sort_index()
        .resample(rule)
        .mean(numeric_only=True)
    )

    return out


def compute_daily_doppler_rms(
    df: pd.DataFrame,
    time_col: str = "UTC_time",
    f_carrier_hz: float = DEFAULT_F_CARRIER_HZ,
    c_mps: float = DEFAULT_C_MPS,
    resample_rule: str = "60s",
    min_samples_per_day: int = 10,
    add_tropo_diagnostic: bool = True,
) -> pd.DataFrame:
    """
    Compute daily detrended Doppler RMS in mm/s.

    Workflow mirrors the current notebook logic:
    - convert Doppler to mm/s
    - resample to 60 s
    - group by day
    - linearly detrend each day's Doppler series
    - compute RMS of detrended series

    If tropo exists and add_tropo_diagnostic=True, also computes the
    notebook's current 'tropo-derived rate RMS' diagnostic.

    Returns
    -------
    pd.DataFrame
        Columns include:
        - day
        - doppler_rms_mm_s
        - n_60s_samples
        - tropo_rate_rms (optional)
    """
    work = add_doppler_unit_columns(df, f_carrier_hz=f_carrier_hz, c_mps=c_mps)
    df_resampled = resample_numeric_time_series(work, time_col=time_col, rule=resample_rule)

    results: list[list[object]] = []

    for day, group in df_resampled.groupby(df_resampled.index.floor("D")):
        doppler = group["doppler_mm_s"].dropna().values

        if len(doppler) < min_samples_per_day:
            continue

        doppler_dt = detrend(doppler, type="linear")
        doppler_rms = float(np.sqrt(np.mean(doppler_dt**2)))

        results.append([day, doppler_rms, int(len(doppler))])

    daily_df = pd.DataFrame(
        results,
        columns=["day", "doppler_rms_mm_s", "n_60s_samples"],
    )

    if daily_df.empty:
        raise ValueError("No daily Doppler RMS values were produced.")

    if add_tropo_diagnostic and "tropo_mm_s" in df_resampled.columns:
        tropo_rows: list[list[object]] = []
        dt_sec = pd.to_timedelta(resample_rule).total_seconds()

        for day, group in df_resampled.groupby(df_resampled.index.floor("D")):
            tropo = group["tropo_mm_s"].dropna().values
            if len(tropo) < min_samples_per_day:
                continue

            tropo_dt = detrend(tropo, type="linear")
            tropo_rate = np.gradient(tropo_dt, dt_sec)
            tropo_rms = float(np.sqrt(np.mean(tropo_rate**2)))
            tropo_rows.append([day, tropo_rms])

        if tropo_rows:
            tropo_df = pd.DataFrame(tropo_rows, columns=["day", "tropo_rate_rms"])
            daily_df = daily_df.merge(tropo_df, on="day", how="left")

    return daily_df.sort_values("day").reset_index(drop=True)


def merge_daily_sep(
    daily_df: pd.DataFrame,
    horizons_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge daily SEP values into the daily RMS dataframe.
    """
    required_daily = {"day"}
    required_sep = {"day", "elongation_deg"}

    if not required_daily.issubset(daily_df.columns):
        raise ValueError(f"daily_df must contain columns: {required_daily}")
    if not required_sep.issubset(horizons_daily.columns):
        raise ValueError(f"horizons_daily must contain columns: {required_sep}")

    out = daily_df.merge(horizons_daily, on="day", how="left")
    return out


def dsn_solar_scintillation_mm_s(
    sep_deg: pd.Series | np.ndarray,
    f_carrier_hz: float = DEFAULT_F_CARRIER_HZ,
    T_int_sec: float = 60.0,
    c_mps: float = DEFAULT_C_MPS,
    C_band: float = 1.9e-6,
) -> np.ndarray:
    """
    DSN solar scintillation model in mm/s.

    Returns
    -------
    np.ndarray
        sigma_v in mm/s
    """
    sep = np.asarray(sep_deg, dtype=float)

    sigma_v2 = np.full_like(sep, np.nan, dtype=float)

    valid = np.isfinite(sep)
    if not np.any(valid):
        return sigma_v2

    sep_valid = np.clip(sep[valid], 1e-6, 180.0)
    sep_rad = np.deg2rad(sep_valid)

    mask_low = sep_valid <= 90.0
    mask_high = sep_valid > 90.0

    sigma_valid = np.full_like(sep_valid, np.nan, dtype=float)

    sigma_valid[mask_low] = (
        0.53
        * C_band
        * c_mps**2
        / (f_carrier_hz**2 * T_int_sec**0.35 * (np.sin(sep_rad[mask_low]) ** 2.45))
    )

    sigma_valid[mask_high] = (
        0.53
        * C_band
        * c_mps**2
        / (f_carrier_hz**2 * T_int_sec**0.35)
    )

    sigma_v2[valid] = sigma_valid
    sigma_v_mps = np.sqrt(sigma_v2)
    return sigma_v_mps * 1000.0


def add_decimal_year(
    df: pd.DataFrame,
    day_col: str = "day",
    out_col: str = "decimal_year",
) -> pd.DataFrame:
    """
    Add decimal year column using day-of-year / 365.25.
    """
    out = df.copy()
    if day_col not in out.columns:
        raise ValueError(f"Missing day column: {day_col}")

    out[out_col] = (
        out[day_col].dt.year
        + (out[day_col].dt.dayofyear - 1) / 365.25
    )
    return out


def add_rolling_median(
    df: pd.DataFrame,
    value_col: str,
    window: int = 7,
    center: bool = True,
    min_periods: int = 1,
    out_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add rolling median smoothing column.
    """
    out = df.copy()
    if value_col not in out.columns:
        raise ValueError(f"Missing value column: {value_col}")

    if out_col is None:
        out_col = f"{value_col}_smooth"

    out[out_col] = (
        out[value_col]
        .rolling(window, center=center, min_periods=min_periods)
        .median()
    )
    return out


def prepare_daily_rms_table(
    dsn_df: pd.DataFrame,
    horizons_daily: pd.DataFrame,
    f_carrier_hz: float = DEFAULT_F_CARRIER_HZ,
    c_mps: float = DEFAULT_C_MPS,
    T_int_sec: float = 60.0,
    C_band: float = 1.9e-6,
    resample_rule: str = "60s",
    min_samples_per_day: int = 10,
    smooth_days: int = 7,
    add_tropo_diagnostic: bool = True,
) -> pd.DataFrame:
    """
    End-to-end helper for the current daily RMS notebook block.

    Produces a daily dataframe with:
    - doppler_rms_mm_s
    - n_60s_samples
    - elongation_deg
    - solar_model_mm_s
    - decimal_year
    - doppler_smooth_mm_s
    - solar_smooth_mm_s
    - tropo_smooth (if available)
    """
    daily_df = compute_daily_doppler_rms(
        dsn_df,
        f_carrier_hz=f_carrier_hz,
        c_mps=c_mps,
        resample_rule=resample_rule,
        min_samples_per_day=min_samples_per_day,
        add_tropo_diagnostic=add_tropo_diagnostic,
    )

    daily_df = merge_daily_sep(daily_df, horizons_daily)

    daily_df["solar_model_mm_s"] = dsn_solar_scintillation_mm_s(
        daily_df["elongation_deg"].values,
        f_carrier_hz=f_carrier_hz,
        T_int_sec=T_int_sec,
        c_mps=c_mps,
        C_band=C_band,
    )

    daily_df = add_decimal_year(daily_df, day_col="day", out_col="decimal_year")
    daily_df = add_rolling_median(
        daily_df,
        value_col="doppler_rms_mm_s",
        window=smooth_days,
        out_col="doppler_smooth_mm_s",
    )
    daily_df = add_rolling_median(
        daily_df,
        value_col="solar_model_mm_s",
        window=smooth_days,
        out_col="solar_smooth_mm_s",
    )

    if "tropo_rate_rms" in daily_df.columns:
        daily_df = add_rolling_median(
            daily_df,
            value_col="tropo_rate_rms",
            window=smooth_days,
            out_col="tropo_smooth",
        )

    return daily_df.sort_values("day").reset_index(drop=True)


def print_daily_summary(daily_df: pd.DataFrame) -> None:
    """
    Convenience notebook summary for daily RMS outputs.
    """
    print("\nDaily RMS rows:", len(daily_df))
    print("Daily RMS time range:", daily_df["day"].min(), "→", daily_df["day"].max())

    if "doppler_rms_mm_s" in daily_df.columns:
        print(
            "Measured Doppler RMS range (mm/s):",
            np.nanmin(daily_df["doppler_rms_mm_s"]),
            "→",
            np.nanmax(daily_df["doppler_rms_mm_s"]),
        )

    if "elongation_deg" in daily_df.columns:
        n_missing = int(daily_df["elongation_deg"].isna().sum())
        print("Missing SEP after daily merge:", n_missing)
        valid = daily_df["elongation_deg"].dropna()
        if len(valid) > 0:
            print("SEP range (deg):", np.nanmin(valid), "→", np.nanmax(valid))
            