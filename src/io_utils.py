"""
io_utils.py

Input / output helpers for DSN Doppler and Horizons geometry files.

Designed around the current 2010/2011 workflow:
- DSN Doppler residuals in text files with columns such as UTC_time, doppler,
  and optional valid, elev, tropo.
- JPL Horizons-style geometry text files with $$SOE ... $$EOE blocks.

Author: project refactor for reusable multi-year workflow
version 1.1 changed horizons loader
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _check_file_exists(filepath: str | Path) -> Path:
    """
    Validate that a file exists and return it as a Path object.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def load_horizons_daily_sep(filepath: str | Path) -> pd.DataFrame:
    """
    Load daily solar elongation (SEP) from a JPL Horizons text file.

    Expected format
    ---------------
    - Plain text file exported from JPL Horizons
    - Contains a data block between:
        $$SOE
        ...
        $$EOE

    - Each ephemeris row begins with:
        YYYY-Mon-DD HH:MM

    - Solar elongation is read from the Horizons field labelled 'S-O-T /r'
      and typically appears in the row as:

        <elongation_deg> /L
        or
        <elongation_deg> /T

      For example:
        2.6407 /L

    Parsing assumptions
    -------------------
    - The first two columns are the UTC date and time
    - The elongation is the numeric token immediately before '/L' or '/T'
    - Intermediate columns are ignored

    Output
    ------
    pandas.DataFrame with columns:
        - day : datetime64[ns], daily resolution
        - elongation_deg : float

    Notes
    -----
    - If multiple entries exist within a day, the daily median is returned
    - Rows with malformed timestamps or missing elongation are skipped
    """
    path = _check_file_exists(filepath)

    rows: list[list[object]] = []
    in_data_block = False

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip()

            if "$$SOE" in line:
                in_data_block = True
                continue

            if "$$EOE" in line:
                break

            if not in_data_block:
                continue

            if not line.strip():
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            # Parse datetime from first two tokens
            dt_str = f"{parts[0]} {parts[1]}"
            dt = pd.to_datetime(dt_str, errors="coerce")

            if pd.isna(dt):
                continue

            # Find elongation as the numeric token immediately before /L or /T
            elong: Optional[float] = None

            for i, token in enumerate(parts):
                if token in ("/L", "/T"):
                    if i == 0:
                        break
                    try:
                        elong = float(parts[i - 1])
                    except ValueError:
                        elong = None
                    break

            if elong is None:
                continue

            # Basic physical sanity check
            if not (0.0 <= elong <= 180.0):
                continue

            rows.append([dt, elong])

    if not rows:
        raise ValueError(
            f"No usable Horizons ephemeris rows with solar elongation were found in: {path}"
        )

    geom = pd.DataFrame(rows, columns=["datetime", "elongation_deg"])
    geom["day"] = geom["datetime"].dt.floor("D")

    geom_daily = (
        geom.groupby("day", as_index=False)["elongation_deg"]
        .median()
        .sort_values("day")
        .reset_index(drop=True)
    )

    if geom_daily.empty:
        raise ValueError(f"No daily SEP values could be derived from: {path}")

    # Final validation
    if geom_daily["elongation_deg"].isna().any():
        print("Warning: NaNs detected in loaded Horizons elongation data.")

    if not ((geom_daily["elongation_deg"] >= 0) & (geom_daily["elongation_deg"] <= 180)).all():
        print("Warning: Loaded elongation values fall outside the expected range 0–180 degrees.")

    return geom_daily


def load_dsn_data(
    filepath: str | Path,
    required_cols: Optional[Iterable[str]] = None,
    keep_optional_cols: Optional[Iterable[str]] = None,
    valid_only: bool = True,
    min_elev_deg: Optional[float] = 15.0,
    max_abs_doppler_hz: Optional[float] = 0.3,
) -> pd.DataFrame:
    """
    Load and clean a DSN Doppler data file.

    This function is based on the current 2010/2011 workflow:
    - reads whitespace-delimited text
    - requires UTC_time and doppler
    - optionally keeps valid, tropo, elev
    - applies valid / elevation / absolute Doppler filters

    Parameters
    ----------
    filepath : str or Path
        Path to DSN text file.
    required_cols : iterable[str], optional
        Required columns. Defaults to ['UTC_time', 'doppler'].
    keep_optional_cols : iterable[str], optional
        Optional columns to keep if present.
        Defaults to ['valid', 'tropo', 'elev'].
    valid_only : bool
        If True and 'valid' exists, keep only valid == 1 rows.
    min_elev_deg : float or None
        If not None and 'elev' exists, keep rows with elev > min_elev_deg.
    max_abs_doppler_hz : float or None
        If not None, keep rows with abs(doppler) < max_abs_doppler_hz.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe sorted by UTC_time.
    """
    path = _check_file_exists(filepath)

    if required_cols is None:
        required_cols = ["UTC_time", "doppler"]
    if keep_optional_cols is None:
        keep_optional_cols = ["valid", "tropo", "elev"]

    df = pd.read_csv(path, sep=r"\s+", header=0)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required DSN columns {missing} in file: {path}")

    keep_cols = list(required_cols)
    for col in keep_optional_cols:
        if col in df.columns and col not in keep_cols:
            keep_cols.append(col)

    df = df[keep_cols].copy()

    df["UTC_time"] = pd.to_datetime(df["UTC_time"], errors="coerce")
    df = df.dropna(subset=["UTC_time"]).copy()
    df = df.sort_values("UTC_time").reset_index(drop=True)

    if valid_only and "valid" in df.columns:
        df = df[df["valid"] == 1].copy()

    if min_elev_deg is not None and "elev" in df.columns:
        df = df[df["elev"] > float(min_elev_deg)].copy()

    if max_abs_doppler_hz is not None:
        df = df[np.abs(df["doppler"]) < float(max_abs_doppler_hz)].copy()

    if df.empty:
        raise ValueError(
            "No DSN samples remain after filtering. "
            f"File: {path}, min_elev_deg={min_elev_deg}, "
            f"max_abs_doppler_hz={max_abs_doppler_hz}"
        )

    return df.reset_index(drop=True)


def print_time_range_summary(
    name: str,
    df: pd.DataFrame,
    time_col: str,
) -> None:
    """
    Convenience helper for quick notebook summaries.
    """
    if df.empty:
        print(f"{name}: empty dataframe")
        return

    tmin = df[time_col].min()
    tmax = df[time_col].max()
    print(f"{name} rows: {len(df)}")
    print(f"{name} time range: {tmin} → {tmax}")