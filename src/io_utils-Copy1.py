"""
io_utils.py

Input / output helpers for DSN Doppler and Horizons geometry files.

Designed around the current 2010/2011 workflow:
- DSN Doppler residuals in text files with columns such as UTC_time, doppler,
  and optional valid, elev, tropo.
- JPL Horizons-style geometry text files with $$SOE ... $$EOE blocks.

Author: project refactor for reusable multi-year workflow
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
    Load daily solar elongation (SEP) from a Horizons-style text file.

    Expected file structure:
    - Geometry block between $$SOE and $$EOE
    - Date and time near the start of each row
    - Elongation value immediately before token '/L' or '/T'

    Returns
    -------
    pd.DataFrame
        Columns:
        - day : datetime64[ns], floored to day
        - elongation_deg : float
    """
    path = _check_file_exists(filepath)

    rows: list[list[object]] = []
    read = False

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "$$SOE" in line:
                read = True
                continue
            if "$$EOE" in line:
                break
            if not read:
                continue

            parts = line.split()
            if len(parts) < 10:
                continue

            # Handle Horizons month markers seen in some exports
            if len(parts) > 2 and parts[2] in ("m", "a"):
                parts.pop(2)

            try:
                date_str = parts[0]
                time_str = parts[1]

                elong_deg: Optional[float] = None
                for i, token in enumerate(parts):
                    if token in ("/L", "/T"):
                        elong_deg = float(parts[i - 1])
                        break

                if elong_deg is None:
                    continue

                rows.append([f"{date_str} {time_str}", elong_deg])

            except Exception:
                # Skip malformed rows quietly; better than hard-failing on one bad line
                continue

    geom = pd.DataFrame(rows, columns=["datetime", "elongation_deg"])
    if geom.empty:
        raise ValueError(f"No usable Horizons rows found in: {path}")

    geom["datetime"] = pd.to_datetime(geom["datetime"], errors="coerce")
    geom = geom.dropna(subset=["datetime"]).copy()
    geom["day"] = geom["datetime"].dt.floor("D")

    geom_daily = (
        geom.groupby("day", as_index=False)["elongation_deg"]
        .median()
        .sort_values("day")
        .reset_index(drop=True)
    )

    if geom_daily.empty:
        raise ValueError(f"No daily SEP values could be derived from: {path}")

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