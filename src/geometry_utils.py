"""
geometry_utils.py

Reusable geometry helpers for DSN/VEX line-of-sight analysis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_horizons_los_geometry(filepath: str | Path) -> pd.DataFrame:
    """
    Load Horizons geometry needed for line-of-sight closest approach.

    Expected Horizons columns include:
    - heliocentric distance of VEX: r [AU]
    - observer-target distance: delta [AU]
    - solar elongation: S-O-T [deg]
    - heliocentric ecliptic longitude/latitude of VEX

    Returns daily geometry.
    """
    rows = []
    in_block = False

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "$$SOE" in line:
                in_block = True
                continue
            if "$$EOE" in line:
                break
            if not in_block:
                continue

            parts = line.split()
            if len(parts) < 18:
                continue

            try:
                dt = pd.to_datetime(f"{parts[0]} {parts[1]}", errors="coerce")
                if pd.isna(dt):
                    continue

                hEcl_lon_deg = float(parts[6])
                hEcl_lat_deg = float(parts[7])
                r_AU = float(parts[8])
                delta_AU = float(parts[10])

                elongation_deg = None
                for i, token in enumerate(parts):
                    if token in ("/L", "/T"):
                        elongation_deg = float(parts[i - 1])
                        break

                if elongation_deg is None:
                    continue

                rows.append({
                    "datetime": dt,
                    "hEcl_lon_deg": hEcl_lon_deg,
                    "hEcl_lat_deg": hEcl_lat_deg,
                    "r_AU": r_AU,
                    "delta_AU": delta_AU,
                    "elongation_deg": elongation_deg,
                })

            except Exception:
                continue

    geom = pd.DataFrame(rows)

    if geom.empty:
        raise ValueError(f"No usable Horizons geometry rows found in {filepath}")

    geom["day"] = geom["datetime"].dt.floor("D")

    geom_daily = (
        geom.groupby("day", as_index=False)
        .median(numeric_only=True)
        .sort_values("day")
        .reset_index(drop=True)
    )

    return geom_daily


def add_los_p_point_geometry(
    windows_df: pd.DataFrame,
    geom_daily: pd.DataFrame,
    mid_col: str = "mid",
    observer_sun_distance_AU: float = 1.0,
) -> pd.DataFrame:
    """
    Add line-of-sight closest-approach geometry.

    For an Earth/Venus radio link, the closest approach of the
    signal ray path to the Sun is approximately:

        p = R_Earth-Sun * sin(SEP)

    where SEP is the Sun-Earth-spacecraft elongation angle.
    """

    out = windows_df.copy()

    if "day" not in geom_daily.columns:
        raise ValueError("geom_daily must contain a 'day' column")

    if "elongation_deg" not in geom_daily.columns:
        raise ValueError("geom_daily must contain 'elongation_deg'")

    geom = geom_daily.set_index("day").sort_index()

    t_mid = pd.to_datetime(out[mid_col]).astype("int64")
    t_geom = geom.index.astype("int64")

    out["elongation_deg"] = np.interp(
        t_mid,
        t_geom,
        geom["elongation_deg"]
    )

    sep_rad = np.deg2rad(out["elongation_deg"].astype(float))

    out["earth_sun_AU"] = observer_sun_distance_AU
    out["p_point_AU"] = observer_sun_distance_AU * np.sin(sep_rad)
    out["los_closest_from_earth_AU"] = observer_sun_distance_AU * np.cos(sep_rad)

    return out