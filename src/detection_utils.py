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

def detect_transient_candidates(
    windows_df,
    threshold=3.0,
    min_consec_windows=2,
    min_duration_hr=0.33,
    max_duration_hr=12.0,
    merge_gap_hr=0.67,
):
    import numpy as np
    import pandas as pd

    w = windows_df.copy().sort_values("mid").reset_index(drop=True)

    w["candidate"] = w["clean_signal"] > threshold

    events=[]
    in_event=False

    for i in range(len(w)):

        if w.loc[i,"candidate"] and not in_event:
            start_idx=i
            in_event=True

        if in_event:

            is_last = i==(len(w)-1)
            next_off = (not is_last and not w.loc[i+1,"candidate"])

            if is_last or next_off:

                sub=w.loc[start_idx:i].copy()

                duration_hr=(
                    sub["end"].max()-sub["start"].min()
                ).total_seconds()/3600

                if (
                    len(sub)>=min_consec_windows
                    and duration_hr>=min_duration_hr
                    and duration_hr<=max_duration_hr
                ):

                    events.append({
                        "start":sub["start"].min(),
                        "end":sub["end"].max(),
                        "mid":sub["mid"].median(),
                        "duration_hr":duration_hr,
                        "n_windows":len(sub),
                        "peak_clean_signal":sub["clean_signal"].max(),
                        "median_clean_signal":sub["clean_signal"].median()
                    })

                in_event=False


    events_df=pd.DataFrame(events)

    if len(events_df)<=1:
        return events_df


    merged=[]
    current=events_df.iloc[0].copy()

    for i in range(1,len(events_df)):

        nxt=events_df.iloc[i]

        gap=(
            nxt["start"]-current["end"]
        ).total_seconds()/3600

        if gap<=merge_gap_hr:

            current["end"]=nxt["end"]

            current["duration_hr"]=(
                current["end"]-current["start"]
            ).total_seconds()/3600

            current["n_windows"]+=nxt["n_windows"]

            current["peak_clean_signal"]=max(
                current["peak_clean_signal"],
                nxt["peak_clean_signal"]
            )

        else:
            merged.append(current)
            current=nxt.copy()

    merged.append(current)

    out=pd.DataFrame(merged)

    out.insert(
        0,
        "event_id",
        np.arange(1,len(out)+1)
    )

    return out

def validate_event_contrast(
    windows_df,
    events_df,
    background_hr=12,
    high_conf_threshold=3.0
):

    import numpy as np
    import pandas as pd

    checks=[]

    for _,e in events_df.iterrows():

        evt=windows_df[
            (windows_df["mid"]>=e["start"]) &
            (windows_df["mid"]<=e["end"])
        ]

        local=windows_df[
            (windows_df["mid"]>=e["start"]-pd.Timedelta(hours=background_hr)) &
            (windows_df["mid"]<=e["end"]+pd.Timedelta(hours=background_hr))
        ]

        outside=local[
            (local["mid"]<e["start"]) |
            (local["mid"]>e["end"])
        ]

        ratio=(
            evt["clean_signal"].median() /
            outside["clean_signal"].median()
        )

        checks.append({
            "event_id":e["event_id"],
            "contrast_ratio":ratio
        })

    checks_df=pd.DataFrame(checks)

    out=events_df.merge(checks_df,on="event_id")

    out["candidate_class"]=np.where(
        out["contrast_ratio"]>high_conf_threshold,
        "high_confidence_cme_candidate",
        "lower_confidence_disturbance"
    )

    return out

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

@dataclass
class FinalCMEConfig:
    threshold: float = 3.0
    min_consec_windows: int = 2
    min_duration_hr: float = 0.33
    max_duration_hr: float = 12.0
    merge_gap_hr: float = 0.67
    local_background_hr: float = 12.0


def load_final_cme_input(input_file) -> pd.DataFrame:
    windows_df = pd.read_csv(input_file)

    for col in ["start", "end", "mid"]:
        windows_df[col] = pd.to_datetime(windows_df[col], errors="coerce")

    required = ["start", "end", "mid", "phase_rms_rad", "clean_signal"]
    missing = [c for c in required if c not in windows_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return windows_df.sort_values("mid").reset_index(drop=True)


def detect_final_cme_candidates(
    windows_df: pd.DataFrame,
    year: str,
    config: FinalCMEConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    w = windows_df.copy().sort_values("mid").reset_index(drop=True)
    w["candidate"] = w["clean_signal"] > config.threshold

    events = []
    group_id = (w["candidate"] != w["candidate"].shift()).cumsum()

    for _, sub in w[w["candidate"]].groupby(group_id):
        duration_hr = (sub["end"].max() - sub["start"].min()).total_seconds() / 3600

        if (
            len(sub) >= config.min_consec_windows
            and config.min_duration_hr <= duration_hr <= config.max_duration_hr
        ):
            row = {
                "start": sub["start"].min(),
                "end": sub["end"].max(),
                "mid": sub["mid"].median(),
                "duration_hr": duration_hr,
                "n_windows": len(sub),
                "peak_clean_signal": sub["clean_signal"].max(),
                "median_clean_signal": sub["clean_signal"].median(),
                "peak_phase_rms_rad": sub["phase_rms_rad"].max(),
                "median_phase_rms_rad": sub["phase_rms_rad"].median(),
            }

            for col in [
                "elongation_deg",
                "p_point_AU",
                "earth_sun_AU",
                "los_closest_from_earth_AU",
                "r_AU",
                "delta_AU",
                "hEcl_lon_deg",
                "hEcl_lat_deg",
            ]:
                if col in sub.columns:
                    row[f"{col}_median"] = sub[col].median()

            events.append(row)

    events_df = pd.DataFrame(events)

    if events_df.empty:
        final_events = events_df.copy()
    else:
        events_df = events_df.sort_values("start").reset_index(drop=True)

        merged = []
        current = events_df.iloc[0].copy()

        for i in range(1, len(events_df)):
            nxt = events_df.iloc[i]
            gap_hr = (nxt["start"] - current["end"]).total_seconds() / 3600

            if gap_hr <= config.merge_gap_hr:
                current["end"] = nxt["end"]
                current["mid"] = current["start"] + (current["end"] - current["start"]) / 2
                current["duration_hr"] = (current["end"] - current["start"]).total_seconds() / 3600
                current["n_windows"] += nxt["n_windows"]
                current["peak_clean_signal"] = max(current["peak_clean_signal"], nxt["peak_clean_signal"])
                current["median_clean_signal"] = np.nanmedian([current["median_clean_signal"], nxt["median_clean_signal"]])
                current["peak_phase_rms_rad"] = max(current["peak_phase_rms_rad"], nxt["peak_phase_rms_rad"])
                current["median_phase_rms_rad"] = np.nanmedian([current["median_phase_rms_rad"], nxt["median_phase_rms_rad"]])
            else:
                merged.append(current)
                current = nxt.copy()

        merged.append(current)
        final_events = pd.DataFrame(merged)

    final_events = final_events.reset_index(drop=True)
    final_events.insert(0, "event_id", np.arange(1, len(final_events) + 1))
    final_events.insert(1, "year", year)

    return w, final_events


def compute_final_cme_contrast(
    windows_df: pd.DataFrame,
    final_events: pd.DataFrame,
    local_background_hr: float = 12.0,
) -> pd.DataFrame:

    rows = []

    for _, e in final_events.iterrows():
        evt = windows_df[
            (windows_df["mid"] >= e["start"]) &
            (windows_df["mid"] <= e["end"])
        ]

        local = windows_df[
            (windows_df["mid"] >= e["start"] - pd.Timedelta(hours=local_background_hr)) &
            (windows_df["mid"] <= e["end"] + pd.Timedelta(hours=local_background_hr))
        ]

        outside = local[
            (local["mid"] < e["start"]) |
            (local["mid"] > e["end"])
        ]

        outside_med = outside["clean_signal"].median()

        rows.append({
            "event_id": e["event_id"],
            "inside_median": evt["clean_signal"].median(),
            "outside_median": outside_med,
            "contrast_ratio": evt["clean_signal"].median() / outside_med
            if pd.notna(outside_med) and outside_med != 0
            else np.nan,
        })

    return pd.DataFrame(rows)


def angular_separation_deg(a, b):
    """
    Smallest absolute angular separation between two angles in degrees.
    """
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def flag_cactus_crossings(
    events_df: pd.DataFrame,
    cactus_df: pd.DataFrame,
    p_col: str = "p_point_AU_median",
    los_angle_col: str = "los_pa_deg_median",
    cactus_time_col: str = "cme_launch_utc",
    cactus_speed_col: str = "cme_vel_kms",
    cactus_angle_col: str = "cme_pa_proxy_deg",
    cactus_width_col: str = "cme_half_width_deg",
    time_tolerance_hr: float = 12.0,
    radial_tolerance_AU: float = 0.08,
    angle_tolerance_deg: float = 15.0,
) -> pd.DataFrame:
    """
    Flag final DSN candidates if a CACTus CME plausibly crosses the
    Earth-VEX ray-path closest approach.

    A match requires:
    1. CME front reaches p_point_AU near the candidate time.
    2. CME angular direction is consistent with the LOS angular position.
    """

    out = events_df.copy()

    if out.empty:
        return out

    required_events = ["mid", p_col]
    missing_events = [c for c in required_events if c not in out.columns]
    if missing_events:
        raise ValueError(f"events_df missing required columns: {missing_events}")

    required_cactus = [
        cactus_time_col,
        cactus_speed_col,
        cactus_angle_col,
        cactus_width_col,
    ]
    missing_cactus = [c for c in required_cactus if c not in cactus_df.columns]
    if missing_cactus:
        raise ValueError(f"cactus_df missing required columns: {missing_cactus}")

    cat = cactus_df.copy()
    cat[cactus_time_col] = pd.to_datetime(cat[cactus_time_col], errors="coerce")
    cat = cat.dropna(subset=[cactus_time_col, cactus_speed_col])

    AU_KM = 149_597_870.7

    flags = []

    for _, e in out.iterrows():
        event_mid = pd.to_datetime(e["mid"])
        p_AU = e[p_col]

        best_match = None
        best_score = np.inf

        for _, cme in cat.iterrows():
            launch = cme[cactus_time_col]
            speed_kms = cme[cactus_speed_col]

            dt_hr = (event_mid - launch).total_seconds() / 3600.0

            if dt_hr <= 0:
                continue

            predicted_AU = speed_kms * (dt_hr * 3600.0) / AU_KM
            radial_error_AU = abs(predicted_AU - p_AU)

            if radial_error_AU > radial_tolerance_AU:
                continue

            # Direction check
            if los_angle_col in out.columns:
                los_angle = e[los_angle_col]
                cme_angle = cme[cactus_angle_col]
                cme_half_width = cme[cactus_width_col]

                angle_error = angular_separation_deg(los_angle, cme_angle)
                angle_allowed = cme_half_width + angle_tolerance_deg

                direction_ok = angle_error <= angle_allowed
            else:
                angle_error = np.nan
                angle_allowed = np.nan
                direction_ok = False

            if not direction_ok:
                continue

            score = radial_error_AU + angle_error / 360.0

            if score < best_score:
                best_score = score
                best_match = {
                    "cactus_crosses_p_point": True,
                    "matched_cactus_launch_utc": launch,
                    "matched_cactus_speed_kms": speed_kms,
                    "matched_cactus_pa_deg": cme[cactus_angle_col],
                    "matched_cactus_half_width_deg": cme[cactus_width_col],
                    "cactus_predicted_distance_AU": predicted_AU,
                    "cactus_radial_error_AU": radial_error_AU,
                    "cactus_angle_error_deg": angle_error,
                    "cactus_angle_allowed_deg": angle_allowed,
                }

        if best_match is None:
            best_match = {
                "cactus_crosses_p_point": False,
                "matched_cactus_launch_utc": pd.NaT,
                "matched_cactus_speed_kms": np.nan,
                "matched_cactus_pa_deg": np.nan,
                "matched_cactus_half_width_deg": np.nan,
                "cactus_predicted_distance_AU": np.nan,
                "cactus_radial_error_AU": np.nan,
                "cactus_angle_error_deg": np.nan,
                "cactus_angle_allowed_deg": np.nan,
            }

        flags.append(best_match)

    flags_df = pd.DataFrame(flags)

    return pd.concat([out.reset_index(drop=True), flags_df], axis=1)

def prepare_cactus_table(cactus_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise local CACTus CME catalogue columns.

    Expected raw columns:
    month, cme_id, t0, dt0_hr, pa_deg, da_deg, v_km_s, ...
    """
    out = cactus_df.copy()

    out = out.rename(columns={
        "t0": "cme_launch_utc",
        "v_km_s": "cme_vel_kms",
        "pa_deg": "cme_pa_deg",
        "da_deg": "cme_width_deg",
    })

    out["cme_launch_utc"] = pd.to_datetime(out["cme_launch_utc"], errors="coerce")
    out["cme_half_width_deg"] = out["cme_width_deg"] / 2.0

    return out


def angular_separation_deg(a, b):
    """
    Smallest angular separation between two angles in degrees.
    """
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def match_cactus_to_dsn_candidates(
    final_events: pd.DataFrame,
    cactus_df: pd.DataFrame,
    p_col: str = "p_point_AU_median",
    event_angle_col: str | None = None,
    radial_tolerance_AU: float = 0.20,
    angle_tolerance_deg: float = 15.0,
) -> pd.DataFrame:
    """
    Match DSN CME-like candidates against CACTus catalogue events.

    A candidate is CACTus-supported when:
    1. A CACTus CME front reaches the candidate P-point distance.
    2. If an event angle is available, the event angle overlaps the
       CACTus PA sector: pa_deg ± da_deg/2 plus tolerance.

    This is a catalogue-consistency flag, not a full 3D CME proof.
    """
    AU_KM = 149_597_870.7

    events = final_events.copy()
    cactus = prepare_cactus_table(cactus_df)

    if p_col not in events.columns:
        raise ValueError(f"final_events missing required column: {p_col}")

    required_cactus = [
        "cme_launch_utc",
        "cme_vel_kms",
        "cme_pa_deg",
        "cme_width_deg",
        "cme_half_width_deg",
    ]
    missing = [c for c in required_cactus if c not in cactus.columns]
    if missing:
        raise ValueError(f"cactus_df missing required columns: {missing}")

    use_angle = event_angle_col is not None and event_angle_col in events.columns

    matches = []

    for _, event in events.iterrows():
        event_mid = pd.to_datetime(event["mid"])
        p_AU = event[p_col]

        best_match = None
        best_score = np.inf

        for _, cme in cactus.iterrows():
            launch = cme["cme_launch_utc"]
            speed = cme["cme_vel_kms"]

            if pd.isna(launch) or pd.isna(speed):
                continue

            dt_hr = (event_mid - launch).total_seconds() / 3600.0

            if dt_hr <= 0:
                continue

            predicted_AU = speed * dt_hr * 3600.0 / AU_KM
            radial_error_AU = abs(predicted_AU - p_AU)

            if radial_error_AU > radial_tolerance_AU:
                continue

            if use_angle:
                event_angle = event[event_angle_col]
                cme_pa = cme["cme_pa_deg"]
                cme_half_width = cme["cme_half_width_deg"]

                angle_error = angular_separation_deg(event_angle, cme_pa)
                angle_allowed = cme_half_width + angle_tolerance_deg
                direction_ok = angle_error <= angle_allowed

                if not direction_ok:
                    continue
            else:
                event_angle = np.nan
                angle_error = np.nan
                angle_allowed = np.nan
                direction_ok = np.nan

            score = radial_error_AU
            if use_angle:
                score += angle_error / 360.0

            if score < best_score:
                best_score = score
                best_match = {
                    "cactus_crosses_p_point": True,
                    "cactus_direction_checked": use_angle,
                    "cactus_direction_consistent": direction_ok,
                    "matched_cactus_id": cme.get("cme_id", np.nan),
                    "matched_cactus_launch_utc": launch,
                    "matched_cactus_speed_kms": speed,
                    "matched_cactus_pa_deg": cme["cme_pa_deg"],
                    "matched_cactus_width_deg": cme["cme_width_deg"],
                    "matched_cactus_half_width_deg": cme["cme_half_width_deg"],
                    "cactus_predicted_distance_AU": predicted_AU,
                    "cactus_radial_error_AU": radial_error_AU,
                    "event_angle_deg": event_angle,
                    "cactus_angle_error_deg": angle_error,
                    "cactus_angle_allowed_deg": angle_allowed,
                }

        if best_match is None:
            best_match = {
                "cactus_crosses_p_point": False,
                "cactus_direction_checked": use_angle,
                "cactus_direction_consistent": False if use_angle else np.nan,
                "matched_cactus_id": np.nan,
                "matched_cactus_launch_utc": pd.NaT,
                "matched_cactus_speed_kms": np.nan,
                "matched_cactus_pa_deg": np.nan,
                "matched_cactus_width_deg": np.nan,
                "matched_cactus_half_width_deg": np.nan,
                "cactus_predicted_distance_AU": np.nan,
                "cactus_radial_error_AU": np.nan,
                "event_angle_deg": np.nan,
                "cactus_angle_error_deg": np.nan,
                "cactus_angle_allowed_deg": np.nan,
            }

        matches.append(best_match)

    matches_df = pd.DataFrame(matches)

    old_cols = [c for c in matches_df.columns if c in events.columns]
    events = events.drop(columns=old_cols)

    out = pd.concat(
        [events.reset_index(drop=True), matches_df.reset_index(drop=True)],
        axis=1,
    )

    out = out.loc[:, ~out.columns.duplicated(keep="last")].copy()

    return out

def add_event_median_columns(
    events_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Add event-level median values from window-level columns.
    """
    out = events_df.copy()

    for i, event in out.iterrows():
        sub = windows_df[
            (windows_df["mid"] >= event["start"]) &
            (windows_df["mid"] <= event["end"])
        ]

        for col in columns:
            if col in sub.columns:
                out.loc[i, f"{col}_median"] = sub[col].median()

    return out