"""
plot_utils.py

Reusable plotting helpers for the DSN / VEX scintillation pipeline.

The goal is to keep notebook code short while preserving the figures
already used in the working analysis.
"""

from __future__ import annotations

from typing import Mapping, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# GLOBAL STYLE HELPERS
# ============================================================

def apply_time_axis_format(ax, month_interval: int = 2) -> None:
    """
    Apply a consistent date axis format for long time-series plots.
    """
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))


def finalize_figure(fig, tight: bool = True) -> None:
    """
    Apply consistent final layout.
    """
    if tight:
        fig.tight_layout()


# ============================================================
# NOTEBOOK 1 — DAILY RMS
# ============================================================

def plot_daily_rms_vs_sep(
    daily_df: pd.DataFrame,
    year: str,
    show_tropo: bool = True,
) -> tuple[plt.Figure, tuple]:
    """
    Plot daily Doppler RMS, smoothed RMS, DSN solar model, and SEP.
    """
    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    ax1.scatter(
        daily_df["decimal_year"],
        daily_df["doppler_rms_mm_s"],
        s=20,
        alpha=0.7,
        label="Daily RMS Doppler residuals"
    )

    if "doppler_smooth_mm_s" in daily_df.columns:
        ax1.plot(
            daily_df["decimal_year"],
            daily_df["doppler_smooth_mm_s"],
            linewidth=2,
            label="Smoothed Doppler noise"
        )

    if "solar_smooth_mm_s" in daily_df.columns:
        ax1.plot(
            daily_df["decimal_year"],
            daily_df["solar_smooth_mm_s"],
            linewidth=2,
            label="DSN solar scintillation model"
        )

    if show_tropo and "tropo_smooth" in daily_df.columns:
        ax1.plot(
            daily_df["decimal_year"],
            daily_df["tropo_smooth"],
            linewidth=1.5,
            linestyle="--",
            label="Measured tropo-derived rate RMS"
        )

    ax1.set_yscale("log")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Noise (mm/s)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    if "elongation_deg" in daily_df.columns:
        ax2.plot(
            daily_df["decimal_year"],
            daily_df["elongation_deg"],
            color="black",
            linewidth=1.5,
            label="SEP"
        )
    ax2.set_ylabel("SEP (deg)")
    ax2.set_ylim(0, 180)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title(f"VEX Daily Doppler Noise vs Solar Elongation ({year})")

    finalize_figure(fig)
    return fig, (ax1, ax2)


# ============================================================
# NOTEBOOK 2 — PHASE WINDOWS
# ============================================================

def plot_phase_scintillation_time_series(
    windows_df: pd.DataFrame,
    year: str,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot phase scintillation time series.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        windows_df["mid"],
        windows_df["phase_rms_rad"],
        linewidth=1.0
    )

    ax.set_yscale("log")
    ax.set_xlabel("UTC Time")
    ax.set_ylabel("Phase RMS (rad)")
    ax.set_title(f"Band-limited Phase Scintillation ({year})")
    ax.grid(True, alpha=0.3)

    apply_time_axis_format(ax)
    finalize_figure(fig)
    return fig, ax


# ============================================================
# NOTEBOOK 3 — BASELINE / DETECTION DIAGNOSTICS
# ============================================================

def plot_baseline_diagnostics(
    w: pd.DataFrame,
    binned: pd.DataFrame,
    year: str,
) -> tuple[plt.Figure, tuple]:
    """
    Plot the three baseline diagnostic panels:
    (a) phase vs elongation
    (b) normalized phase vs elongation
    (c) normalized phase vs time
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # panel 1
    axes[0].scatter(
        w["elongation_deg"],
        w["phase_rms_rad"],
        s=3,
        alpha=0.2,
        label="Data"
    )
    axes[0].plot(
        binned["elong_med"],
        binned["phase_med"],
        linewidth=2,
        label="Median baseline"
    )
    axes[0].plot(
        binned["elong_med"],
        binned["phase_baseline"],
        linewidth=2,
        linestyle="--",
        label="Hybrid quiet baseline"
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Elongation (deg)")
    axes[0].set_ylabel("Phase RMS (rad)")
    axes[0].set_title("Phase vs Elongation")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # panel 2
    axes[1].scatter(
        w["elongation_deg"],
        w["phase_ratio"],
        s=3,
        alpha=0.3
    )
    axes[1].axhline(1, color="red", linestyle="--")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Elongation (deg)")
    axes[1].set_ylabel("Normalised Phase (ratio)")
    axes[1].set_title("Elongation-corrected Signal")
    axes[1].grid(True, alpha=0.3)

    # panel 3
    axes[2].plot(
        w["mid"],
        w["phase_ratio"],
        linewidth=1
    )
    axes[2].axhline(1, color="red", linestyle="--")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("UTC Time")
    axes[2].set_ylabel("Normalised Phase (ratio)")
    axes[2].set_title("Elongation-corrected Phase")
    axes[2].grid(True, alpha=0.3)
    apply_time_axis_format(axes[2])

    fig.suptitle(f"Baseline construction and elongation correction ({year})", y=1.02)

    finalize_figure(fig)
    return fig, axes


def plot_cir_detection(
    w: pd.DataFrame,
    cir_df: pd.DataFrame,
    year: str,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot CIR detection result.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(w["mid"], w["phase_ratio"], alpha=0.2, label="Raw ratio")
    ax.plot(w["mid"], w["phase_smooth"], linewidth=2, label="Smoothed")

    if cir_df is not None and not cir_df.empty:
        for _, r in cir_df.iterrows():
            ax.axvspan(r["start"], r["end"], color="orange", alpha=0.3)

    ax.axhline(1, color="black", linestyle="--")
    ax.set_yscale("log")
    ax.set_xlabel("UTC Time")
    ax.set_ylabel("Normalised Phase")
    ax.set_title(f"CIR Detection ({year})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    apply_time_axis_format(ax)
    finalize_figure(fig)
    return fig, ax


def plot_transient_detection(
    w: pd.DataFrame,
    events_df: pd.DataFrame,
    year: str,
    threshold: float = 3.0,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot transient / CME-like detection result.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(w["mid"], w["clean_signal"], alpha=0.5, label="CIR-removed signal")

    if events_df is not None and not events_df.empty:
        for _, e in events_df.iterrows():
            ax.axvspan(e["start"], e["end"], color="red", alpha=0.3)

    ax.axhline(1, color="black", linestyle="--", label="Baseline")
    ax.axhline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold}")

    ax.set_yscale("log")
    ax.set_xlabel("UTC Time")
    ax.set_ylabel("CIR-removed signal")
    ax.set_title(f"Transient Detection ({year})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    apply_time_axis_format(ax)
    finalize_figure(fig)
    return fig, ax


def plot_pipeline_multi_panel(
    windows_df: pd.DataFrame,
    binned: pd.DataFrame,
    cir_df: pd.DataFrame,
    events_df: pd.DataFrame,
    year: str,
    transient_threshold: float = 3.0,
) -> tuple[plt.Figure, tuple]:
    """
    Publication-style 4-panel figure for the full detection pipeline.
    """
    plot_df = windows_df.copy().sort_values("mid")
    plot_ratio = plot_df.dropna(subset=["phase_ratio"]).copy()
    plot_clean = plot_df.dropna(subset=["clean_signal"]).copy()
    plot_base = plot_df.dropna(subset=["phase_rms_rad", "elongation_deg"]).copy()

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=False)
    ax1, ax2, ax3, ax4 = axes

    # Panel A
    ax1.plot(
        plot_df["mid"],
        plot_df["phase_rms_rad"],
        linewidth=0.8,
        alpha=0.5,
        label="Observed phase RMS"
    )

    if "phase_expected" in plot_df.columns:
        ax1.plot(
            plot_df["mid"],
            plot_df["phase_expected"],
            linewidth=2,
            label="Expected quiet level"
        )

    ax1.set_yscale("log")
    ax1.set_ylabel("Phase RMS (rad)")
    ax1.set_title(f"DSN phase-scintillation detection pipeline ({year})")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    ax1.text(0.01, 0.92, "(a)", transform=ax1.transAxes, fontsize=12, fontweight="bold")

    # Panel B
    ax2.scatter(
        plot_base["elongation_deg"],
        plot_base["phase_rms_rad"],
        s=4,
        alpha=0.12,
        label="Observed windows"
    )

    ax2.plot(
        binned["elong_med"],
        binned["phase_baseline"],
        linewidth=2.5,
        label="Quiet baseline"
    )

    if "phase_p25" in binned.columns:
        ax2.plot(
            binned["elong_med"],
            binned["phase_p25"],
            linestyle="--",
            linewidth=1.5,
            label="25th percentile"
        )

    if "phase_med" in binned.columns:
        ax2.plot(
            binned["elong_med"],
            binned["phase_med"],
            linestyle="--",
            linewidth=1.5,
            label="Median"
        )

    ax2.set_yscale("log")
    ax2.set_xlabel("Elongation (deg)")
    ax2.set_ylabel("Phase RMS (rad)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")
    ax2.text(0.01, 0.92, "(b)", transform=ax2.transAxes, fontsize=12, fontweight="bold")

    # Panel C
    ax3.plot(
        plot_ratio["mid"],
        plot_ratio["phase_ratio"],
        linewidth=0.9,
        label="Phase ratio"
    )

    if "phase_smooth" in plot_ratio.columns:
        ax3.plot(
            plot_ratio["mid"],
            plot_ratio["phase_smooth"],
            linewidth=2,
            label="CIR-scale background"
        )

    if cir_df is not None and not cir_df.empty:
        for _, r in cir_df.iterrows():
            ax3.axvspan(r["start"], r["end"], alpha=0.18)

    ax3.axhline(1.0, linestyle="--", linewidth=1.2, label="Quiet level")
    ax3.set_yscale("log")
    ax3.set_ylabel("Phase ratio")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right")
    ax3.text(0.01, 0.92, "(c)", transform=ax3.transAxes, fontsize=12, fontweight="bold")

    # Panel D
    ax4.plot(
        plot_clean["mid"],
        plot_clean["clean_signal"],
        linewidth=0.9,
        label="Clean signal"
    )

    if events_df is not None and not events_df.empty:
        for _, e in events_df.iterrows():
            ax4.axvspan(e["start"], e["end"], color="red", alpha=0.22)

    ax4.axhline(1.0, linestyle="--", linewidth=1.2, label="Baseline")
    ax4.axhline(transient_threshold, linestyle="--", linewidth=1.2, label="Transient threshold")

    ax4.set_yscale("log")
    ax4.set_xlabel("UTC time")
    ax4.set_ylabel("Clean signal")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper right")
    ax4.text(0.01, 0.92, "(d)", transform=ax4.transAxes, fontsize=12, fontweight="bold")

    for ax in [ax1, ax3, ax4]:
        apply_time_axis_format(ax)

    finalize_figure(fig)
    return fig, axes


# ============================================================
# NOTEBOOK 4 — MULTI-YEAR OVERVIEW
# ============================================================

def plot_multi_year_summary(
    summary_df: pd.DataFrame,
) -> tuple[plt.Figure, tuple]:
    """
    Plot 2x2 summary figure for multi-year comparison.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].bar(summary_df["year"], summary_df["n_cir"])
    axes[0, 0].set_title("Detected CIR regions per year")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].bar(summary_df["year"], summary_df["n_transient"])
    axes[0, 1].set_title("Detected transient events per year")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(summary_df["year"], summary_df["median_phase_rms"], marker="o")
    axes[1, 0].set_title("Median phase RMS per year")
    axes[1, 0].set_ylabel("Phase RMS (rad)")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(summary_df["year"], summary_df["max_clean_signal"], marker="o")
    axes[1, 1].set_title("Maximum clean signal per year")
    axes[1, 1].set_ylabel("Clean signal")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, alpha=0.3)

    finalize_figure(fig)
    return fig, axes


def plot_multi_year_overview(
    years: list[str],
    all_windows: Mapping[str, pd.DataFrame],
    all_cir: Mapping[str, pd.DataFrame],
    all_events: Mapping[str, pd.DataFrame],
) -> tuple[plt.Figure, list]:
    """
    Plot a stacked multi-year phase-ratio overview.
    """
    fig, axes = plt.subplots(len(years), 1, figsize=(14, 3 * len(years)), sharex=False)

    if len(years) == 1:
        axes = [axes]

    for ax, year in zip(axes, years):
        windows_df = all_windows.get(year)

        if windows_df is None or windows_df.empty:
            ax.text(0.5, 0.5, f"No data for {year}", ha="center", va="center")
            continue

        plot_df = windows_df.sort_values("mid").copy()

        ax.plot(
            plot_df["mid"],
            plot_df["phase_ratio"],
            linewidth=0.8,
            alpha=0.6,
            label="Phase ratio"
        )

        if "phase_smooth" in plot_df.columns:
            ax.plot(
                plot_df["mid"],
                plot_df["phase_smooth"],
                linewidth=1.5,
                label="CIR background"
            )

        cir_df = all_cir.get(year)
        if cir_df is not None and not cir_df.empty:
            for _, r in cir_df.iterrows():
                ax.axvspan(r["start"], r["end"], alpha=0.15)

        events_df = all_events.get(year)
        if events_df is not None and not events_df.empty:
            for _, e in events_df.iterrows():
                ax.axvspan(e["start"], e["end"], color="red", alpha=0.18)

        ax.axhline(1.0, linestyle="--", linewidth=1.0)
        ax.set_yscale("log")
        ax.set_ylabel(year)
        ax.grid(True, alpha=0.3)

        apply_time_axis_format(ax)

    axes[0].set_title("Multi-year DSN phase-ratio overview")
    axes[-1].set_xlabel("UTC time")

    finalize_figure(fig)
    return fig, axes