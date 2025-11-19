"""Visualization utilities for BCD corrections."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from matisse_pipeline.types import FloatArray

from .config import BCDConfig

logger = logging.getLogger(__name__)


def plot_corrections(
    wavelengths: FloatArray,
    corrections_mean: FloatArray,
    corrections_spectral: FloatArray,
    combined_spectral: list[FloatArray],
    poly_coef: FloatArray,
    config: BCDConfig,
    save_plots: bool = False,
) -> None:
    """
    Generate all diagnostic plots for BCD corrections.

    Parameters
    ----------
    wavelengths : np.ndarray
        Wavelength array in meters
    corrections_mean : np.ndarray
        Mean corrections per baseline (n_files, 6)
    corrections_spectral : np.ndarray
        Spectral corrections (n_files, 6, n_wavelengths)
    combined_spectral : list of np.ndarray
        Combined baseline corrections (3 elements)
    config : BCDConfig
        Configuration object
    save_plots : bool
        Whether to save plots to disk
    """
    logger.info("Generating diagnostic plots")

    # Convert wavelengths to microns for display
    wavs_um = wavelengths * 1e6

    # Plot 1: Mean corrections by baseline
    _plot_mean_corrections(corrections_mean, config)

    # Plot 2: Histograms of mean corrections
    _plot_histograms(corrections_mean, config)

    # Plot 3: Spectral corrections with uncertainties
    _plot_spectral_with_uncertainties(wavs_um, corrections_spectral, config)

    # Plot 4: Combined baselines with polynomial fits
    _plot_combined_with_fits(wavs_um, combined_spectral, poly_coef, config)

    if save_plots:
        _save_all_plots(config.output_dir)

    plt.show()


def _plot_mean_corrections(corrections_mean: FloatArray, config: BCDConfig) -> None:
    """Plot mean corrections per baseline."""
    plt.figure(figsize=(10, 6))

    # Plot all individual measurements
    plt.plot(corrections_mean.T, c="b", alpha=0.3)

    # Plot median
    median = np.nanmedian(corrections_mean, axis=0)
    plt.plot(median, c="k", linewidth=2, label="Median")

    plt.ylabel(
        f"Av OUT_OUT/{config.bcd_mode} {config.wavelength_low * 1e6:.1f}-{config.wavelength_high * 1e6:.1f} μm"
    )
    plt.xlabel("Baseline")
    plt.ylim(0.5, 1.8)
    plt.title("Mean BCD Corrections per Baseline")
    plt.legend()
    plt.grid(alpha=0.3)


def _plot_histograms(corrections_mean: FloatArray, config: BCDConfig) -> None:
    """Plot histograms of mean corrections for each baseline."""
    plt.figure(figsize=(12, 8))

    for i in range(6):
        plt.subplot(3, 2, i + 1)
        plt.hist(
            corrections_mean[:, i], bins=30, range=(0, 3), alpha=0.7, edgecolor="black"
        )
        plt.xlabel(f"Av OUT_OUT/{config.bcd_mode}")
        plt.ylabel("Count")
        plt.title(f"Baseline {i}")
        plt.grid(alpha=0.3)

    plt.tight_layout()


def _plot_spectral_corrections(
    wavs_um: FloatArray, corrections_spectral: FloatArray, config: BCDConfig
) -> None:
    """Plot spectral corrections for all files."""
    plt.figure(figsize=(12, 10))

    n_files, n_baselines, n_wav = corrections_spectral.shape

    for i in range(6):
        plt.subplot(3, 2, i + 1)

        # Plot all individual spectra
        plt.plot(wavs_um, corrections_spectral[:, i].T, c="b", alpha=0.1)

        # Plot median
        median = np.nanmedian(corrections_spectral[:, i], axis=0)
        plt.plot(wavs_um, median, c="k", linewidth=2)

        plt.ylim(0, 5)
        plt.ylabel(f"OUT_OUT/{config.bcd_mode}")
        plt.xlabel("Wavelength (μm)")
        plt.title(f"Baseline {i}")
        plt.grid(alpha=0.3)

    plt.tight_layout()


def _plot_spectral_with_uncertainties(
    wavs_um: FloatArray, corrections_spectral: FloatArray, config: BCDConfig
) -> None:
    """Plot spectral corrections with uncertainty bands using modern styling."""

    # Modern color palette
    COLORS = {
        "data": "#2E86AB",
        "uncertainty": "#2E86AB",
        "grid": "#E8E8E8",
        "background": "#FAFAFA",
    }

    # Modern font configuration
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Arial", "Helvetica Neue", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
        }
    )

    # Compact figure with tight vertical spacing
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(10, 6),
        facecolor="white",
        sharex=True,
        gridspec_kw={"hspace": 0.08, "wspace": 0.25},
    )
    fig.patch.set_facecolor("white")

    n_files = corrections_spectral.shape[0]

    for i in range(6):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        ax.set_facecolor(COLORS["background"])

        # Calculate percentiles (16%, 50%, 84% ~ ±1σ)
        percentiles = np.nanpercentile(corrections_spectral[:, i], [16, 50, 84], axis=0)

        median = percentiles[1]
        lower_err = (percentiles[0] - median) / np.sqrt(n_files)
        upper_err = (percentiles[2] - median) / np.sqrt(n_files)

        # Uncertainty band
        ax.fill_between(
            wavs_um,
            median + lower_err,
            median + upper_err,
            color=COLORS["uncertainty"],
            alpha=0.15,
            linewidth=0,
            label="±1 SE",
            zorder=2,
        )

        # Median line
        ax.plot(
            wavs_um,
            median,
            color=COLORS["data"],
            linewidth=2.2,
            label="Median",
            alpha=0.9,
            zorder=3,
        )

        ax.set_ylim(0, 2.5)
        ax.set_ylabel(r"$q$", fontsize=12, labelpad=8)

        # xlabel only for bottom row
        if row == 2:
            ax.set_xlabel("Wavelength (μm)", fontsize=10, labelpad=6)
        else:
            ax.tick_params(axis="x", labelbottom=False)

        # Baseline number as inset (top left)
        ax.text(
            0.05,
            0.95,
            f"BL {i}",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="600",
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor="#CCCCCC",
                alpha=0.9,
                linewidth=1.2,
            ),
            zorder=10,
        )

        ax.grid(True, color=COLORS["grid"], linewidth=0.7, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

        # Legend only on first plot
        if i == 0:
            legend = ax.legend(
                loc="upper right",
                frameon=True,
                fancybox=True,
                framealpha=0.95,
                edgecolor="lightgray",
                fontsize=8,
            )
            legend.get_frame().set_facecolor("white")

        # Spine styling
        for spine in ax.spines.values():
            spine.set_edgecolor("#CCCCCC")
            spine.set_linewidth(1.0)

    # Figure title
    bcd_formatted = config.bcd_mode.replace("_", "-")
    fig.suptitle(
        f"Spectral Corrections — OUT-OUT / {bcd_formatted}",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    # plt.tight_layout(pad=1)
    plt.subplots_adjust(
        left=0.06, right=0.99, top=0.95, bottom=0.08, hspace=0.08, wspace=0.2
    )
    # plt.subplots_adjust(top=0.97)


def _plot_combined_with_fits(
    wavs_um: FloatArray,
    combined_spectral: list[FloatArray],
    poly_coef: FloatArray,
    config: BCDConfig,
) -> None:
    """
    Plot combined baselines with polynomial fits using modern styling.

    Uses contemporary color schemes and enhanced visualization for polynomial fits.
    """
    # Modern color palette (inspired by Scientific Python 2025 trends)
    COLORS = {
        "data": "#2E86AB",  # Deep blue for data
        "fit": "#A23B72",  # Purple-magenta for fit
        "uncertainty": "#2E86AB",  # Matching blue for uncertainty band
        "fit_window": "#F18F01",  # Warm orange for fit windows
        "grid": "#E8E8E8",  # Soft gray for grid
        "background": "#FAFAFA",  # Very light gray background
    }

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Arial"
    plt.rcParams["mathtext.it"] = "Arial:italic"
    plt.rcParams["mathtext.bf"] = "Arial:bold"

    # Set modern style
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), facecolor="white", sharex=True)
    fig.patch.set_facecolor("white")

    # Wavelength windows for polynomial fits
    if config.band == "LM":  # and config.resolution == "LOW":
        wa = [3.2, 4.55]  # Start wavelengths
        wd = [3.8, 4.9]  # End wavelengths
        wtran_um = 4.35  # Transition point
        band_names = ["L", "M"]
    # elif config.band == "LM" and config.resolution == "MED":
    #     wa = [3.2]
    #     wd = [3.8]
    #     wtran_um = None
    #     band_names = ["L"]
    else:  # N band
        wa = [8.2]
        wd = [12.0]
        wtran_um = 10.0
        band_names = ["N"]

    # Plot each baseline
    for i in range(1, 3):
        ax = axes[i - 1]
        data = combined_spectral[i]
        median = np.nanmedian(data, axis=0)
        std = np.nanstd(data, axis=0)
        n_obs = data.shape[0]

        # Set background color
        ax.set_facecolor(COLORS["background"])

        # Plot fit windows first (as background reference)
        for w_start, w_end in zip(wa, wd, strict=True):
            idx_low = np.argmin(np.abs(wavs_um - w_start))
            idx_high = np.argmin(np.abs(wavs_um - w_end))
            wdown, wup = sorted([idx_low, idx_high])

            # Highlight fit window with subtle shading
            ax.axvspan(
                wavs_um[wdown],
                wavs_um[wup],
                color=COLORS["fit_window"],
                alpha=0.08,
                zorder=0,
            )

        # Uncertainty band (±1 standard error)
        ax.fill_between(
            wavs_um,
            median - std / np.sqrt(n_obs),
            median + std / np.sqrt(n_obs),
            color=COLORS["uncertainty"],
            alpha=0.15,
            linewidth=0,
            label=r"±1 $\sigma$",
            zorder=2,
        )

        ax.plot(
            wavs_um,
            median,
            color=COLORS["data"],
            linewidth=2.5,
            label="Median data",
            alpha=0.9,
            zorder=3,
        )

        # Compute and plot full fitted curve
        fitted_full = np.zeros_like(median)

        for j, (w_start, w_end) in enumerate(zip(wa, wd, strict=True)):
            idx_low = np.argmin(np.abs(wavs_um - w_start))
            idx_high = np.argmin(np.abs(wavs_um - w_end))
            wdown, wup = sorted([idx_low, idx_high])

            poly_coeffs = poly_coef[i - 1, j, :]
            fitted_window = np.polyval(poly_coeffs, wavs_um[wdown:wup])

            # Plot individual fit segments with subtle style
            ax.plot(
                wavs_um[wdown:wup],
                fitted_window,
                color=COLORS["fit"],
                linestyle="--",
                linewidth=1.8,
                alpha=0.6,
                zorder=1,
            )

            ax.set_ylim(0, 2.5)

            if i == 1:
                x_center = (wavs_um[wdown] + wavs_um[wup]) / 2
                y_pos = ax.get_ylim()[1] * 0.92  # 92% de la hauteur
                ax.text(
                    x_center,
                    y_pos,
                    f"{band_names[j]}-band",
                    fontsize=9,
                    color="#666666",
                    ha="center",
                    va="top",
                    bbox=dict(
                        boxstyle="round,pad=0.4",
                        facecolor="white",
                        edgecolor=COLORS["fit_window"],
                        alpha=0.2,
                        linewidth=1,
                    ),
                    zorder=10,
                )

            # Store for full reconstruction
            if len(wa) > 1:
                wtran_idx = np.argmin(np.abs(wavs_um - wtran_um))
                if j == 1:
                    fitted_full[:wtran_idx] = np.polyval(
                        poly_coeffs, wavs_um[:wtran_idx]
                    )
                else:
                    fitted_full[wtran_idx:] = np.polyval(
                        poly_coeffs, wavs_um[wtran_idx:]
                    )
            else:
                fitted_full = np.polyval(poly_coeffs, wavs_um)

        # Plot full composite fit with emphasis
        ax.plot(
            wavs_um,
            fitted_full,
            color=COLORS["fit"],
            linewidth=2.8,
            label="Polynomial fit",
            alpha=0.95,
            zorder=4,
        )

        # Add transition marker if multiple windows
        if len(wa) > 1:
            wtran_idx = np.argmin(np.abs(wavs_um - wtran_um))
            ax.axvline(
                wavs_um[wtran_idx],
                color="gray",
                linestyle=":",
                linewidth=1.2,
                alpha=0.5,
                zorder=1,
            )
            if i == 2:
                ax.text(
                    wavs_um[wtran_idx],
                    ax.get_ylim()[1] * 0.95,
                    f"  {wtran_um:.2f} μm",
                    fontsize=9,
                    color="gray",
                    verticalalignment="top",
                )

        frac_bcd = r"$_{\frac{\mathrm{OUT{-}OUT}}{\mathrm{IN{-}IN}}}$"
        ax.set_ylabel(
            r"Correction factor $\mathbf{q}$ " + frac_bcd,
            fontsize=13,
            fontweight="normal",
            fontfamily="sans-serif",
            labelpad=10,
        )

        if i != 1:
            ax.set_xlabel(
                r"Wavelength $\lambda$ (μm)",
                fontsize=12,
                fontweight="normal",
                fontfamily="sans-serif",
                labelpad=8,
            )
        else:
            ax.tick_params(axis="x", which="both", labelbottom=False)

        ax.set_title(
            f"Magic number for baseline pair {i}",
            fontsize=13,
            fontweight="600",
            fontfamily="sans-serif",
            color="gray",
            pad=12,
        )

        ax.grid(True, color=COLORS["grid"], linewidth=0.8, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

        # Legend with modern styling
        legend = ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=True,
            shadow=False,
            framealpha=0.95,
            edgecolor="lightgray",
            fontsize=9,
        )
        legend.get_frame().set_facecolor("white")

        # Spines styling
        for spine in ax.spines.values():
            spine.set_edgecolor("#CCCCCC")
            spine.set_linewidth(1.2)

    # Optional: add figure-level title
    fig.suptitle(
        f"BCD Magic Numbers - {config.bcd_mode.replace('_', '-')} Configuration",
        fontsize=13,
        fontweight="bold",
        color="#414147",
        y=0.98,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)


def _save_all_plots(output_dir: Path) -> None:
    """Save all currently open plots."""
    figs = [plt.figure(n) for n in plt.get_fignums()]

    year = str(output_dir).split("numbers")[1]
    for i, fig in enumerate(figs):
        filename = output_dir / f"bcd_diagnostic_{year}_{i + 1}.png"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {filename}")
