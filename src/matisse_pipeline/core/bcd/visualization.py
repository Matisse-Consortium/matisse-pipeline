"""Visualization utilities for BCD corrections."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    file_labels: list[str] | None = None,
    baseline_names: list[str] | None = None,
    target_names: list[str] | None = None,
    tau0_values: list[float] | None = None,
    bcd_mode: str | None = None,
) -> list[plt.Figure]:
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

    figures = [
        _plot_mean_corrections(
            corrections_mean,
            config,
            file_labels,
            baseline_names,
            target_names,
            tau0_values,
        ),
        _plot_histograms(corrections_mean, config),
        _plot_spectral_with_uncertainties(wavs_um, corrections_spectral, config),
        _plot_combined_with_fits(wavs_um, combined_spectral, poly_coef, config),
    ]

    if save_plots:
        _save_all_plots(config.output_dir, figures, bcd_mode)

    return figures


def _plot_mean_corrections(
    corrections_mean: FloatArray,
    config: BCDConfig,
    file_labels: list[str] | None = None,
    baseline_names: list[str] | None = None,
    target_names: list[str] | None = None,
    tau0_values: list[float] | None = None,
) -> plt.Figure:
    """Plot mean corrections per baseline with interactive file identification."""
    fig = plt.figure(figsize=(7, 4))
    ax = plt.gca()

    # Plot all individual measurements with labels
    lines = []
    for i in range(corrections_mean.shape[0]):
        label = file_labels[i] if file_labels else f"File {i + 1}"
        line = ax.plot(corrections_mean[i], c="#A2C8E8", alpha=0.3, picker=5)[0]
        line.set_label(label)
        lines.append(line)

    # Plot median
    spectral_mean = np.nanmean(corrections_mean, axis=0)
    spectral_std = np.nanstd(corrections_mean, axis=0)
    mean_line = ax.errorbar(
        range(len(spectral_mean)),
        spectral_mean,
        yerr=spectral_std,
        fmt="o",
        color="k",
        zorder=101,
    )

    plt.ylabel(
        f"Av OUT_OUT/{config.bcd_mode} {config.wavelength_low * 1e6:.1f}-{config.wavelength_high * 1e6:.1f} μm"
    )
    plt.xlabel("Baseline")
    plt.ylim(0.5, 1.8)
    plt.title("Mean BCD Corrections per Baseline (hover to identify)")

    # Create legend with proper handles
    if lines:
        plt.legend([lines[0], mean_line], ["Individual files", "Average"])

    plt.grid(alpha=0.3)

    # Add interactive tooltips if mplcursors is available
    try:
        import mplcursors

        cursor = mplcursors.cursor(lines, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            line_idx = lines.index(sel.artist)
            label = file_labels[line_idx] if file_labels else f"File {line_idx + 1}"
            baseline_idx = int(sel.target[0])
            baseline_name = (
                baseline_names[baseline_idx]
                if baseline_names and baseline_idx < len(baseline_names)
                else f"Baseline {baseline_idx}"
            )

            # Build tooltip text with quality indicator
            tooltip_lines = [label, baseline_name]

            if target_names and line_idx < len(target_names):
                target = target_names[line_idx]
                tooltip_lines.append(f"Target: {target}")

            if tau0_values and line_idx < len(tau0_values):
                tau0 = tau0_values[line_idx]
                # Quality indicator based on tau0
                if tau0 > 3.0:
                    quality = "[Good]"
                    edge_color = "#28a745"  # Green
                elif tau0 > 1.5:
                    quality = "[Medium]"
                    edge_color = "#ffc107"  # Yellow/Orange
                else:
                    quality = "[Poor]"
                    edge_color = "#dc3545"  # Red
                tooltip_lines.append(f"tau0: {tau0:.2f} ms {quality}")
            else:
                edge_color = "#666666"

            sel.annotation.set_text("\n".join(tooltip_lines))
            sel.annotation.get_bbox_patch().set(
                fc="white", alpha=0.95, edgecolor=edge_color, linewidth=2
            )
            sel.annotation.set_fontsize(9)
    except ImportError:
        logger.debug(
            "mplcursors not available - install with 'pip install mplcursors' for interactive tooltips"
        )

    plt.tight_layout()
    return fig


def _plot_histograms(corrections_mean: FloatArray, config: BCDConfig) -> plt.Figure:
    """Plot histograms of mean corrections for each baseline."""
    fig = plt.figure(figsize=(10, 6))

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
    return fig


def _plot_spectral_with_uncertainties(
    wavs_um: FloatArray, corrections_spectral: FloatArray, config: BCDConfig
) -> plt.Figure:
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
    return fig


def _plot_combined_with_fits(
    wavs_um: FloatArray,
    combined_spectral: list[FloatArray],
    poly_coef: FloatArray,
    config: BCDConfig,
) -> plt.Figure:
    """
    Plot combined baselines with polynomial fits using modern styling.

    Uses contemporary color schemes and enhanced visualization for polynomial fits.
    """
    # Modern color palette (inspired by Scientific Python 2025 trends)
    COLORS = {
        "data": "#2E86AB",  # Deep blue for data
        "fit": "#D289CB",  # Purple-magenta for fit
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
                color="k",
                linestyle="--",
                linewidth=1.8,
                alpha=0.6,
                zorder=5,
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
            linewidth=2,
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
    return fig


def _save_all_plots(
    output_dir: Path, figs: list[plt.Figure] | None = None, bcd_mode: str | None = None
) -> None:
    """Persist diagnostic figures to PNG files in diagnostic_plot subfolder."""
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]

    # Create diagnostic_plot subfolder
    diag_dir = Path(output_dir) / "diagnostic_plot"
    diag_dir.mkdir(parents=True, exist_ok=True)

    bcd_suffix = f"_{bcd_mode}" if bcd_mode else ""
    for i, fig in enumerate(figs):
        filename = diag_dir / f"bcd_diagnostic{bcd_suffix}_{i + 1}.png"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {filename}")


def plot_poly_corrections_results(
    output_dir: str | Path, bcd_mode: str = "IN_IN"
) -> plt.Figure:
    """
    Plot BCD corrections and overlays from polynomial fit.

    Parameters
    ----------
    output_dir : str | Path
        Path to the output directory containing correction CSV files.
    bcd_mode : str
        BCD mode (e.g., "IN_IN", "IN_OUT", "OUT_IN"). Used to construct CSV filenames.
    """
    output_dir = Path(output_dir)
    corrections_csv = output_dir / f"bcd_{bcd_mode}_spectral_corrections.csv"
    poly_csv = output_dir / f"bcd_{bcd_mode}_poly_coeffs.csv"

    df = pd.read_csv(corrections_csv)
    poly_df = pd.read_csv(poly_csv)

    # Extract baseline names (remove _std suffix)
    baseline_names = []
    for col in df.columns[1:]:  # Skip first column (index/wavelength)
        if not col.endswith("_std"):
            baseline_names.append(col)

    # Check polynomial order from CSV columns
    has_coef_x3 = "coef_x3" in poly_df.columns
    poly_order = 3 if has_coef_x3 else 2

    # Plot corrections for each baseline
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    x = df.iloc[:, 0] * 1e6  # Wavelength in micrometers

    for idx, baseline in enumerate(baseline_names):
        if idx < len(axes):
            y = df[baseline]
            y_std = df[f"{baseline}_std"]

            # Plot with fill_between for error
            axes[idx].plot(x, y, "-", linewidth=2, label="Correction", color="C0")
            axes[idx].fill_between(
                x, y - y_std, y + y_std, alpha=0.3, color="C0", label="±σ"
            )

            # Overlay polynomial fits from CSV mapping: direct on baseline_idx1
            sub_direct = poly_df[poly_df["baseline_idx1"] == idx]
            for _, row in sub_direct.iterrows():
                x_fit = np.linspace(row["wl_start_um"], row["wl_end_um"], 200)
                if poly_order == 3:
                    y_fit = (
                        row["coef_x3"] * x_fit**3
                        + row["coef_x2"] * x_fit**2
                        + row["coef_x1"] * x_fit
                        + row["coef_x0"]
                    )
                else:  # poly_order == 2
                    y_fit = (
                        row["coef_x2"] * x_fit**2
                        + row["coef_x1"] * x_fit
                        + row["coef_x0"]
                    )
                if row["window"] == poly_df["window"].min():
                    label = "Polynomial fit"
                else:
                    label = None
                axes[idx].plot(
                    x_fit,
                    y_fit,
                    "--",
                    color="C1",
                    linewidth=2,
                    zorder=10,
                    label=label,
                )

            # Inverse fits for rows where current baseline equals baseline_idx2
            sub_inverse = poly_df[poly_df["baseline_idx2"] == idx]
            for _, row in sub_inverse.iterrows():
                x_fit = np.linspace(row["wl_start_um"], row["wl_end_um"], 200)
                if poly_order == 3:
                    y_fit = (
                        row["coef_x3"] * x_fit**3
                        + row["coef_x2"] * x_fit**2
                        + row["coef_x1"] * x_fit
                        + row["coef_x0"]
                    )
                else:  # poly_order == 2
                    y_fit = (
                        row["coef_x2"] * x_fit**2
                        + row["coef_x1"] * x_fit
                        + row["coef_x0"]
                    )
                y_fit = np.where(np.abs(y_fit) > 1e-12, 1.0 / y_fit, np.nan)
                if row["window"] == 1:
                    label = "Polynomial fit (inv)"
                else:
                    label = None
                axes[idx].plot(
                    x_fit,
                    y_fit,
                    "--",
                    color="C3",
                    linewidth=2,
                    zorder=10,
                    label=label,
                )

            axes[idx].set_title(f"Baseline: {baseline}")
            axes[idx].set_xlabel("Wavelength (µm)")
            axes[idx].set_ylabel("Magic number BCD correction")
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend(fontsize=8)

    # Hide unused subplots
    for idx in range(len(baseline_names), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    return fig
