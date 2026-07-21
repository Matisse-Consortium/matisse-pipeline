from collections.abc import Sequence

import numpy as np
import pandas as pd

from .config import BCD_BASELINE_MAP, BCD_MODES_TO_CORRECT
from .io_utils import load_bcd_corrections

# Baselines unaffected by BCD swaps (same position in all modes)
_BCD_UNAFFECTED = {0, 1}

# Quality thresholds for σ_BCD / σ_ref ratio
_RATIO_GOOD = 1.0  # ≤ 1.0: green (noise-limited)
_RATIO_WARN = 2.0  # ≤ 2.0: yellow (acceptable)
# > 2.0: red (poor correction)

# Wavelength ranges (µm) for each sub-band
_SUBBAND_RANGES: dict[str, tuple[float, float]] = {
    "L": (3.2, 4.0),
    "M": (4.6, 4.8),
    "N": (8.0, 13.0),
}


def compute_correction_metrics(
    dict_corr,
    dict_raw,
    corrections_dir,
    sub_band: str | Sequence[float] | None = None,
):
    """Compute quality metrics for the BCD correction.

    For each baseline, computes (in-band only):
      - mean_std_uncorr : mean σ_BCD before correction
      - mean_std_corr   : mean σ_BCD after correction
      - improvement      : ratio σ_uncorr / σ_corr
      - mean_err_ref     : mean VIS2ERR of OUT_OUT
      - ratio_std_err    : σ_BCD(corr) / σ_OUT-OUT  (ideally ≤ 1)
      - mean_rel_resid   : mean |V2_corr - V2_ref| / V2_ref per BCD mode

    Parameters
    ----------
    dict_corr : dict
        Corrected data dictionary (from apply_bcd_corrections).
    dict_raw : dict
        Uncorrected (raw) data dictionary.
    corrections_dir : Path
        Directory containing the CSV correction files.
    sub_band : {"L", "M", "N"} or sequence of 2 floats or None, optional
        Restrict computation to a specific sub-band wavelength range. Pass two
        values ``[wl_low_um, wl_high_um]`` for an explicit wavelength range in microns.
        When *None* (default) the full range from the CSV files is used.

    Returns
    -------
    df_metrics : DataFrame
        Summary table with one row per baseline.
    """
    all_modes = ["OUT_OUT"] + BCD_MODES_TO_CORRECT
    wl = dict_corr["OUT_OUT"].wavelength * 1e6

    # Build band mask
    df0 = load_bcd_corrections(BCD_MODES_TO_CORRECT[0], corrections_dir)
    wl_ranges = df0[["wl_start_um", "wl_end_um"]].drop_duplicates()
    band_mask = np.zeros(len(wl), dtype=bool)
    for _, row in wl_ranges.iterrows():
        band_mask |= (wl >= row["wl_start_um"]) & (wl <= row["wl_end_um"])

    # Optionally restrict to a specific sub-band
    if sub_band is not None:
        if isinstance(sub_band, str):
            if sub_band not in _SUBBAND_RANGES:
                raise ValueError(
                    f"Unknown sub_band '{sub_band}'. Choose from {list(_SUBBAND_RANGES)}."
                )
            wl_lo, wl_hi = _SUBBAND_RANGES[sub_band]
        else:
            if len(sub_band) != 2:
                raise ValueError(
                    "sub_band range must contain exactly two values: [wl_low_um, wl_high_um]."
                )
            wl_lo, wl_hi = float(sub_band[0]), float(sub_band[1])

        if wl_hi <= wl_lo:
            raise ValueError(
                f"Invalid sub_band range [{wl_lo}, {wl_hi}] um: upper bound must be greater than lower bound."
            )
        band_mask &= (wl >= wl_lo) & (wl <= wl_hi)

    rows = []
    for i in range(6):
        blname = dict_corr["OUT_OUT"].blname[i]

        # STD across BCD modes (corrected)
        vis2_corr_stack = []
        for bcd_mode in all_modes:
            bcd_order = BCD_BASELINE_MAP[bcd_mode]
            vis2_corr_stack.append(dict_corr[bcd_mode].vis2["VIS2"][bcd_order][i])
        std_corr = np.std(np.array(vis2_corr_stack)[:, band_mask], axis=0)

        # STD across BCD modes (uncorrected)
        vis2_raw_stack = []
        for bcd_mode in all_modes:
            bcd_order = BCD_BASELINE_MAP[bcd_mode]
            vis2_raw_stack.append(dict_raw[bcd_mode].vis2["VIS2"][bcd_order][i])
        std_raw = np.std(np.array(vis2_raw_stack)[:, band_mask], axis=0)

        # OUT_OUT error
        vis2err_ref = dict_corr["OUT_OUT"].vis2["VIS2ERR"][i][band_mask]
        vis2_ref = dict_corr["OUT_OUT"].vis2["VIS2"][i][band_mask]

        # Mean relative residual per corrected BCD mode
        rel_resids = []
        for bcd_mode in BCD_MODES_TO_CORRECT:
            bcd_order = BCD_BASELINE_MAP[bcd_mode]
            vis2_c = dict_corr[bcd_mode].vis2["VIS2"][bcd_order][i][band_mask]
            rel_resids.append(np.nanmean(np.abs(vis2_c - vis2_ref) / np.abs(vis2_ref)))

        mean_std_raw = np.nanmean(std_raw)
        mean_std_corr = np.nanmean(std_corr)
        mean_err = np.nanmean(vis2err_ref)

        rows.append(
            {
                "baseline": blname,
                "affected": i not in _BCD_UNAFFECTED,
                "σ_BCD_uncorr": mean_std_raw,
                "σ_BCD_corr": mean_std_corr,
                "improvement": mean_std_raw / mean_std_corr
                if mean_std_corr > 0
                else np.inf,
                "σ_OUT-OUT": mean_err,
                "σ_BCD/σ_ref": mean_std_corr / mean_err if mean_err > 0 else np.inf,
                "mean_rel_resid": np.mean(rel_resids),
            }
        )

    return pd.DataFrame(rows)


def print_correction_metrics(df_metrics):
    """Print a modern, color-coded metrics table using rich."""
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    console = Console()

    table = Table(
        title="BCD Correction Quality Metrics (in-band average)",
        title_style="bold cyan",
        header_style="bold white on dark_blue",
        border_style="bright_blue",
        show_lines=True,
        padding=(0, 1),
    )

    table.add_column("Baseline", style="bold", justify="center")
    table.add_column("σ_BCD uncorr", justify="right")
    table.add_column("σ_BCD corr", justify="right")
    table.add_column("Improvement", justify="right")
    table.add_column("σ_OUT-OUT", justify="right")
    table.add_column("σ_BCD/σ_ref", justify="right")
    table.add_column("Rel. Resid.", justify="right")
    table.add_column("Status", justify="center")

    for _, row in df_metrics.iterrows():
        ratio = row["σ_BCD/σ_ref"]
        improvement = row["improvement"]
        affected = row["affected"]

        # Color for σ_BCD/σ_ref (always shown)
        if ratio <= _RATIO_GOOD:
            ratio_style = "bold green"
            status = Text("✓ GOOD", style="bold green")
        elif ratio <= _RATIO_WARN:
            ratio_style = "bold yellow"
            status = Text("~ FAIR", style="bold yellow")
        else:
            ratio_style = "bold red"
            status = Text("✗ POOR", style="bold red")

        # Improvement: only relevant for BCD-affected baselines
        if not affected:
            imp_cell = Text("— N/A", style="dim italic")
        else:
            if improvement > 1.5:
                imp_style = "green"
            elif improvement > 1.0:
                imp_style = "yellow"
            else:
                imp_style = "red"
            imp_cell = Text(f"{improvement:.2f}x", style=imp_style)

        table.add_row(
            row["baseline"],
            f"{row['σ_BCD_uncorr']:.4f}",
            f"{row['σ_BCD_corr']:.4f}",
            imp_cell,
            f"{row['σ_OUT-OUT']:.4f}",
            Text(f"{ratio:.2f}", style=ratio_style),
            f"{row['mean_rel_resid']:.4f}",
            status,
        )

    console.print()
    console.print(table)
    console.print("  [dim]improvement = σ_uncorr / σ_corr  (>1 = better)[/dim]")
    console.print(
        "  [dim]σ_BCD/σ_ref = σ_BCD(corr) / σ_OUT-OUT  "
        f"([green]≤{_RATIO_GOOD:.0f} good[/green], "
        f"[yellow]≤{_RATIO_WARN:.0f} fair[/yellow], "
        f"[red]>{_RATIO_WARN:.0f} poor[/red])[/dim]"
    )
    console.print()
