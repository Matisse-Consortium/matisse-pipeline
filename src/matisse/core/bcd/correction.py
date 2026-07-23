"""BCD correction computation."""

import warnings
from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from astropy.io import fits
from numpy.typing import NDArray
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from matisse.core.bcd.io_utils import load_bcd_corrections
from matisse.core.bcd.merge import merge_by_tpl_start
from matisse.core.bcd.quality_metrics import (
    compute_correction_metrics,
    print_correction_metrics,
)
from matisse.core.bcd.visualization import plot_bcd_correction
from matisse.core.utils.log_utils import console, log
from matisse.core.utils.oifits_reader import OIFitsReader

from .config import BASELINE_PAIRS, BCD_BASELINE_MAP, BCD_MODES_TO_CORRECT, BCDConfig
from .outlier_filter import filter_outliers_custom

FloatArray = NDArray[np.floating[Any]]


class CorrectionOutputs(TypedDict):
    corrections: list[Path]
    csv: Path | None
    poly_csv: Path | None


def compute_bcd_corrections(
    folders: list[str],
    config: BCDConfig,
    chopping: bool = True,
    show_plots: bool = True,
    progress: Any | None = None,
    task_id: Any | None = None,
) -> dict[str, Any]:
    """
    Compute BCD corrections from file pairs.

    Parameters
    ----------
    folders : list of str
        Folders containing OIFITS files
    config : BCDConfig
        Configuration object
    chopping : bool
        Use chopping files
    show_plots : bool
        Generate diagnostic plots
    progress : Progress, optional
        Rich progress bar instance
    task_id : TaskID, optional
        Task ID for progress updates

    Returns
    -------
    dict
        Dictionary containing:
        - 'n_files': number of processed file pairs
        - 'correction_files': list of correction file paths
        - 'corrections': dict with correction data
    """
    # Find file pairs
    file_pairs = _find_bcd_file_pairs(
        folders=folders,
        bcd_mode=config.bcd_mode,
        band=config.band,
        resolution=config.resolution,
        chopping=chopping,
        tau0_min=config.tau0_min,
    )

    if not file_pairs:
        raise FileNotFoundError(
            f"No valid file pairs found for {config.bcd_mode} in {len(folders)} folders"
        )

    log.info(f"Processing {len(file_pairs)} file pairs")

    # Storage for corrections
    corrections_mean_list: list[FloatArray] = []  # VL in original
    corrections_spectral_list: list[FloatArray] = []  # VLwav in original
    file_labels: list[str] = []  # Track file identifiers
    target_names: list[str] = []  # Target names from OIFITS
    tau0_values: list[float] = []  # Coherence time values
    baseline_names: list[str] | None = None  # Baseline station names
    wavelengths: FloatArray | None = None

    # Process each file pair
    for idx, (out_out_path, bcd_path) in enumerate(file_pairs):
        if progress and task_id:
            progress.update(
                task_id,
                completed=idx,
                total=len(file_pairs),
                description=f"[cyan]Processing pair {idx + 1}/{len(file_pairs)}...",
            )

        try:
            # Read wavelengths and baseline names from first file
            if wavelengths is None:
                with fits.open(bcd_path) as hdul:
                    wavelengths = np.asarray(
                        deepcopy(hdul[3].data["eff_wave"]), dtype=float
                    )
                    log.info(f"Wavelength array shape: {wavelengths.shape}")

                # Extract baseline station names using OIFitsReader
                if baseline_names is None:
                    reader = OIFitsReader(bcd_path)
                    oifits_data = reader.read()
                    if oifits_data is not None:
                        if oifits_data.blname.size > 0:
                            baseline_names = [str(name) for name in oifits_data.blname]
                        else:
                            baseline_names = [f"Baseline {i}" for i in range(6)]
                        log.info(f"Baseline names: {baseline_names}")
                    else:
                        baseline_names = [f"Baseline {i}" for i in range(6)]

            with fits.open(bcd_path) as hdul:
                wavelengths_later = np.asarray(
                    deepcopy(hdul[3].data["eff_wave"]), dtype=float
                )
            if wavelengths_later.shape != wavelengths.shape:
                log.warning(
                    f"Wave shape seems to be different ({wavelengths_later.shape}/{wavelengths.shape}) for bcd {bcd_path}"
                )

            # Process this pair
            assert wavelengths is not None
            corr_mean, corr_spectral = _process_file_pair(
                out_out_path, bcd_path, config, wavelengths
            )

            if corr_mean is not None and corr_spectral is not None:
                corrections_mean_list.append(corr_mean)
                corrections_spectral_list.append(corr_spectral)
                # Extract meaningful identifier from filename (e.g., date-obs or file stem)
                file_labels.append(
                    bcd_path.stem.split("_")[0]
                )  # First part of filename

                # Extract target name and tau0 for visualization
                reader = OIFitsReader(bcd_path)
                oifits_data = reader.read()
                if oifits_data is not None:
                    target_names.append(oifits_data.target_name or "Unknown")
                    tau0_values.append(oifits_data.tau0 * 1e3)
                else:
                    target_names.append("Unknown")
                    tau0_values.append(0.0)

        except Exception as e:
            log.warning(f"Failed to process {bcd_path.name}: {e}")
        # continue

    if not corrections_mean_list:
        raise ValueError("No valid corrections computed")

    log.info(f"Successfully processed {len(corrections_mean_list)} file pairs")

    # Convert to arrays
    corrections_mean_arr = np.asarray(corrections_mean_list)  # (n_files, 6)
    corrections_spectral_arr = np.asarray(
        corrections_spectral_list
    )  # (n_files, 6, n_wav)

    # Combine baseline pairs
    combined_spectral = _combine_baseline_pairs(
        corrections_spectral_arr, config.bcd_mode
    )
    # NOTES: shape = (n_bl_pairs, n_files*2, n_wav)
    # combined_spectral shape should be (3, n_files*2, n_wav) for 3 baseline pairs
    # [1, 2] not affected, [3, 4, 5, 6] combined following BCD_BASELINE_MAP

    # Fit magic number with polynomial laws
    assert wavelengths is not None
    wavs_um = wavelengths * 1e6  # convert to meter

    result_poly_coef = fit_magic_numbers(wavs_um, combined_spectral, config)

    # Compute medians
    median_corrections: list[FloatArray] = []
    std_corrections: list[FloatArray] = []
    for i in range(6):
        if corrections_spectral_arr.shape[0] > 1:
            median_corr = np.nanmedian(corrections_spectral_arr[:, i], axis=0)
            std_corr = np.nanstd(corrections_spectral_arr[:, i], axis=0)
        else:
            median_corr = corrections_spectral_arr[0, i]
            std_corr = np.zeros_like(median_corr)
        median_corrections.append(np.asarray(median_corr, dtype=float))
        std_corrections.append(np.asarray(std_corr, dtype=float))

    # Build a user-friendly DataFrame (wl, per-baseline corrections and std)
    if baseline_names is None:
        baseline_names = [f"Baseline {i}" for i in range(6)]

    df_dict: dict[str, FloatArray] = {"wl": np.asarray(wavelengths, dtype=float)}
    for i in range(6):
        bl_name = str(baseline_names[i])
        df_dict[bl_name] = median_corrections[i]
        df_dict[f"{bl_name}_std"] = std_corrections[i]
    corrections_df = pd.DataFrame(df_dict)

    # Save corrections
    output_files = _save_corrections(
        wavelengths=wavelengths,
        corrections=median_corrections,
        stds=std_corrections,
        baseline_names=baseline_names,
        df=corrections_df,
        poly_coef=result_poly_coef,
        config=config,
        n_files=len(corrections_mean_arr),
    )

    # Generate plots if requested
    figures = None
    if show_plots:
        from .visualization import plot_corrections

        figures = plot_corrections(
            wavelengths=wavelengths,
            corrections_mean=corrections_mean_arr,
            corrections_spectral=corrections_spectral_arr,
            combined_spectral=combined_spectral,
            poly_coef=result_poly_coef,
            config=config,
            save_plots=True,
            file_labels=file_labels,
            baseline_names=baseline_names,
            target_names=target_names,
            tau0_values=tau0_values,
            bcd_mode=config.bcd_mode,
        )

    return {
        "n_files": len(corrections_mean_arr),
        "correction_files": output_files["corrections"],
        "csv_file": output_files.get("csv"),
        "poly_csv_file": output_files.get("poly_csv"),
        "corrections": {
            "mean": corrections_mean_arr,
            "mean_over_files": np.nanmean(corrections_mean_arr, axis=0),
            "std_over_files": np.nanstd(corrections_mean_arr, axis=0),
            "spectral": corrections_spectral_arr,
            "combined": combined_spectral,
            "wavelengths": wavelengths,
            "figures": figures,
            "dataframe": corrections_df,
            "poly_coef": result_poly_coef,
            "baseline_pairs": BASELINE_PAIRS.get(config.bcd_mode, []),
            "baseline_names": baseline_names,
        },
    }


def fit_magic_numbers(
    wavs_um: FloatArray, combined_spectral: list[FloatArray], config: BCDConfig
) -> FloatArray:
    """Fit polynomes on magic numbers for a given spectral range."""

    # Wavelength windows for polynomial fits
    if config.band == "LM":  #  & (config.resolution == "LOW"):
        wa = [3.2, 4.55]  # Start wavelengths
        wd = [3.8, 4.9]  # End wavelengths
    # elif (config.band == "LM") & (config.resolution == "MED"):
    #     wa = [3.2]  # Start wavelengths
    #     wd = [3.8]  # End wavelengths
    else:  # N band
        wa = [8.2]
        wd = [12.0]

    n_files = combined_spectral[0].shape[0] // 2
    log.info(f"Fitting polynomial to magic numbers for {n_files} combined spectra")

    # Only plot baselines 1 and 2 (combined pairs)
    result_poly_coef: list[FloatArray] = []
    for i in range(1, 3):
        data = combined_spectral[i]
        # Median over all stacked estimates of the pair (baseline a and the
        # inverse of its swap partner). Also correct for a single file: use the
        # median of the 2 rows, NOT data[0] alone, so the fitted polynomial
        # matches the "Median data" that is displayed.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(data, axis=0)

        coef_window: list[FloatArray] = []
        for w_start, w_end in zip(wa, wd, strict=True):
            # Find wavelength indices
            idx_low = np.argmin(np.abs(wavs_um - w_start))
            idx_high = np.argmin(np.abs(wavs_um - w_end))
            wdown, wup = sorted([idx_low, idx_high])

            # Fit polynomial
            poly_coeffs = np.polyfit(
                wavs_um[wdown:wup], median[wdown:wup], config.poly_order
            )

            # Evaluate polynomial
            coef_window.append(np.asarray(poly_coeffs, dtype=float))

        result_poly_coef.append(np.asarray(coef_window, dtype=float))

    return np.asarray(result_poly_coef, dtype=float)


def apply_bcd_corrections(
    data_dir: Path,
    corrections_dir: Path | None = None,
    chopping: bool = False,
    merge: bool = True,
    split_chopping: bool = True,
    plot: bool = True,
    verbose: bool = False,
    sub_band: str | Sequence[float] | None = None,
) -> None:
    """Apply BCD polynomial corrections for all BCD positions.

    For each BCD mode (IN_IN, IN_OUT, OUT_IN), load the data and correction
    coefficients, make a deep copy of the data, and apply the correction to the
    affected baselines in storage order (without reordering by BCD). The BCD
    reordering is only applied later for display or merge.

    For each pair of swapped baselines (baseline_idx1, baseline_idx2):
      - baseline_idx1 (in display order) → storage index bcd_order[idx1] is
        multiplied by poly_corr
      - baseline_idx2 (in display order) → storage index bcd_order[idx2] is
        divided by poly_corr

    Parameters
    ----------
    data_dir : Path
        Directory containing the OIFITS files.
    corrections_dir : Path
        Directory containing the CSV correction coefficient files.
    chopping : bool, optional
        Process chopped data. Default is False.
    merge : bool, optional
        Apply TPL start merging after correction. Default is True.
    plot : bool, optional
        Generate correction plots. Default is True.
    verbose : bool, optional
        Show detailed metrics tables for each file. Default is False.
    sub_band : {"L", "M", "N"} or sequence of 2 floats or None, optional
        Sub-band to use when computing quality metrics. Pass two values
        ``[wl_low_um, wl_high_um]`` for an explicit wavelength range in microns.
        Default is None (full band).
    """

    bases = _find_calibrator_filename_bases(data_dir, chopping=chopping)
    if not bases:
        raise ValueError(f"No calibrator files found in {data_dir}")

    # Log processing configuration
    console.rule("[bold cyan]BCD Correction Application[/bold cyan]")
    console.print(f"[cyan]Data directory:[/] {data_dir.resolve()}")
    if corrections_dir is not None:
        console.print(f"[cyan]Corrections directory:[/] {corrections_dir.resolve()}")
    else:
        console.print(
            "[cyan]Corrections directory:[/] [green]bundled master calibration[/green]"
        )
    console.print(f"[yellow]Chopping:[/] {'ON' if chopping else 'OFF'}")
    console.print(f"[yellow]Merge after correction:[/] {'ON' if merge else 'OFF'}")
    console.print(f"[dim]Verbose:[/] {'ON' if verbose else 'OFF'}")

    console.print(f"[magenta]Target files found:[/] {len(bases)} -> {', '.join(bases)}")
    console.print()

    dict_all = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Applying BCD corrections...", total=len(bases))
        for base in bases:
            progress.update(task, description=f"[cyan]Applying corrections: {base}...")
            try:
                dict_all[base] = _apply_bcd_corrections_single(
                    data_dir, corrections_dir, base
                )
            except FileNotFoundError as e:
                console.print(
                    f"[red]Error: Missing correction file for {base}[/red]\n"
                    f"  {e}\n"
                    f"[yellow]Skipping this file...please run 'matisse bcd compute' first to generate corrections.[/yellow]"
                )
            finally:
                progress.advance(task)

    chop_status = "noChop"
    if chopping:
        chop_status = "Chop"

    if not dict_all:
        raise FileNotFoundError(
            f"Correction files are missing for some calibrator in {corrections_dir}. "
            f"Please run 'matisse bcd compute' first to generate corrections."
        )

    # Collect summary metrics for all files
    summary_results = []

    for filename_base, dict_data in dict_all.items():
        # Load raw data for metrics comparison
        all_modes = ["OUT_OUT"] + BCD_MODES_TO_CORRECT
        dict_raw = {}
        for bcd_mode in all_modes:
            dict_raw[bcd_mode] = OIFitsReader(
                data_dir / f"{filename_base}_{bcd_mode}_{chop_status}.fits"
            ).read()

        # Save corrected OIFITS files
        out_dir = _save_corrected_oifits(dict_data, data_dir, filename_base)

        # Compute and print correction quality metrics
        df_metrics = compute_correction_metrics(
            dict_data, dict_raw, corrections_dir, sub_band=sub_band
        )

        if verbose:
            console.rule(f"[bold cyan]{filename_base}[/bold cyan]")
            print_correction_metrics(df_metrics)

        mean_ratio = np.mean(df_metrics["σ_BCD/ε_OUT-OUT"][2:])

        if verbose:
            log.info(
                f"Mean ratio of σ_BCD/ε_OUT-OUT (excluding unaffected baselines): {mean_ratio:.3f}"
            )

        is_recommended = mean_ratio <= 1.0
        quality_label = "GOOD" if is_recommended else "POOR"
        quality_color = "green" if is_recommended else "red"

        # Store summary info
        summary_results.append(
            {
                "filename": filename_base,
                "mean_ratio": mean_ratio,
                "quality": quality_label,
                "color": quality_color,
                "recommended": is_recommended,
            }
        )

        if verbose:
            recommendation_label = (
                "recommended" if is_recommended else "not recommended"
            )
            log.info(
                f"Overall correction quality: [{quality_color}]{quality_label}[/{quality_color}] "
                f"=> merge is [{quality_color}]{recommendation_label}[/{quality_color}]"
            )

        if plot:
            plot_bcd_correction(
                dict_data,
                corrections_dir=corrections_dir,
                title=r"$V^2$ Comparison after BCD correction",
            )

        if merge:
            merge_by_tpl_start(
                str(out_dir),
                save=True,
                output_dir=out_dir,
                separate_chopping=split_chopping,
            )

    # Display summary table for multiple files
    if len(summary_results) > 1 or not verbose:
        from rich.table import Table

        console.rule("[bold]BCD Correction Summary[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("File", style="cyan", no_wrap=False)
        table.add_column("Mean σ_BCD/ε_OUT-OUT", justify="right")
        table.add_column("Quality", justify="center")
        table.add_column("Merge", justify="center")

        for result in summary_results:
            quality_styled = (
                f"[{result['color']}]{result['quality']}[/{result['color']}]"
            )
            merge_styled = f"[{result['color']}]{'✓' if result['recommended'] else '✗'}[/{result['color']}]"
            table.add_row(
                result["filename"],
                f"{result['mean_ratio']:.3f}",
                quality_styled,
                merge_styled,
            )

        console.print(table)

        # Overall recommendation
        all_good = all(r["recommended"] for r in summary_results)
        if all_good:
            console.print(
                "[green]All corrections are GOOD quality. "
                "Suggestion: rerun with --merge enabled for these corrections.[/green]"
            )
        else:
            poor_count = sum(1 for r in summary_results if not r["recommended"])
            console.print(
                f"[yellow]Warning: {poor_count}/{len(summary_results)} "
                f"corrections show POOR quality. Review before merging.[/yellow]"
            )


def _save_corrected_oifits(dict_data, data_dir, filename_base, chopping=False):
    """Save BCD-corrected data to new OIFITS files.

    For each BCD position, re-opens the original FITS file, injects the
    corrected VIS2DATA column (without reordering baselines), adds history
    keywords to the primary header, and writes to a sibling directory
    named ``<parent>_bcd_corr/``.

    Output files are named ``<filename_base>_<BCD>_<CHOP>_bcd_corr.fits``.

    Parameters
    ----------
    dict_data : dict
        Dictionary {bcd_mode: data_object} returned by
        :func:`apply_bcd_corrections`.  The OUT_OUT entry is written
        unchanged (no correction applied).
    data_dir : Path
        Directory containing the original OIFITS files.
    filename_base : str
        Common filename prefix.

    Returns
    -------
    saved_files : dict
        Mapping {bcd_mode: Path} of the written files.
    """
    data_dir = Path(data_dir)
    out_dir = data_dir.parent / "bcd_corrected"
    out_dir.mkdir(parents=True, exist_ok=True)

    chop_status = "noChop"
    if chopping:
        chop_status = "Chop"

    all_modes = ["OUT_OUT"] + BCD_MODES_TO_CORRECT

    for bcd_mode in all_modes:
        orig_path = data_dir / f"{filename_base}_{bcd_mode}_{chop_status}.fits"
        out_path = out_dir / f"{filename_base}_{bcd_mode}_{chop_status}_bcd_corr.fits"

        with fits.open(orig_path) as hdul:
            # Inject corrected VIS2 (baselines kept in storage order)
            corrected_vis2 = dict_data[bcd_mode].vis2["VIS2"]
            hdul["OI_VIS2"].data["VIS2DATA"] = corrected_vis2

            # Mark the file as corrected in the primary header
            hdr = hdul[0].header
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            hdr["HIERARCH BCD_CORR"] = (
                True,
                "BCD polynomial correction applied",
            )
            hdr["HIERARCH BCD_CORR_DATE"] = (
                now,
                "UTC date of BCD correction",
            )
            hdr["HIERARCH BCD_CORR_MODE"] = (
                bcd_mode,
                "BCD position of this file",
            )
            hdr["HISTORY"] = (
                f"BCD polynomial correction applied on {now} (mode={bcd_mode})"
            )

            hdul.writeto(out_path, overwrite=True)

    return out_dir


def _apply_bcd_corrections_single(
    data_dir, corrections_dir, filename_base, chopping=False
):
    """Apply BCD polynomial corrections for a single file."""
    dict_data = {}

    chop_status = "noChop"
    if chopping:
        chop_status = "Chop"

    # Load OUT_OUT reference (no correction needed)
    data_outout = OIFitsReader(
        data_dir / f"{filename_base}_OUT_OUT_{chop_status}.fits"
    ).read()
    dict_data["OUT_OUT"] = data_outout

    for bcd_mode in BCD_MODES_TO_CORRECT:
        # Load data for this BCD position
        data_bcd = OIFitsReader(
            data_dir / f"{filename_base}_{bcd_mode}_{chop_status}.fits"
        ).read()

        # Deep copy to preserve the original
        data_bcd_corr = deepcopy(data_bcd)

        # Load correction coefficients
        df_coeffs = load_bcd_corrections(bcd_mode, corrections_dir)
        bcd_order = BCD_BASELINE_MAP[bcd_mode]
        wl = data_bcd.wavelength * 1e6

        # Apply corrections for each pair of affected baselines (j=0 and j=2)
        for j in [0, 2]:
            i_bl = df_coeffs["baseline_idx1"][j]
            i_bl_unv = df_coeffs["baseline_idx2"][j]

            poly_corr = _compute_poly_correction(wl, df_coeffs, j)

            # Convert display indices to storage indices
            storage_idx_bl = bcd_order[i_bl]
            storage_idx_unv = bcd_order[i_bl_unv]

            # Apply corrections in storage order
            data_bcd_corr.vis2["VIS2"][storage_idx_bl] *= poly_corr
            data_bcd_corr.vis2["VIS2"][storage_idx_unv] /= poly_corr

        dict_data[bcd_mode] = data_bcd_corr

    return dict_data


def _compute_poly_correction(
    wl: FloatArray,
    df_coeffs: pd.DataFrame,
    j: int,
) -> FloatArray:
    """
    Compute piecewise polynomial correction across wavelength windows.

    Evaluates polynomial coefficients (stored in descending power order)
    over two wavelength windows and concatenates results. Supports any
    polynomial order by dynamically detecting coefficient columns.

    Parameters
    ----------
    wl : ndarray
        Wavelength array in microns.
    df_coeffs : DataFrame
        Correction coefficients with columns coef_xN, ..., coef_x1, coef_x0
        (descending power order), wl_start_um, wl_end_um.
    j : int
        Row index in df_coeffs; row j+1 is used for the second window.

    Returns
    -------
    ndarray
        Polynomial correction concatenated across wavelength windows.

    Raises
    ------
    ValueError
        If no polynomial coefficient columns are found.
    """
    # Extract and sort polynomial coefficient columns in descending power order
    coef_cols = sorted(
        [col for col in df_coeffs.columns if col.startswith("coef_x")],
        key=lambda x: int(x.split("_x")[1]),
        reverse=True,
    )

    if not coef_cols:
        raise ValueError(
            "No polynomial coefficient columns found (expected coef_x0, coef_x1, ...)"
        )

    # Compute wavelength cut between windows
    wl_cut_corr = (df_coeffs["wl_end_um"].min() + df_coeffs["wl_start_um"].max()) / 2.0

    # First window (low wavelengths)
    coeffs_1 = np.array([df_coeffs[col][j] for col in coef_cols])
    wl_mask1 = (wl >= wl.min()) & (wl <= wl_cut_corr)
    poly_correction1 = np.polyval(coeffs_1, wl[wl_mask1])

    # Second window (high wavelengths)
    coeffs_2 = np.array([df_coeffs[col][j + 1] for col in coef_cols])
    wl_mask2 = (wl > wl_cut_corr) & (wl <= wl.max())
    poly_correction2 = np.polyval(coeffs_2, wl[wl_mask2])

    # Concatenate in wavelength order (high to low)
    return np.concatenate([poly_correction2, poly_correction1])


def _find_calibrator_filename_bases(data_dir, chopping=False):
    """Find filename bases in data_dir (CAL and SCI)."""
    data_dir = Path(data_dir)
    bases = []
    if chopping:
        suffix = "_OUT_OUT_Chop.fits"
    else:
        suffix = "_OUT_OUT_noChop.fits"

    for path in sorted(data_dir.glob(f"*{suffix}")):
        try:
            data = OIFitsReader(path).read()
        except Exception:
            continue

        if getattr(data, "band", "").upper() == "N":
            continue

        name = path.name
        if not name.endswith(suffix):
            continue
        base = name[: -len(suffix)]
        if _has_all_bcd_files(data_dir, base):
            bases.append(base)

    return bases


def _has_all_bcd_files(data_dir, filename_base):
    """Return True if all BCD mode files exist for a given base name."""
    data_dir = Path(data_dir)
    all_modes = ["OUT_OUT"] + BCD_MODES_TO_CORRECT
    for bcd_mode in all_modes:
        if not (data_dir / f"{filename_base}_{bcd_mode}_noChop.fits").exists():
            return False
    return True


def _combine_baseline_pairs(data: FloatArray, bcd_mode: str) -> list[FloatArray]:
    """
    Combine equivalent baseline pairs by inverting one.

    Parameters
    ----------
    data : np.ndarray
        Visibility data with shape (n_obs, 6, n_wavelengths)
    bcd_mode : str
        BCD configuration mode

    Returns
    -------
    list of np.ndarray
        List of 3 combined baseline arrays
    """
    if bcd_mode not in BASELINE_PAIRS:
        raise ValueError(f"No baseline pairs defined for {bcd_mode}")

    pairs = BASELINE_PAIRS[bcd_mode]
    log.info(f"Combining baseline pairs for {bcd_mode}: {pairs}")

    data[data == 0] = (
        np.nan
    )  # Replace zeros with NaN to avoid division issues during inversion
    # Combine baselines
    combined_raw = [
        # Baselines 0 and 1 (unchanged, just concatenated)
        np.append(data[:, 0], data[:, 1], axis=0),
        # First pair with inversion
        np.append(data[:, pairs[0][0]], data[:, pairs[0][1]] ** -1, axis=0),
        # Second pair with inversion
        np.append(data[:, pairs[1][0]], data[:, pairs[1][1]] ** -1, axis=0),
    ]
    combined: list[FloatArray] = [
        np.asarray(array, dtype=float) for array in combined_raw
    ]

    log.debug(f"Combined baselines: shapes {[c.shape for c in combined]}")
    return combined


def _find_bcd_file_pairs(
    folders: list[str],
    bcd_mode: str,
    band: str = "LM",
    resolution: str = "LOW",
    chopping: bool = True,
    tau0_min: float | None = None,
) -> list[tuple[Path, Path]]:
    """
    Find matching pairs of (OUT_OUT, BCD) files with optional tau0 filtering.

    Parameters
    ----------
    folders : list of str
        Folders to search in
    bcd_mode : str
        BCD configuration ('IN_IN', 'OUT_IN', 'IN_OUT')
    band : str
        Spectral band
    resolution : str
        Spectral resolution
    chopping : bool
        Search for chopping files
    tau0_min : float, optional
        Minimum coherence time in ms (None = no filter)

    Returns
    -------
    list of tuple
        List of (out_out_path, bcd_path) pairs
    """
    chop_suffix = "_Chop*" if chopping else "_noChop*"
    pattern = f"*{band}*{resolution}*{bcd_mode}*{chop_suffix}.fits"

    log.info(f"Searching for {pattern} in {len(folders)} folders")

    # Find BCD files
    bcd_files = []
    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            log.warning(f"Folder does not exist: {folder}")
            continue

        matches = list(folder_path.glob(pattern))
        bcd_files.extend(matches)

    log.info(f"Found {len(bcd_files)} {bcd_mode} files")

    # Create pairs with corresponding OUT_OUT files and apply tau0 filter
    pairs = []
    rejected_count = 0
    for bcd_file in sorted(bcd_files):
        out_out_file = Path(str(bcd_file).replace(bcd_mode, "OUT_OUT"))

        if not out_out_file.exists():
            log.debug(f"OUT_OUT file not found for {bcd_file.name}")
            continue

        # Apply tau0 filter if configured
        if tau0_min is not None:
            reader = OIFitsReader(bcd_file)
            oifits_data = reader.read()

            if oifits_data is not None:
                current_tau0 = oifits_data.tau0 * 1e3  # Convert to ms
                current_target = oifits_data.target_name or "Unknown"

                if current_tau0 < tau0_min:
                    log.warning(
                        f"Rejected {bcd_file.name}: tau0={current_tau0:.2f}ms < {tau0_min:.2f}ms (target: {current_target})"
                    )
                    rejected_count += 1
                    continue

        pairs.append((out_out_file, bcd_file))

    if tau0_min is not None:
        log.info(f"Rejected {rejected_count} files due to tau0 < {tau0_min:.2f}ms")
    log.info(f"Created {len(pairs)} valid (OUT_OUT, {bcd_mode}) file pairs")
    return pairs


def _process_file_pair(
    out_out_path: Path, bcd_path: Path, config: BCDConfig, wavelengths: FloatArray
) -> tuple[FloatArray | None, FloatArray | None]:
    """
    Process a single (OUT_OUT, BCD) file pair.

    Returns
    -------
    tuple
        (mean_correction, spectral_correction) or (None, None) if failed
    """

    with fits.open(out_out_path) as out_hdul, fits.open(bcd_path) as in_hdul:
        # Validate files
        if not _validate_file(in_hdul, config):
            return None, None

        n_telescopes = len(out_hdul["OI_ARRAY"].data["STA_INDEX"])
        n_baseline = n_telescopes * (n_telescopes - 1) // 2

        # Find wavelength window indices
        wlow_idx = np.argmin(np.abs(wavelengths - config.wavelength_low))
        whigh_idx = np.argmin(np.abs(wavelengths - config.wavelength_high))
        wdown, wup = sorted([wlow_idx, whigh_idx])
        wavelength_window = (int(wdown), int(wup))

        # Extract visibility data
        out_vis = np.asarray(
            deepcopy(out_hdul[config.extension].data[config.vis_column]), dtype=float
        )
        in_vis = np.asarray(
            deepcopy(in_hdul[config.extension].data[config.vis_column]), dtype=float
        )

        # Reshape to (n_exposures, 6, n_wavelengths)
        n_exp_out = out_vis.shape[0] // n_baseline
        n_exp_in = in_vis.shape[0] // n_baseline
        out_vis = out_vis.reshape((n_exp_out, n_baseline, -1))
        in_vis = in_vis.reshape((n_exp_in, n_baseline, -1))

        # Filter outliers
        thr_sigma = config.outlier_threshold

        mjd = out_hdul[0].header["DATE-OBS"]

        out_vis, n_out = filter_outliers_custom(out_vis, wavelength_window, thr_sigma)
        in_vis, n_bcd = filter_outliers_custom(in_vis, wavelength_window, thr_sigma)
        if n_out > 0:
            log.info(
                f"OUT-OUT/{config.bcd_mode} = {n_out}/{n_bcd} outliers for file {mjd}"
            )

        # Compute corrections
        # Mean correction (averaged over wavelength window and exposures)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            out_mean = np.nanmean(np.nanmean(out_vis[:, :, wdown:wup], axis=-1), axis=0)
            in_mean = np.nanmean(np.nanmean(in_vis[:, :, wdown:wup], axis=-1), axis=0)

        # Apply baseline remapping
        baseline_map = BCD_BASELINE_MAP[config.bcd_mode]
        in_mean_remapped = in_mean[baseline_map]

        correction_mean = out_mean / in_mean_remapped

        # Spectral correction (averaged over exposures only)
        out_spectral = np.nanmean(out_vis, axis=0)  # (6, n_wav)
        in_spectral = np.nanmean(in_vis, axis=0)  # (6, n_wav)
        in_spectral_remapped = in_spectral[baseline_map]

        # Count and handle near-zero values to avoid division by zero
        near_zero_threshold = 1e-10
        near_zero_mask = np.abs(in_spectral_remapped) < near_zero_threshold
        n_near_zero = np.count_nonzero(near_zero_mask)

        if n_near_zero > 0:
            log.warning(
                f"Found {n_near_zero} near-zero values in spectral data for {bcd_path.name} "
                f"({100 * n_near_zero / in_spectral_remapped.size:.1f}% of data)"
            )
            # Replace near-zero values with NaN to avoid division issues
            in_spectral_remapped = in_spectral_remapped.copy()
            in_spectral_remapped[near_zero_mask] = np.nan

        correction_spectral = out_spectral / in_spectral_remapped

        return (
            np.asarray(correction_mean, dtype=float),
            np.asarray(correction_spectral, dtype=float),
        )


def _validate_file(hdul: fits.HDUList, config: BCDConfig) -> bool:
    """Validate FITS file meets requirements."""
    # Check if calibrator
    obj = hdul[0].header.get("OBJECT")
    if obj not in ("STD", "LAMP"):
        log.debug("Not a standard calibrator or lamp (transfunc)")
        return False

    # Check wavelength array length
    wav_len = len(hdul[3].data["eff_wave"])
    if wav_len != config.spectral_binlen:
        log.debug(f"Spectral length {wav_len} != {config.spectral_binlen}")
        return False

    # Check amplitude type if needed
    if config.correlated_flux:
        amptyp = hdul[config.extension].header.get("AMPTYP")
        if amptyp != "correlated flux":
            log.debug("Not correlated flux")
            return False
    else:
        amptyp = hdul[config.extension].header.get("AMPTYP")
        if amptyp == "correlated flux":
            log.debug("Is correlated flux (not wanted)")
            return False

    return True


def _save_corrections(
    wavelengths: FloatArray,
    corrections: Sequence[FloatArray],
    stds: Sequence[FloatArray],
    baseline_names: Sequence[str],
    df: pd.DataFrame | None,
    poly_coef: FloatArray | None,
    config: BCDConfig,
    n_files: int | None = None,
) -> CorrectionOutputs:
    """Save correction files."""
    # Wavelengths are included in CSV; no separate .npy saved

    # Do not save individual baseline .npy files; rely on CSV
    correction_files: list[Path] = []

    # Save CSV with wl, per-baseline corrections and std columns
    if df is None:
        data: dict[str, FloatArray] = {"wl": np.asarray(wavelengths, dtype=float)}
        for i in range(6):
            bl_name = str(baseline_names[i])
            data[bl_name] = corrections[i]
            data[f"{bl_name}_std"] = stds[i]
        df = pd.DataFrame(data)

    csv_file = config.output_dir / f"bcd_{config.bcd_mode}_spectral_corrections.csv"
    df.to_csv(csv_file, index=False)
    log.info(f"Saved corrections to {str(csv_file.name)}")

    # Save polynomial coefficients in separate CSV
    poly_csv_file = None
    if poly_coef is not None:
        poly_csv_file = config.output_dir / f"bcd_{config.bcd_mode}_poly_coeffs.csv"
        _save_poly_coefficients(poly_coef, poly_csv_file, config, config.bcd_mode)

    # Save summary text file
    summary_file = config.output_dir / f"bcd_{config.bcd_mode}_summary.txt"
    _save_summary(
        summary_file,
        config,
        wavelengths,
        baseline_names,
        n_files,
        csv_file,
        poly_csv_file,
    )

    return {"corrections": correction_files, "csv": csv_file, "poly_csv": poly_csv_file}


def _save_poly_coefficients(
    poly_coef: FloatArray, output_path: Path, config: BCDConfig, bcd_mode: str
) -> None:
    """
    Save polynomial coefficients to CSV.

    Parameters
    ----------
    poly_coef : FloatArray
        Shape (2, n_windows, poly_order+1) - coefficients for 2 baseline pairs
    output_path : Path
        Output CSV file path
    config : BCDConfig
        Configuration object
    bcd_mode : str
        BCD mode to look up baseline pairs
    """
    # Determine wavelength windows based on band
    if config.band == "LM":
        windows = [
            (3.2, 3.8),
            (4.55, 4.9),
        ]  # match the fit range wd in fit_magic_numbers
    else:  # N band
        windows = [(8.2, 12.0)]

    # Get baseline pairs for this BCD mode
    baseline_pairs = BASELINE_PAIRS.get(bcd_mode, [])

    # Build rows for CSV
    rows = []
    for pair_idx in range(poly_coef.shape[0]):  # 2 baseline pairs
        for win_idx in range(poly_coef.shape[1]):  # n_windows
            # Get the actual baseline indices from BASELINE_PAIRS
            if pair_idx < len(baseline_pairs):
                bl_idx1, bl_idx2 = baseline_pairs[pair_idx]
            else:
                bl_idx1, bl_idx2 = None, None

            row = {
                "baseline_idx1": bl_idx1,
                "baseline_idx2": bl_idx2,
                "window": win_idx,
                "wl_start_um": windows[win_idx][0],
                "wl_end_um": windows[win_idx][1],
            }
            # Add polynomial coefficients (descending powers)
            coeffs = poly_coef[pair_idx, win_idx]
            for power_idx, coeff_val in enumerate(coeffs):
                power = len(coeffs) - 1 - power_idx  # descending power
                row[f"coef_x{power}"] = coeff_val
            rows.append(row)

    poly_df = pd.DataFrame(rows)
    poly_df.to_csv(output_path, index=False)
    log.info(f"Saved polynomial coefficients to {output_path.name}")


def _save_summary(
    output_path: Path,
    config: BCDConfig,
    wavelengths: FloatArray,
    baseline_names: Sequence[str],
    n_files: int | None,
    csv_file: Path,
    poly_csv_file: Path | None,
) -> None:
    """
    Save a human-readable summary of the BCD correction computation.

    Parameters
    ----------
    output_path : Path
        Output summary file path
    config : BCDConfig
        Configuration object
    wavelengths : FloatArray
        Wavelength array
    baseline_names : Sequence[str]
        Baseline names
    n_files : int, optional
        Number of source files processed
    csv_file : Path
        Path to corrections CSV
    poly_csv_file : Path, optional
        Path to polynomial coefficients CSV
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format wavelength range
    wl_min_um = wavelengths[0] * 1e6
    wl_max_um = wavelengths[-1] * 1e6

    # Get wavelength windows based on band
    if config.band == "LM":
        windows = [
            (3.2, 3.8),
            (4.55, 4.9),
        ]  # match the fit range wd in fit_magic_numbers
    else:  # N band
        windows = [(8.2, 12.0)]

    summary_lines = [
        "=" * 70,
        "BCD (Beam Commuting Device) CORRECTION SUMMARY",
        "=" * 70,
        "",
        f"Generated: {timestamp}",
        "",
        "PROCESSING PARAMETERS",
        "-" * 70,
        f"BCD Mode:              {config.bcd_mode}",
        f"Spectral Band:         {config.band}",
        f"Spectral Resolution:   {config.resolution}",
        f"Extension Type:        {config.extension}",
        f"Polynomial Order:      {config.poly_order}",
        "",
        "DATA PROCESSING",
        "-" * 70,
        f"Number of File Pairs:  {n_files if n_files is not None else 'N/A'}",
        f"Wavelength Range:      {wl_max_um:.2f} - {wl_min_um:.2f} µm",
        f"Number of Channels:    {len(wavelengths)}",
        "",
        "FILTERING & VALIDATION",
        "-" * 70,
        f"Outlier Threshold:     {config.outlier_threshold} σ",
        f"Correlated Flux Only:  {config.correlated_flux}",
        f"Min Coherence Time:    {config.tau0_min} ms"
        if config.tau0_min
        else "Min Coherence Time:    No filter",
        f"Wavelength Low:        {config.wavelength_low * 1e6:.2f} µm",
        f"Wavelength High:       {config.wavelength_high * 1e6:.2f} µm",
        "",
        "BASELINE PAIRS",
        "-" * 70,
    ]

    # Add baseline information
    baseline_pairs = BASELINE_PAIRS.get(config.bcd_mode, [])
    for pair_idx, (bl1, bl2) in enumerate(baseline_pairs, 1):
        bl1_name = (
            baseline_names[bl1] if bl1 < len(baseline_names) else f"Baseline {bl1}"
        )
        bl2_name = (
            baseline_names[bl2] if bl2 < len(baseline_names) else f"Baseline {bl2}"
        )
        summary_lines.append(
            f"Pair {pair_idx}:  [{bl1}] {bl1_name} ↔ [{bl2}] {bl2_name}"
        )

    summary_lines.extend(
        [
            "",
            "SPECTRAL WINDOWS (for polynomial fits)",
            "-" * 70,
        ]
    )

    for win_idx, (wl_start, wl_end) in enumerate(windows):
        summary_lines.append(f"Window {win_idx}: {wl_start:.2f} - {wl_end:.2f} µm")

    summary_lines.extend(
        [
            "",
            "OUTPUT FILES",
            "-" * 70,
            f"Corrections CSV:       {csv_file.name}",
        ]
    )

    if poly_csv_file:
        summary_lines.append(f"Polynomial Coeffs CSV: {poly_csv_file.name}")

    summary_lines.extend(
        [
            f"Output Directory:      {config.output_dir}",
            "",
            "BASELINES",
            "-" * 70,
        ]
    )

    for i, bl_name in enumerate(baseline_names):
        summary_lines.append(f"  [{i}] {bl_name}")

    summary_lines.extend(
        [
            "",
            "=" * 70,
            "For more details, see the CSV files above.",
            "=" * 70,
        ]
    )

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(summary_lines))

    log.info(f"Saved summary to {output_path.name}")
