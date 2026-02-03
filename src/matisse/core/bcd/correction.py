"""BCD correction computation."""

import warnings
from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from astropy.io import fits
from numpy.typing import NDArray

from matisse.core.utils.log_utils import log
from matisse.core.utils.oifits_reader import OIFitsReader

from .config import BASELINE_PAIRS, BCD_BASELINE_MAP, BCDConfig
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

    # Fit magic number with polynomial laws
    assert wavelengths is not None
    wavs_um = wavelengths * 1e6  # convert to meter

    result_poly_coef = fit_magic_numbers(wavs_um, combined_spectral, config)

    # Compute medians
    median_corrections: list[FloatArray] = []
    std_corrections: list[FloatArray] = []
    for i in range(6):
        median_corr = np.nanmedian(corrections_spectral_arr[:, i], axis=0)
        std_corr = np.nanstd(corrections_spectral_arr[:, i], axis=0)
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

    # Only plot baselines 1 and 2 (combined pairs)
    result_poly_coef: list[FloatArray] = []
    for i in range(1, 3):
        data = combined_spectral[i]
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
    log.info("Not saving individual baseline .npy files; using CSV output only")
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
    log.info(f"Saved user-readable corrections to {csv_file}")

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
        windows = [(3.2, 4.0), (4.55, 4.9)]
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
    log.info(f"Saved polynomial coefficients to {output_path}")


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
        windows = [(3.2, 4.0), (4.55, 4.9)]
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

    log.info(f"Saved summary to {output_path}")
