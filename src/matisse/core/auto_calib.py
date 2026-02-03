"""
Core helpers for the MATISSE pipeline GUI series.

Created in 2016
Authors: pbe, fmillour, ame

Revised in 2026
Contributor: aso

This module exposes the core function of the calibration step of the
MATISSE data reduction pipeline interface.
"""

from pathlib import Path

from tqdm import tqdm

from matisse.core.lib_auto_calib import (
    cleanup_intermediate_files,
    generate_sof_files,
    rename_calibrated_outputs,
    run_esorex_calibration,
)
from matisse.core.utils.log_utils import log


def run_calibration(
    input_dir: Path,
    output_dir: Path,
    bands: list[str],
    timespan: float = 0.04,
    cumul_block: bool = True,
    custom_recipes_dir: Path | None = None,
) -> None:
    """Run complete MATISSE calibration for specified bands.

    Parameters
    ----------
    input_dir : Path
        Directory containing raw FITS files.
    output_dir : Path
        Output directory for calibrated OIFITS.
    bands : list[str]
        List of spectral bands to process ('N', 'LM').
    timespan : float, optional
        Time window in days for calibrator matching (default is 0.04).
    cumul_block : bool, optional
        Enable cumulBlock parameter in esorex (default is True).
    custom_recipes_dir: Path | None = None, optional
        Custom directory for MATISSE recipes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for band in bands:
        band_flag = f"-{band}"
        log.info(f"Processing band {band}")

        # Generate SOF files
        sof_files = generate_sof_files(
            input_dir=input_dir,
            output_dir=output_dir,
            band=band_flag,
            timespan=timespan,
        )

        if not sof_files:
            log.warning(f"No SOF files generated for band {band}")
            continue

        # Process each SOF file
        for sof_file in tqdm(sof_files, desc=f"Calibrating {band}", unit="file"):
            # Run esorex mat_cal_oifits
            success = run_esorex_calibration(
                sof_path=sof_file,
                output_dir=output_dir,
                cumul_block=cumul_block,
                custom_recipes_dir=custom_recipes_dir,
            )

            if not success:
                log.error(f"Calibration failed for {sof_file.name}")
                continue

            # Extract base name and rename outputs
            tokens = sof_file.stem.split("_")
            base_name = "_".join(tokens[:5])

            rename_calibrated_outputs(
                output_dir=output_dir,
                base_name=base_name,
            )

        # Cleanup intermediate calibration files
        cleanup_intermediate_files(output_dir)

    log.info("Calibration pipeline completed")
