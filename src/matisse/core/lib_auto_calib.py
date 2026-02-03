"""
Core helpers for the MATISSE pipeline GUI series.

Created in 2016
Authors: pbe, fmillour, ame

Revised in 2026
Authors: aso

This module exposes the high-level utilities used to calibrate
the data by the calibrator sources (apply the transfer function).
"""

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from astropy.io import fits

from matisse.core.utils.log_utils import log


def generate_sof_files(
    input_dir: Path,
    output_dir: Path,
    band: str,
    timespan: float = 0.04,
) -> list[Path]:
    """Generate SOF files associating targets with calibrators.

    Parameters
    ----------
    input_dir : Path
        Directory containing FITS files.
    output_dir : Path
        Output directory for SOF files.
    band : str
        Spectral band flag ('-N' or '-LM').
    timespan : float, optional
        Time window in days to associate calibrations (default is 0.04).

    Returns
    -------
    list[Path]
        List of generated SOF file paths.
    """
    log.info(f"Scanning FITS files in {input_dir} for band {band}")

    targets = []
    calibs = []

    files = (f for f in input_dir.glob(f"*{band}*.fits") if "LAMP" not in f.name)
    # Single pass: read all headers
    for fits_file in files:
        with fits.open(fits_file) as hdul:
            hdr = hdul[0].header

        catg = hdr.get("ESO PRO CATG", "")

        if catg == "TARGET_RAW_INT":
            targets.append(
                {
                    "path": fits_file,
                    "mjd": hdr["MJD-OBS"],
                    "chip": hdr["ESO DET CHIP NAME"],
                    "dit": hdr["ESO DET SEQ1 DIT"],
                    "tplstart": hdr["ESO TPL START"],
                }
            )
        elif catg == "CALIB_RAW_INT":
            calibs.append(
                {
                    "path": fits_file,
                    "mjd": hdr["MJD-OBS"],
                    "chip": hdr["ESO DET CHIP NAME"],
                    "dit": hdr["ESO DET SEQ1 DIT"],
                }
            )

    log.info(f"Found {len(targets)} targets and {len(calibs)} calibrators.")
    # Group targets by TPL START
    tpl_groups = defaultdict(list)
    for target in targets:
        tpl_groups[target["tplstart"]].append(target)

    # Generate SOF files
    sof_files = []

    for _, target_list in tpl_groups.items():
        ref_target = target_list[0]

        # Build SOF filename
        tokens = ref_target["path"].stem.split("_")
        sof_name = f"{'_'.join(tokens[:5])}_cal_oifits.sof"
        sof_path = output_dir / sof_name

        # Find matching calibrators (vectorized)
        calib_mjds = np.array([c["mjd"] for c in calibs])
        time_diffs = np.abs(calib_mjds - ref_target["mjd"])

        matching_calibs = [
            calibs[i]
            for i in range(len(calibs))
            if (
                time_diffs[i] < timespan
                and calibs[i]["chip"] == ref_target["chip"]
                and calibs[i]["dit"] == ref_target["dit"]
            )
        ]

        # Write SOF file
        with open(sof_path, "w") as f:
            # Calculate relative path from output_dir to input_dir
            rel_input_dir = os.path.relpath(input_dir, output_dir)

            for target in target_list:
                rel_path = Path(rel_input_dir) / target["path"].name
                f.write(f"{rel_path}\tTARGET_RAW_INT\n")

            for calib in matching_calibs:
                rel_path = Path(rel_input_dir) / calib["path"].name
                f.write(f"{rel_path}\tCALIB_RAW_INT\n")

        sof_files.append(sof_path)

    log.info(f"Generated {len(sof_files)} SOF files for band {band}")
    return sof_files


def run_esorex_calibration(
    sof_path: Path,
    output_dir: Path,
    cumul_block: bool = True,
    custom_recipes_dir: Path | None = None,
) -> bool:
    """Run esorex mat_cal_oifits recipe on a SOF file.

    Parameters
    ----------
    sof_path : Path
        Path to SOF file.
    output_dir : Path
        Working directory for esorex.
    cumul_block : bool, optional
        Enable cumulBlock parameter (default is True).
    custom_recipes_dir : Path or None, optional
        Custom directory for MATISSE recipes.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    relative_sof = Path("..") / sof_path

    # Build esorex command
    cmd_parts = ["esorex"]

    if custom_recipes_dir is not None:
        cmd_parts.extend(["--recipe-dir", str(custom_recipes_dir)])

    cmd_parts.extend(
        [
            "mat_cal_oifits",
            f"--cumulBlock={'TRUE' if cumul_block else 'FALSE'}",
            str(relative_sof),
        ]
    )

    cmd = " ".join(cmd_parts)
    log_file = output_dir / "calibration.log"

    # Redirect output to log file
    cmd_with_log = f"cd {output_dir} && {cmd} >> {log_file.name} 2>&1"

    log.debug(f"Running: {cmd}")

    try:
        status = os.system(cmd_with_log)
        if status != 0:
            log.error(f"esorex failed for {sof_path.name} (exit code: {status})")
            # Show last lines of log file to help diagnose the issue
            if log_file.exists():
                with open(log_file) as f:
                    lines = f.readlines()
                    last_lines = "".join(lines[-10:])
                    log.error(f"Last lines of {log_file.name}:\n{last_lines}")
            return False
        return True
    except Exception as e:
        log.error(f"Unexpected error running esorex: {e}")
        return False


def rename_calibrated_outputs(output_dir: Path, base_name: str) -> None:
    """Rename calibrated FITS files according to MATISSE conventions.

    Parameters
    ----------
    output_dir : Path
        Directory containing output files.
    base_name : str
        Base name extracted from SOF file.
    """
    # Rename BCD mode files
    for fits_file in output_dir.glob("TARGET_CAL_INT_????.fits"):
        try:
            with fits.open(fits_file) as hdul:
                hdr = hdul[0].header

            bcd_mode = hdr["ESO CFG BCD MODE"]
            chop_status = hdr["ESO ISS CHOP ST"]
            suffix = "nochop" if chop_status == "F" else "chop"

            new_name = f"{base_name}_{bcd_mode}_{suffix}.fits"
            fits_file.rename(output_dir / new_name)
            log.debug(f"Renamed {fits_file.name} → {new_name}")

        except Exception as e:
            log.warning(f"Failed to rename {fits_file.name}: {e}")

    # Rename main output file
    nobcd_file = output_dir / "TARGET_CAL_INT_noBCD.fits"
    if nobcd_file.exists():
        nobcd_file.rename(output_dir / f"{base_name}.fits")
        log.debug(f"Renamed TARGET_CAL_INT_noBCD.fits → {base_name}.fits")


def cleanup_intermediate_files(output_dir: Path) -> None:
    """Remove intermediate calibration files matching patterns."""
    patterns = ["CALCPHASE*.fits", "CALDPHASE*.fits", "CALVIS*.fits"]

    removed_count = 0
    for pattern in patterns:
        for file_path in output_dir.glob(pattern):
            try:
                file_path.unlink()
                removed_count += 1
                log.debug(f"Removed: {file_path.name}")
            except OSError as e:
                log.warning(f"Failed to remove {file_path.name}: {e}")

    if removed_count > 0:
        log.info(f"Cleaned up {removed_count} intermediate file(s)")
