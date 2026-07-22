"""
Core logic for spectrophotometric flux calibration of MATISSE data.

The workflow is:
1. Discover and sort science / calibrator OIFITS files in the input directory.
2. Match each science file to the closest calibrator (same BCD, resolution,
   chopping state) within a user-defined time window.
3. Apply spectrophotometric calibration (total flux or correlated flux)
   using calibrator synthetic spectra databases.

Authors: aso (2026) adapted from ame legacy python script (mat_calFlux.py).
"""

from __future__ import annotations

import glob
import logging
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from matisse.core.flux.airmass import compute_airmass_correction
from matisse.core.flux.calibrator_spectrum import lookup_calibrator_spectrum
from matisse.core.flux.databases import get_cal_databases_dir
from matisse.core.flux.diagnostics import (
    plot_airmass_correction,
    plot_calibrated_flux,
    plot_calibrator_spectrum,
)
from matisse.core.flux.transfer_function import (
    calibrate_correlated_flux,
    calibrate_total_flux,
    resample_model_spectrum,
    write_provenance_headers,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class OifitsFileInfo:
    """Metadata extracted from a single OIFITS file header."""

    filename: str
    filepath: Path
    mjd_obs: float
    tpl_start: str
    bcd: str
    chop_status: bool
    resolution: str
    spectral_average: float


@dataclass
class FluxCalibrationConfig:
    """Configuration for the flux calibration run."""

    input_dir: Path
    output_dir: Path | None = None
    fig_dir: Path | None = None
    sci_name: str = ""
    cal_name: str = ""
    mode: str = "flux"  # "flux", "corrflux", or "both"
    band: str = "LM"  # "LM" or "N"
    timespan: float = 1.0  # maximum SCI-CAL time gap in hours
    do_airmass_correction: bool = False
    match_radius: float = 25.0  # arcsec for calibrator database matching
    spectral_features: bool = False  # annotate spectral features in diagnostics


def _extract_exposure_times(
    hdul: fits.HDUList,
) -> tuple[np.ndarray | None, float | None]:
    """Extract exposure times from an OIFITS HDUList.

    Priority:
    1. ``INT_TIME`` column from OI_FLUX / OI_VIS / OI_T3 tables.
    2. Fallback to primary-header detector exposure keywords.
    """

    int_time_flux: np.ndarray | None = None
    int_time_med: float | None = None
    for ext_name in ("OI_FLUX", "OI_VIS", "OI_T3"):
        try:
            table = hdul[ext_name].data
        except KeyError:
            continue
        if table is not None and "INT_TIME" in table.names:
            int_time_flux = np.asarray(table["INT_TIME"], dtype=float)
            break

    if int_time_flux is not None and int_time_flux.size > 0:
        if_one_time = np.median(int_time_flux) == np.min(int_time_flux)
        if if_one_time:
            int_time_med = float(int_time_flux[0])
        else:
            int_time_med = float(np.median(int_time_flux))

    hdr = hdul[0].header
    dit = hdr.get("HIERARCH ESO DET SEQ1 DIT", hdr.get("HIERARCH ESO DET DIT"))
    if dit is None:
        dit = hdr.get("EXPTIME")
    if dit is None:
        return None, int_time_med

    ndit = hdr.get("HIERARCH ESO DET NDIT", 1.0)
    return np.asarray([float(dit) * float(ndit)], dtype=float), int_time_med


# ---------------------------------------------------------------------------
# File discovery & sorting
# ---------------------------------------------------------------------------


def _dispersion_keyword(band: str) -> str:
    """Return the ESO header keyword suffix for spectral dispersion."""
    return "DIL" if band == "LM" else "DIN"


def _extract_file_info(filepath: Path, band: str) -> OifitsFileInfo:
    """Read header metadata from a single OIFITS file.

    Parameters
    ----------
    filepath : Path
        Full path to the FITS file.
    band : str
        Spectral band ('LM' or 'N').

    Returns
    -------
    OifitsFileInfo
        Dataclass with the relevant header metadata.
    """
    id_disp = _dispersion_keyword(band)
    with fits.open(filepath) as hdul:
        hdr = hdul[0].header
        bcd1 = hdr["HIERARCH ESO INS BCD1 ID"]
        bcd2 = hdr["HIERARCH ESO INS BCD2 ID"]
        spectralAverage = np.nan
        for i in range(1, 20):
            hdrkey = "HIERARCH ESO PRO REC1 PARAM" + "%d" % i + " NAME"
            if hdrkey in hdr:
                if np.logical_or(
                    "spectralBinning" in hdr[hdrkey], "spectralAverage" in hdr[hdrkey]
                ):
                    hdrkey2 = "HIERARCH ESO PRO REC1 PARAM" + "%d" % i + " VALUE"
                    spectralAverage = float(hdr[hdrkey2])
        return OifitsFileInfo(
            filename=filepath.name,
            filepath=filepath,
            mjd_obs=hdr["MJD-OBS"],
            tpl_start=hdr["HIERARCH ESO TPL START"],
            bcd=f"{bcd1}-{bcd2}",
            chop_status=hdr["HIERARCH ESO ISS CHOP ST"],
            resolution=hdr[f"HIERARCH ESO INS {id_disp} NAME"],
            spectral_average=spectralAverage,
        )


def discover_oifits_files(
    input_dir: Path,
    band: str,
    sci_name: str = "",
    cal_name: str = "",
) -> tuple[list[OifitsFileInfo], list[OifitsFileInfo]]:
    """Discover and sort science / calibrator OIFITS files.

    Files are matched by filename pattern and the ``HIERARCH ESO PRO CATG``
    header keyword when no explicit calibrator name is given.

    Parameters
    ----------
    input_dir : Path
        Directory containing OIFITS files.
    band : str
        Spectral band ('LM' or 'N').
    sci_name : str
        Science target name (substring in filename).
    cal_name : str
        Calibrator name (substring in filename). If empty, the closest
        calibrator in time (identified by ``CATG == CALIB_RAW_INT``) is used.

    Returns
    -------
    tuple[list[OifitsFileInfo], list[OifitsFileInfo]]
        Sorted lists of (science_files, calibrator_files).
    """
    band_tag = f"IR-{band}"
    abs_dir = str(input_dir.resolve()) + "/"

    sci_pattern = f"{abs_dir}*{sci_name}*_{band_tag}*hop*.fits"
    # In case of sci_name not specified, we want to consider all files as potential science targets
    sci_paths = []
    for fpath in sorted(glob.glob(sci_pattern)):
        with fits.open(fpath) as hdu:
            catg = hdu[0].header.get("HIERARCH ESO PRO CATG", "")
            if catg == "TARGET_RAW_INT" or catg == "TARGET_CAL_INT":
                sci_paths.append(Path(fpath))

    if cal_name:
        cal_pattern = f"{abs_dir}*{cal_name}*_{band_tag}*Chop*.fits"
        cal_paths = [Path(p) for p in sorted(glob.glob(cal_pattern))]
    else:
        logger.info(
            "No calibrator name specified – selecting closest calibrator in time."
        )
        all_pattern = f"{abs_dir}*_{band_tag}*hop*.fits"
        cal_paths = []
        for fpath in sorted(glob.glob(all_pattern)):
            with fits.open(fpath) as hdu:
                catg = hdu[0].header.get("HIERARCH ESO PRO CATG", "")
                if catg == "CALIB_RAW_INT":
                    cal_paths.append(Path(fpath))

    sci_infos = sorted(
        [_extract_file_info(p, band) for p in sci_paths],
        key=lambda x: x.mjd_obs,
    )
    cal_infos = sorted(
        [_extract_file_info(p, band) for p in cal_paths],
        key=lambda x: x.mjd_obs,
    )

    logger.info(
        "Found %d science and %d calibrator files.", len(sci_infos), len(cal_infos)
    )
    return sci_infos, cal_infos


# ---------------------------------------------------------------------------
# SCI ↔ CAL matching
# ---------------------------------------------------------------------------


def match_sci_cal(
    sci_files: list[OifitsFileInfo],
    cal_files: list[OifitsFileInfo],
    timespan_hours: float = 1.0,
) -> list[tuple[OifitsFileInfo, OifitsFileInfo | None]]:
    """Match each science file with the closest calibrator.

    Matching criteria (must all agree):
    - Same BCD configuration
    - Same spectral resolution
    - Same chopping status
    - Time difference < *timespan_hours*

    Parameters
    ----------
    sci_files : list[OifitsFileInfo]
        Sorted science files.
    cal_files : list[OifitsFileInfo]
        Sorted calibrator files.
    timespan_hours : float
        Maximum allowed time gap in hours.

    Returns
    -------
    list[tuple[OifitsFileInfo, OifitsFileInfo | None]]
        List of (science, best_calibrator) pairs. ``best_calibrator`` is
        ``None`` when no suitable match was found.
    """
    cal_mjds = np.array([c.mjd_obs for c in cal_files]) if cal_files else np.array([])
    cal_bcds = np.array([c.bcd for c in cal_files]) if cal_files else np.array([])
    cal_res = np.array([c.resolution for c in cal_files]) if cal_files else np.array([])
    cal_chop = (
        np.array([c.chop_status for c in cal_files]) if cal_files else np.array([])
    )

    pairs: list[tuple[OifitsFileInfo, OifitsFileInfo | None]] = []

    for sci in sci_files:
        if len(cal_files) == 0:
            logger.warning("No calibrator files available for %s", sci.filename)
            pairs.append((sci, None))
            continue

        # Filter by BCD, resolution, chopping
        mask = (
            (cal_bcds == sci.bcd)
            & (cal_res == sci.resolution)
            & (cal_chop == sci.chop_status)
        )
        candidates = np.where(mask)[0]

        if len(candidates) == 0:
            logger.warning(
                "No calibrator matching BCD/resolution/chop for %s", sci.filename
            )
            pairs.append((sci, None))
            continue

        delta_t = np.abs(sci.mjd_obs - cal_mjds[candidates]) * 24.0  # days → hours
        best_idx = candidates[np.argmin(delta_t)]
        best_dt = np.min(delta_t)

        if best_dt < timespan_hours:
            logger.info(
                "Matched %s ↔ %s (Δt=%.2f h)",
                sci.filename,
                cal_files[best_idx].filename,
                best_dt,
            )
            pairs.append((sci, cal_files[best_idx]))
        else:
            logger.warning(
                "Closest calibrator for %s exceeds timespan (%.2f h > %.2f h)",
                sci.filename,
                best_dt,
                timespan_hours,
            )
            pairs.append((sci, None))

    return pairs


# ---------------------------------------------------------------------------
# Calibration engine (placeholder – to be filled with ported logic)
# ---------------------------------------------------------------------------


def calibrate_flux(
    sci_path: Path,
    cal_path: Path,
    output_path: Path,
    bcd: str,
    cal_databases_dir: Path,
    *,
    mode: str = "flux",
    do_airmass_correction: bool = False,
    match_radius: float = 25.0,
    spectral_features: bool = False,
    fig_dir: Path | None = None,
    internal_cont: int = 0,
) -> int:
    """Apply spectrophotometric calibration to a single SCI/CAL pair.

    This is the modern equivalent of ``libFluxCal_STARSFLUX.fluxcal()``.

    Parameters
    ----------
    sci_path : Path
        Path to the science OIFITS file.
    cal_path : Path
        Path to the calibrator OIFITS file.
    output_path : Path
        Destination for the calibrated file.
    bcd : str
        BCD configuration (e.g. "IN-OUT") for provenance recording.
    cal_databases_dir : Path
        Directory containing calibrator synthetic spectra databases
        (``vBoekelDatabase.fits``, ``calib_spec_db_v10.fits``, etc.).
    mode : str
        ``'flux'`` (total flux), ``'corrflux'`` (correlated flux),
        or ``'both'``.
    do_airmass_correction : bool
        Apply differential airmass correction between SCI and CAL.
    match_radius : float
        Matching radius in arcsec for calibrator database cross-match.
    spectral_features : bool
        Annotate common spectral features when ``True``.
    fig_dir : Path | None
        Directory to save diagnostic plots. If ``None``, no plots are generated.

    Returns
    -------
    int
        Status code: 0 = success, >0 = error.
    """
    # 1. Copy science file → output
    shutil.copy2(sci_path, output_path)

    # 2. Open all three HDU lists
    try:
        hdul_sci = fits.open(sci_path)
    except FileNotFoundError:
        logger.error("Science file not found: %s", sci_path)
        output_path.unlink(missing_ok=True)
        return 3

    try:
        hdul_cal = fits.open(cal_path)
    except FileNotFoundError:
        logger.error("Calibrator file not found: %s", cal_path)
        hdul_sci.close()
        output_path.unlink(missing_ok=True)
        return 2

    hdul_out = fits.open(output_path, mode="update")

    # 3. Extract header metadata
    hdr_cal = hdul_cal[0].header
    cal_name = hdul_cal["OI_TARGET"].data["TARGET"][0]
    ra_cal = float(hdul_cal["OI_TARGET"].data["RAEP0"][0])
    dec_cal = float(hdul_cal["OI_TARGET"].data["DECEP0"][0])
    airmass_cal = (
        hdr_cal["HIERARCH ESO ISS AIRM START"] + hdr_cal["HIERARCH ESO ISS AIRM END"]
    ) / 2.0
    pwv_cal = (
        hdr_cal["HIERARCH ESO ISS AMBI IWV30D START"]
        + hdr_cal["HIERARCH ESO ISS AMBI IWV30D END"]
    ) / 2.0
    seeing_cal = (
        hdr_cal["HIERARCH ESO ISS AMBI FWHM START"]
        + hdr_cal["HIERARCH ESO ISS AMBI FWHM END"]
    ) / 2.0
    tau0_cal = (
        hdr_cal["HIERARCH ESO ISS AMBI TAU0 START"]
        + hdr_cal["HIERARCH ESO ISS AMBI TAU0 END"]
    ) / 2.0
    tpl_start_cal = hdr_cal["HIERARCH ESO TPL START"]
    band = hdr_cal["HIERARCH ESO DET CHIP TYPE"]  # 'IR-LM' or 'IR-N'
    spectral_average = np.nan
    for i in range(1, 20):
        hdrkey = "HIERARCH ESO PRO REC1 PARAM" + "%d" % i + " NAME"
        if hdrkey in hdr_cal:
            if np.logical_or(
                "spectralBinning" in hdr_cal[hdrkey],
                "spectralAverage" in hdr_cal[hdrkey],
            ):
                hdrkey2 = "HIERARCH ESO PRO REC1 PARAM" + "%d" % i + " VALUE"
                spectral_average = float(hdr_cal[hdrkey2])

    hdr_sci = hdul_sci[0].header
    airmass_sci = (
        hdr_sci["HIERARCH ESO ISS AIRM START"] + hdr_sci["HIERARCH ESO ISS AIRM END"]
    ) / 2.0
    pwv_sci = (
        hdr_sci["HIERARCH ESO ISS AMBI IWV30D START"]
        + hdr_sci["HIERARCH ESO ISS AMBI IWV30D END"]
    ) / 2.0
    detector = hdr_sci["HIERARCH ESO DET CHIP NAME"]
    exp_sci, int_time_med_sci = _extract_exposure_times(hdul_sci)
    exp_cal, int_time_med_cal = _extract_exposure_times(hdul_cal)

    if exp_sci is not None and exp_cal is not None:
        if not np.allclose(exp_sci, exp_cal, rtol=0.0, atol=1e-6):
            logger.warning(
                "SCI/CAL exposure times differ: SCI=%.2s, CAL=%.2fs",
                float(np.mean(exp_sci)),
                float(np.mean(exp_cal)),
            )
        else:
            logger.info(
                "SCI/CAL exposure times match (NDIT x DIT = %.1fs).",
                float(exp_sci[0]),
            )
            if (
                int_time_med_sci is not None
                and int_time_med_cal is not None
                and not np.isclose(
                    int_time_med_sci,
                    int_time_med_cal,
                    rtol=0.0,
                    atol=1e-3,
                )
            ):
                logger.warning(
                    f"OI_FLUX integration time differs between SCI and CAL: {int_time_med_sci:.1f}s vs. {int_time_med_cal:.1f}s for the CAL)"
                )

    else:
        logger.warning("Could not extract exposure times from SCI and/or CAL OIFITS.")

    logger.info(
        "SCI airmass=%.2f | CAL: %s (RA=%.4f Dec=%.4f) airmass=%.2f",
        airmass_sci,
        cal_name,
        ra_cal,
        dec_cal,
        airmass_cal,
    )

    # 4. Wavelength grids
    wav_cal = hdul_cal["OI_WAVELENGTH"].data["EFF_WAVE"]  # m
    wav_sci = hdul_sci["OI_WAVELENGTH"].data["EFF_WAVE"]  # m

    if np.min(wav_cal) < 0.0:
        logger.error("Invalid wavelength grid in calibrator file.")
        _close_all(hdul_sci, hdul_cal, hdul_out, output_path)
        return 5

    # 5. Look up calibrator model spectrum
    cal_db_paths = sorted(glob.glob(str(cal_databases_dir / "*.fits")))
    cal_spectrum = lookup_calibrator_spectrum(
        cal_name,
        ra_cal,
        dec_cal,
        cal_database_paths=[Path(p) for p in cal_db_paths],
        match_radius=match_radius,
        band=band,
    )

    if cal_spectrum is None:
        logger.error("Calibrator '%s' not found in any database.", cal_name)
        _close_all(hdul_sci, hdul_cal, hdul_out, output_path)
        return 1

    if math.isnan(cal_spectrum.diameter_mas):
        logger.error("Calibrator diameter not found for '%s'.", cal_name)
        if mode != "flux":
            _close_all(hdul_sci, hdul_cal, hdul_out, output_path)
            return 4

    logger.info(
        "Diameter = %.2f ± %.2f mas (%s, db=%s)",
        cal_spectrum.diameter_mas,
        cal_spectrum.diameter_err_mas,
        cal_spectrum.name,
        cal_spectrum.database,
    )

    # Ensure model wavelength and flux are sorted ascending
    wav_model = cal_spectrum.wavelength.copy()
    flux_model = cal_spectrum.flux.copy()
    if "calib_spec" in cal_spectrum.database:
        wav_model = np.flip(wav_model)
        flux_model = np.flip(flux_model)

    logger.debug(
        "Model wavelengths: %.2f to %.2f µm", wav_model[0] * 1e6, wav_model[-1] * 1e6
    )

    # Flip observation wavelengths for L-band processing
    wav_cal_proc = np.flip(wav_cal) if "L" in band else wav_cal
    wav_sci_proc = np.flip(wav_sci) if "L" in band else wav_sci

    logger.debug(
        "Observation wavelengths: %.2f to %.2f µm",
        wav_sci_proc[0] * 1e6,
        wav_sci_proc[-1] * 1e6,
    )

    # Determination of the overlap region
    overlap_mask = (wav_model > 0.99 * np.nanmin(wav_cal_proc)) & (
        wav_model < 1.01 * np.nanmax(wav_cal_proc)
    )
    wav_model = wav_model[overlap_mask]
    flux_model = flux_model[overlap_mask]

    # 6. Resample model spectrum to observation grid
    dwav_cal_proc = np.mean(np.gradient(wav_cal_proc))
    dwav_model = np.mean(np.gradient(wav_model))

    is_dense_model = dwav_cal_proc > dwav_model
    spectrum_resampled = resample_model_spectrum(
        wav_model, flux_model, wav_cal_proc, detector, spectral_average
    )
    logger.info("Model spectrum resampled to %d channels.", len(spectrum_resampled))

    data = np.column_stack((wav_cal_proc * 1e6, spectrum_resampled))
    outfile = os.path.join(fig_dir, "spec_resampled_new.txt")
    np.savetxt(outfile, data, "%.6f", header="wavelength flux")

    # Only generate the calibrator spectrum diagnostic for the first SCI-CAL pair to avoid redundancy
    if internal_cont == 0:
        # 6b. Diagnostic: calibrator model spectrum
        plot_calibrator_spectrum(
            fig_dir,
            cal_name=cal_name,
            band=band,
            wav_model=wav_model,
            flux_model=flux_model,
            wav_obs=wav_cal_proc,
            spectrum_resampled=spectrum_resampled,
            hdul_cal=hdul_cal,
            diameter_mas=cal_spectrum.diameter_mas,
            is_dense_model=is_dense_model,
        )

    # 7. Airmass correction
    if do_airmass_correction:
        output_dir = output_path.parent
        tag_sci = sci_path.stem
        tag_cal = cal_path.stem
        airmass_correction = compute_airmass_correction(
            hdul_sci=hdul_sci,
            hdul_cal=hdul_cal,
            wav_sci_m=wav_sci,
            wav_cal_m=wav_cal,
            airmass_sci=airmass_sci,
            airmass_cal=airmass_cal,
            pwv_sci=pwv_sci,
            pwv_cal=pwv_cal,
            output_dir=output_dir,
            tag_sci=tag_sci,
            tag_cal=tag_cal,
        )
    else:
        airmass_correction = np.ones_like(wav_sci_proc)

    # 7b. Diagnostic: airmass correction factor
    if do_airmass_correction:
        output_tag = f"{sci_path.stem}_cal_{cal_path.stem}"
        plot_airmass_correction(
            fig_dir,
            wav_sci_m=wav_sci_proc,
            airmass_correction=airmass_correction,
            output_tag=output_tag,
        )

    logger.debug(
        "Airmass correction applied: %s", "Yes" if do_airmass_correction else "No"
    )
    # 8. Apply calibration
    if mode in ("flux", "both"):
        calibrate_total_flux(
            hdul_sci,
            hdul_cal,
            hdul_out,
            spectrum_resampled,
            airmass_correction,
            band,
        )

    logger.debug("Total flux calibration applied: %s", mode in ("flux", "both"))
    if mode in ("corrflux", "both"):
        calibrate_correlated_flux(
            hdul_sci,
            hdul_cal,
            hdul_out,
            spectrum_resampled,
            airmass_correction,
            cal_spectrum.diameter_mas,
            cal_spectrum.diameter_err_mas,
            wav_cal_proc,
            band,
        )
    logger.debug(
        "Correlated flux calibration applied: %s", mode in ("corrflux", "both")
    )
    # 9. Write provenance headers
    write_provenance_headers(
        hdul_out,
        cal_name=cal_name,
        ra_cal=ra_cal,
        dec_cal=dec_cal,
        airmass_cal=airmass_cal,
        pwv_cal=pwv_cal,
        seeing_cal=seeing_cal,
        tau0_cal=tau0_cal,
        tpl_start_cal=tpl_start_cal,
        cal_spectrum=cal_spectrum,
    )

    logger.debug("Provenance headers written to output file.")
    # 10. Diagnostic: calibrated flux summary
    sci_name = hdul_sci["OI_TARGET"].data["TARGET"][0]

    plot_calibrated_flux(
        fig_dir,
        hdul_out=hdul_out,
        cal_name=cal_name,
        sci_name=sci_name,
        mode=mode,
        band=band,
        bcd=bcd,
        spectral_features=spectral_features,
    )

    # 11. Flush & close
    hdul_out.flush()
    hdul_out.close()
    hdul_cal.close()
    hdul_sci.close()
    logger.info("Calibrated file written: %s", output_path)
    return 0


def _close_all(
    hdul_sci: fits.HDUList,
    hdul_cal: fits.HDUList,
    hdul_out: fits.HDUList,
    output_path: Path,
) -> None:
    """Close all HDUs and remove partial output on error."""
    hdul_out.flush()
    hdul_out.close()
    hdul_cal.close()
    hdul_sci.close()
    output_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def run_flux_calibration(config: FluxCalibrationConfig) -> None:
    """Run the full flux calibration workflow.

    Parameters
    ----------
    config : FluxCalibrationConfig
        All parameters for the calibration run.
    """
    input_dir = config.input_dir.resolve()

    # Determine output directory
    if config.output_dir is None:
        suffix = "calflux" if config.mode == "flux" else "calcorrflux"
        output_dir = input_dir / suffix
    else:
        output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Locate calibrator spectra databases (Zenodo / local override / legacy bundle)
    cal_databases_dir = get_cal_databases_dir()
    logger.info("Using calibrator databases from: %s", cal_databases_dir)

    # Discover files
    sci_files, cal_files = discover_oifits_files(
        input_dir,
        band=config.band,
        sci_name=config.sci_name,
        cal_name=config.cal_name,
    )

    if not sci_files:
        logger.error("No science files found in %s for band %s", input_dir, config.band)
        return

    # Match SCI ↔ CAL
    pairs = match_sci_cal(sci_files, cal_files, timespan_hours=config.timespan)

    i_pair = 0  # Counter for internal diagnostics
    # Calibrate each pair
    for sci, cal in pairs:
        if cal is None:
            logger.warning("Skipping %s – no calibrator match.", sci.filename)
            continue

        suffix = "calflux" if config.mode == "flux" else "calcorrflux"
        output_name = sci.filename.replace(".fits", f"_{suffix}.fits")
        output_path = output_dir / output_name

        logger.info(
            "Calibrating %s with %s → %s", sci.filename, cal.filename, output_name
        )

        status = calibrate_flux(
            sci_path=sci.filepath,
            cal_path=cal.filepath,
            output_path=output_path,
            cal_databases_dir=cal_databases_dir,
            mode=config.mode,
            bcd=sci.bcd,
            do_airmass_correction=config.do_airmass_correction,
            match_radius=config.match_radius,
            fig_dir=config.fig_dir,
            spectral_features=config.spectral_features,
            internal_cont=i_pair,
        )

        if status == 0:
            logger.info("✓ %s calibrated successfully.", output_name)
        else:
            logger.error(
                "✗ Calibration failed for %s (status=%d).", sci.filename, status
            )
        i_pair += 1
