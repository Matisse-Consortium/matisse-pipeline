"""
Airmass correction for MATISSE flux calibration.

Computes a differential atmospheric transmission correction between the
science target and the spectrophotometric calibrator using ESO's
**SkyCalc** command-line tool.

Workflow:
1. Generate SkyCalc input files for SCI and CAL (airmass, PWV, λ range).
2. Run ``skycalc_cli`` to obtain atmospheric transmission spectra.
3. Resample the transmission curves to the actual MATISSE spectral
   resolution (Gaussian convolution + spectral binning).
4. Compute the correction factor ``trans_cal / trans_sci``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
from astropy.convolution import Box1DKernel, Gaussian1DKernel, convolve
from astropy.io import fits
from numpy.polynomial.polynomial import polyval
from scipy.interpolate import interp1d

from matisse.core.flux.utils import (
    find_nearest_idx,
    get_dl_coeffs,
    get_dlambda,
    get_spectral_average,
)

logger = logging.getLogger(__name__)

# Allowed PWV values for SkyCalc (mm)
_PWV_ALLOWED = [0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 20.0, 30.0]


# ---------------------------------------------------------------------------
# SkyCalc input file generation
# ---------------------------------------------------------------------------


def create_skycalc_input(
    output_path: Path,
    airmass: float,
    pwv: float,
    wmin_nm: float,
    wmax_nm: float,
    *,
    wdelta: float = 0.1,
    wgrid_mode: str = "fixed_wavelength_step",
    wres: float = 20000.0,
    lsf_type: str = "none",
    lsf_gauss_fwhm: float = 5.0,
) -> None:
    """Write a SkyCalc CLI input parameter file.

    Parameters
    ----------
    output_path : Path
        Where to write the input file.
    airmass : float
        Airmass value.
    pwv : float
        Precipitable water vapour in mm (snapped to nearest allowed value).
    wmin_nm, wmax_nm : float
        Wavelength range in nanometres.
    wdelta : float
        Wavelength step in nm (for ``fixed_wavelength_step`` mode).
    wgrid_mode : str
        ``'fixed_wavelength_step'`` or ``'fixed_spectral_resolution'``.
    wres : float
        Spectral resolution (for ``fixed_spectral_resolution`` mode).
    lsf_type : str
        Line spread function type: ``'none'``, ``'Gaussian'``, ``'Boxcar'``.
    lsf_gauss_fwhm : float
        FWHM of the Gaussian LSF in pixels.
    """
    pwv_snap = _PWV_ALLOWED[find_nearest_idx(_PWV_ALLOWED, pwv)]

    content = (
        f"airmass         :  {airmass:f}\n"
        f"pwv_mode        :  pwv \n"
        f"season          :  0 \n"
        f"time            :  0 \n"
        f"pwv             :  {pwv_snap:f}\n"
        f"msolflux        :  130.0\n"
        f"incl_moon       :  Y\n"
        f"moon_sun_sep    :  90.0\n"
        f"moon_target_sep :  45.0\n"
        f"moon_alt        :  45.0\n"
        f"moon_earth_dist :  1.0\n"
        f"incl_starlight  :  Y\n"
        f"incl_zodiacal   :  Y\n"
        f"ecl_lon         :  135.0\n"
        f"ecl_lat         :  90.0\n"
        f"incl_loweratm   :  Y\n"
        f"incl_upperatm   :  Y\n"
        f"incl_airglow    :  Y\n"
        f"incl_therm      :  N\n"
        f"therm_t1        :  0.0\n"
        f"therm_e1        :  0.0\n"
        f"therm_t2        :  0.0\n"
        f"therm_e2        :  0.0\n"
        f"therm_t3        :  0.0\n"
        f"therm_e3        :  0.0\n"
        f"vacair          :  vac\n"
        f"wmin            :  {wmin_nm:f}\n"
        f"wmax            :  {wmax_nm:f}\n"
        f"wgrid_mode      :  {wgrid_mode}\n"
        f"wdelta          :  {wdelta:f}\n"
        f"wres            :  {wres:f}\n"
        f"lsf_type        :  {lsf_type}\n"
        f"lsf_gauss_fwhm  :  {lsf_gauss_fwhm:f}\n"
        f"lsf_boxcar_fwhm :  5.0\n"
        f"observatory     :  paranal"
    )

    output_path.write_text(content)
    logger.debug("SkyCalc input written to %s", output_path)


# ---------------------------------------------------------------------------
# SkyCalc execution
# ---------------------------------------------------------------------------


def _find_skycalc_cli() -> str | None:
    """Locate the ``skycalc_cli`` executable on the system."""
    path = shutil.which("skycalc_cli")
    if path is None:
        logger.warning("skycalc_cli not found on PATH.")
    return path


def run_skycalc(
    input_path: Path,
    output_path: Path,
) -> bool:
    """Run ``skycalc_cli`` and return True on success.

    Parameters
    ----------
    input_path : Path
        SkyCalc input parameter file.
    output_path : Path
        Where SkyCalc should write the output FITS.

    Returns
    -------
    bool
        ``True`` if the command succeeded.
    """
    cli = _find_skycalc_cli()
    if cli is None:
        return False

    cmd = [cli, "-i", str(input_path), "-o", str(output_path)]
    logger.info("Running SkyCalc: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as exc:
        logger.error("SkyCalc failed: %s", exc.stderr)
        return False


def read_skycalc_output(fpath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read wavelength (µm) and transmission from a SkyCalc output FITS.

    Parameters
    ----------
    fpath : Path
        Path to the SkyCalc output FITS file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (wavelength_um, transmission) arrays.
    """
    with fits.open(fpath) as hdul:
        wl_um = hdul[1].data["lam"] * 1e-3  # nm → µm
        trans = hdul[1].data["trans"]
    return np.asarray(wl_um), np.asarray(trans)


# ---------------------------------------------------------------------------
# Spectral resampling to real MATISSE resolution
# ---------------------------------------------------------------------------


def resample_to_matisse_resolution(
    wl_orig: np.ndarray,
    spec_orig: np.ndarray,
    dl_coeffs: list[float],
    wl_final: np.ndarray,
    spectral_binning: float,
    *,
    kernel_width_px: float = 10.0,
) -> np.ndarray:
    """Resample a spectrum to the actual MATISSE spectral resolution.

    Steps:
    1. Build a non-uniform wavelength grid with spacing ∝ Δλ(λ).
    2. Interpolate the input spectrum onto this grid.
    3. Convolve with a Gaussian kernel (simulates instrumental LSF).
    4. Interpolate back onto the final (data) wavelength grid.
    5. Convolve with a boxcar kernel (spectral binning).

    Parameters
    ----------
    wl_orig : np.ndarray
        Original wavelength grid (µm).
    spec_orig : np.ndarray
        Original spectrum values.
    dl_coeffs : list[float]
        Polynomial coefficients [c0, c1, c2, c3] for Δλ(λ) via ``polyval``.
    wl_final : np.ndarray
        Target wavelength grid (µm) — the MATISSE data grid.
    spectral_binning : float
        Spectral binning factor from the reduction recipe.
    kernel_width_px : float
        Gaussian kernel width in pixels (default 10).

    Returns
    -------
    np.ndarray
        Resampled spectrum on the ``wl_final`` grid.
    """
    # 1. Build non-uniform grid with instrumental Δλ spacing
    min_wl, max_wl = float(np.min(wl_orig)), float(np.max(wl_orig))
    wl_new: list[float] = [min_wl]
    wl = min_wl
    while wl < max_wl:
        wl += float(polyval(wl, dl_coeffs) / kernel_width_px)
        wl_new.append(wl)
    wl_arr = np.array(wl_new)

    # 2. Interpolate onto instrumental grid
    f_interp = interp1d(wl_orig, spec_orig, kind="cubic", fill_value="extrapolate")
    spec_new = f_interp(wl_arr)

    # 3. Convolve with Gaussian kernel (instrumental LSF)
    sigma = kernel_width_px / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    kernel = Gaussian1DKernel(stddev=sigma)

    # Pad edges to avoid boundary artifacts
    half_k = int(kernel.dimension / 2.0)
    if half_k > 0:
        spec_new[0] = np.nanmedian(spec_new[:half_k])
        spec_new[-1] = np.nanmedian(spec_new[-half_k:]) if half_k > 0 else spec_new[-1]
    spec_convolved = convolve(spec_new, kernel, boundary="extend")

    # 4. Interpolate back to data wavelength grid
    f_final = interp1d(wl_arr, spec_convolved, kind="cubic", fill_value="extrapolate")
    spec_interp = f_final(wl_final)

    # 5. Apply spectral binning (boxcar convolution)
    if spectral_binning > 1:
        box_kernel = Box1DKernel(spectral_binning)
        spec_final = convolve(spec_interp, box_kernel, boundary="extend")
    else:
        spec_final = spec_interp

    return np.asarray(spec_final)


# ---------------------------------------------------------------------------
# Wavelength correlation offset (diagnostic)
# ---------------------------------------------------------------------------


def calc_corr_offset(
    spectrum1: np.ndarray,
    spectrum2: np.ndarray,
    shift_max: int,
) -> list[float]:
    """Compute Pearson correlation as a function of pixel shift.

    This is a diagnostic to check the wavelength alignment between
    a raw spectrum and an atmospheric transmission spectrum.

    Parameters
    ----------
    spectrum1, spectrum2 : np.ndarray
        The two spectra to cross-correlate.
    shift_max : int
        Maximum shift in pixels (both directions).

    Returns
    -------
    list[float]
        Pearson correlation coefficient for each shift in
        ``range(-shift_max, +shift_max)``.
    """
    import scipy.stats

    finite_mask = np.isfinite(spectrum2)
    s1 = spectrum1[finite_mask]
    s2 = spectrum2[finite_mask]
    n = len(s1)

    rp: list[float] = []
    for k in range(-shift_max, shift_max):
        if k < 0:
            r = scipy.stats.pearsonr(s1[: n + k], s2[-k:])[0]
        else:
            r = scipy.stats.pearsonr(s1[k:], s2[: n - k])[0]
        rp.append(r)
    return rp


# ---------------------------------------------------------------------------
# Top-level airmass correction
# ---------------------------------------------------------------------------


def compute_airmass_correction(
    hdul_sci: fits.HDUList,
    hdul_cal: fits.HDUList,
    wav_sci_m: np.ndarray,
    wav_cal_m: np.ndarray,
    airmass_sci: float,
    airmass_cal: float,
    pwv_sci: float,
    pwv_cal: float,
    output_dir: Path,
    tag_sci: str,
    tag_cal: str,
) -> np.ndarray:
    """Compute the full airmass correction factor for a SCI/CAL pair.

    This orchestrates:
    1. SkyCalc runs for SCI and CAL
    2. Resampling to MATISSE spectral resolution
    3. Division ``trans_cal / trans_sci``

    Parameters
    ----------
    hdul_sci, hdul_cal : fits.HDUList
        Open FITS files (needed for spectral resolution parameters).
    wav_sci_m, wav_cal_m : np.ndarray
        Science and calibrator wavelength grids in metres.
    airmass_sci, airmass_cal : float
        Mean airmass values.
    pwv_sci, pwv_cal : float
        Mean precipitable water vapour in mm.
    output_dir : Path
        Directory for SkyCalc intermediate files.
    tag_sci, tag_cal : str
        Identifiers for file naming (typically the FITS filename stem).

    Returns
    -------
    np.ndarray
        Correction factor array on the science wavelength grid.
        Returns all-ones if SkyCalc is unavailable.
    """
    skycalc_dir = output_dir / "skycalc"
    skycalc_dir.mkdir(parents=True, exist_ok=True)

    # --- SCI ---
    wmin_sci = float(np.min(wav_sci_m)) * 1e9  # m → nm
    wmax_sci = float(np.max(wav_sci_m)) * 1e9
    margin = 0.1 * (wmax_sci - wmin_sci)
    dlambda_sci = get_dlambda(hdul_sci)

    if "IR-N" in tag_sci:
        detected_band = "N"
    elif "IR-LM" in tag_sci:
        detected_band = "LM"
    else:
        detected_band = "unknown"
    logger.info(
        "Computing airmass correction for SCI (band=%s, airmass=%.3f, PWV=%.1f mm)",
        detected_band,
        airmass_sci,
        pwv_sci,
    )

    input_sci = skycalc_dir / f"skycalc_input_sci_{tag_sci}.txt"
    output_sci = skycalc_dir / f"skycalc_output_sci_{tag_sci}.fits"
    create_skycalc_input(
        input_sci,
        airmass_sci,
        pwv_sci,
        wmin_sci - margin,
        wmax_sci + margin,
        wdelta=dlambda_sci,
    )
    if not run_skycalc(input_sci, output_sci):
        logger.warning("SkyCalc failed for SCI — returning unit correction.")
        return np.ones_like(wav_sci_m)

    # --- CAL ---
    wmin_cal = float(np.min(wav_cal_m)) * 1e9
    wmax_cal = float(np.max(wav_cal_m)) * 1e9
    margin_cal = 0.1 * (wmax_cal - wmin_cal)
    dlambda_cal = get_dlambda(hdul_cal)

    input_cal = skycalc_dir / f"skycalc_input_cal_{tag_cal}.txt"
    output_cal = skycalc_dir / f"skycalc_output_cal_{tag_cal}.fits"
    create_skycalc_input(
        input_cal,
        airmass_cal,
        pwv_cal,
        wmin_cal - margin_cal,
        wmax_cal + margin_cal,
        wdelta=dlambda_cal,
    )
    if not run_skycalc(input_cal, output_cal):
        logger.warning("SkyCalc failed for CAL — returning unit correction.")
        return np.ones_like(wav_sci_m)

    # --- Read transmission spectra ---
    wl_um_sci, trans_sci = read_skycalc_output(output_sci)
    wl_um_cal, trans_cal = read_skycalc_output(output_cal)

    # --- Resample to MATISSE resolution ---
    kernel_width_px = 10.0

    dl_coeffs_sci = get_dl_coeffs(hdul_sci)
    binning_sci = get_spectral_average(hdul_sci)
    trans_sci_final = resample_to_matisse_resolution(
        wl_um_sci,
        trans_sci,
        dl_coeffs_sci,
        wav_sci_m * 1e6,  # m → µm
        binning_sci,
        kernel_width_px=kernel_width_px,
    )

    dl_coeffs_cal = get_dl_coeffs(hdul_cal)
    binning_cal = get_spectral_average(hdul_cal)
    trans_cal_final = resample_to_matisse_resolution(
        wl_um_cal,
        trans_cal,
        dl_coeffs_cal,
        wav_cal_m * 1e6,
        binning_cal,
        kernel_width_px=kernel_width_px,
    )

    # --- Correction factor ---
    with np.errstate(divide="ignore", invalid="ignore"):
        correction = trans_cal_final / trans_sci_final
        correction = np.where(np.isfinite(correction), correction, 1.0)
        if detected_band == "LM":
            correction = correction[
                ::-1
            ]  # Reverse to match skycalc output order in LM band

    logger.info(
        "Airmass correction computed (median=%.3f, range=[%.3f, %.3f])",
        np.nanmedian(correction),
        np.nanmin(correction),
        np.nanmax(correction),
    )

    return correction
