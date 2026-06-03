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
from astropy.io import fits

from matisse.core.flux.utils import (
    find_nearest_idx,
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
    wres: float = 300000.0,
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


def _resample_to_matisse_resolution(
    wav_model: np.ndarray,
    flux_model: np.ndarray,
    wav_obs: np.ndarray,
    size_spec_channel: float,
    nsigma: int,
) -> np.ndarray:
    """Resample by weighted bin integration using a wavelength-dependent gaussian kernel (for dense model spectra).

    For each wavelength of wav_obs, sums the model flux contributions
    weighted by a wavelength-dependent gaussian kernel, within a window of +- nsigma*sigma around the local wavelength of wav_obs, where sigma is the standard deviation of the gaussian kernel.

    Parameters
    ----------
    wav_model : np.ndarray
        Model wavelength grid of the calibrator(same units as *wav_obs*; must be sorted ascending).
    flux_model : np.ndarray
        Model spectrum of the calibrator (Jy).
    wav_obs : np.ndarray
        Observation wavelength grid (sorted ascending).
         Model wavelength grid of the calibrator(same units as *wav_obs*; must be sorted ascending).
    size_spec_channel : float
        size of a MATISSE spectral channel (in pix).
    nsigma : int
        size of the window considered for the gaussian averaging (in unit of sigma of the gaussian kernel).

    Returns
    -------
    np.ndarray
        Resampled flux on the ``wav_obs`` grid (Jy).
    """

    spec_resampled = np.zeros_like(wav_obs, dtype=np.float64)

    # computation of the wavelength spacing per pixel
    dlam = np.abs(np.gradient(wav_obs))
    for i, lam0 in enumerate(wav_obs):
        # lam0 = lam_out[i] + 0.5*dlam[i]

        # Computation of the local gaussian kernel at lam0, corresponding to the local size of a MATISSE spectral channel (in wavelength).
        fwhm = size_spec_channel * dlam[i]
        sigma = fwhm / 2.355

        # Size of the spectral window around lam0 where gaussian-weighted averaging will be performed on flux_model
        mask = np.abs(wav_model - lam0) < nsigma * sigma

        lam_mask = wav_model[mask]
        f_mask = flux_model[mask]
        # Case where we are too close to the edges of the wavelength grid
        if len(lam_mask) < 5:
            spec_resampled[i] = np.nan
            continue

        # Gaussian weighted averaging within the spectral window defined by mask
        w = np.exp(-0.5 * ((lam_mask - lam0) / sigma) ** 2)
        w /= np.sum(w)
        spec_resampled[i] = np.sum(f_mask * w)

    return spec_resampled


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
    #    dlambda_sci = get_dlambda(hdul_sci)
    if "IR-N" in tag_sci:
        detected_band = "N"
        detector = "AQUARIUS"
    elif "IR-LM" in tag_sci:
        detected_band = "LM"
        detector = "HAWAII-2RG"
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
        # wdelta=dlambda_sci,
    )
    if not run_skycalc(input_sci, output_sci):
        logger.warning("SkyCalc failed for SCI — returning unit correction.")
        return np.ones_like(wav_sci_m)

    # --- CAL ---
    wmin_cal = float(np.min(wav_cal_m)) * 1e9
    wmax_cal = float(np.max(wav_cal_m)) * 1e9
    margin_cal = 0.1 * (wmax_cal - wmin_cal)
    #   dlambda_cal = get_dlambda(hdul_cal)

    input_cal = skycalc_dir / f"skycalc_input_cal_{tag_cal}.txt"
    output_cal = skycalc_dir / f"skycalc_output_cal_{tag_cal}.fits"
    create_skycalc_input(
        input_cal,
        airmass_cal,
        pwv_cal,
        wmin_cal - margin_cal,
        wmax_cal + margin_cal,
        # wdelta=dlambda_cal,
    )
    if not run_skycalc(input_cal, output_cal):
        logger.warning("SkyCalc failed for CAL — returning unit correction.")
        return np.ones_like(wav_sci_m)

    # --- Read transmission spectra ---
    wl_um_sci, trans_sci = read_skycalc_output(output_sci)
    wl_um_cal, trans_cal = read_skycalc_output(output_cal)
    # --- Resample to MATISSE resolution ---
    # size of the spectral window in unit of sigma of the wavelength-dependent gaussian kernel applied in resample_to_matisse_resolution
    nsigma = 5

    # Determination of the size of a MATISSE spectral channel (in pix)
    if "HAWAI" in detector:
        SIZE_SPEC_CHANNEL = 4.92
    elif "AQUARIUS" in detector:
        SIZE_SPEC_CHANNEL = 7.87

    trans_sci_final = _resample_to_matisse_resolution(
        wl_um_sci,
        trans_sci,
        wav_sci_m * 1e6,  # m → µm
        SIZE_SPEC_CHANNEL,
        nsigma,
    )

    trans_cal_final = _resample_to_matisse_resolution(
        wl_um_cal, trans_cal, wav_cal_m * 1e6, SIZE_SPEC_CHANNEL, nsigma
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
