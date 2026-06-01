"""
Transfer function computation and application for MATISSE flux calibration.

This module implements the core calibration algebra:

**Total flux** (mode ``'flux'``)::

    F_sci_cal = (F_sci / F_cal) × S_model × C_airmass

Written to ``OI_FLUX.FLUXDATA``.

**Correlated flux** (mode ``'corrflux'``)::

    CF_sci_cal = (CF_sci / CF_cal) × V_cal_UD × S_model × C_airmass

where ``V_cal_UD = 2·J₁(z)/z`` is the uniform-disk visibility of the
calibrator, and the result is written to ``OI_VIS.VISAMP``.

Also handles:
- Resampling the calibrator model spectrum to the observation wavelength grid.
- Matching baselines between SCI and CAL by station index.
- Propagating errors from the raw spectra and the calibrator diameter uncertainty.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.special import j1, jv

from matisse.core.flux.calibrator_spectrum import CalibratorSpectrum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model spectrum resampling
# ---------------------------------------------------------------------------


def resample_model_spectrum(
    wav_model: np.ndarray,
    flux_model: np.ndarray,
    wav_obs: np.ndarray,
    detector_type: str,
    nsigma: int = 5,
) -> np.ndarray:
    """Resample a calibrator model spectrum onto the MATISSE wavelength grid.

    Two strategies are used depending on the relative sampling:

    - If the model is **much denser** than the observation (>2× more points
      in the overlap region), a wavelength-dependent gaussin kernel is computed and applied to properly average the model within each MATISSE spectral channel.
    - Otherwise, a **cubic interpolation** is used.

    Parameters
    ----------
    wav_model : np.ndarray
        Model wavelength grid of the calibrator(same units as *wav_obs*; must be sorted ascending).
    flux_model : np.ndarray
        Model spectrum of the calibrator (Jy).
    wav_obs : np.ndarray
        Observation wavelength grid (sorted ascending).
         Model wavelength grid of the calibrator(same units as *wav_obs*; must be sorted ascending).
    detector_type : str
        Detector used for the observations ("HAWAII-2RG" or "AQUARIUS").
    nsigma : int
        size of the window considered for the gaussian averaging (in unit of sigma of the gaussian kernel).

    Returns
    -------
    np.ndarray
        Resampled flux on the ``wav_obs`` grid (Jy).
    """

    # Check sampling ratio in the overlap region
    overlap_mask = (wav_model > np.nanmin(wav_obs)) & (wav_model < np.nanmax(wav_obs))
    n_model_in_range = int(np.nansum(overlap_mask))

    # Get the size of a spectral channel (in pix) depending on the detector (spectral band)
    if "HAWAII" in detector_type:
        size_spec_channel = 4.92
    else:
        size_spec_channel = 7.87

    if 2.0 * len(wav_obs) < n_model_in_range:
        # Model is much denser → bin-weighted integration
        logger.debug(
            "Model denser than obs (%d vs %d) → bin integration.",
            n_model_in_range,
            len(wav_obs),
        )
        return _resample_by_gaussian_kernel(
            wav_obs,
            wav_model,
            flux_model,
            size_spec_channel,
            nsigma,
        )
    else:
        # Comparable or sparser → cubic interpolation
        logger.debug("Model comparable/sparser → cubic interpolation.")
        f = interp1d(wav_model, flux_model, kind="cubic", bounds_error=False)
        return f(wav_obs)


# def resample_model_spectrum(
#    wav_model: np.ndarray,
#    flux_model: np.ndarray,
#    wav_obs: np.ndarray,
# ) -> np.ndarray:
#    """Resample a calibrator model spectrum onto the observation wavelength grid.

#    Two strategies are used depending on the relative sampling:

#    - If the model is **much denser** than the observation (>2× more points
#      in the overlap region), a **bin-weighted integration** is performed
#      to properly average the model within each observation channel.
#    - Otherwise, a **cubic interpolation** is used.

#    Parameters
#    ----------
#    wav_model : np.ndarray
#        Model wavelength grid (same units as *wav_obs*; must be sorted ascending).
#    flux_model : np.ndarray
#        Model flux array (Jy).
#    wav_obs : np.ndarray
#        Observation wavelength grid (sorted ascending).

#    Returns
#    -------
#    np.ndarray
#        Resampled flux on the ``wav_obs`` grid (Jy).
#    """
#    # Build bin edges for the model grid
#    wav_bin_lower_model, wav_bin_upper_model, d_wav_model = _compute_bin_edges(
#        wav_model
#    )

# Check sampling ratio in the overlap region
#    overlap_mask = (wav_model > np.nanmin(wav_obs)) & (wav_model < np.nanmax(wav_obs))
#    n_model_in_range = int(np.nansum(overlap_mask))

#    if 2.0 * len(wav_obs) < n_model_in_range:
#        # Model is much denser → bin-weighted integration
#        logger.debug(
#            "Model denser than obs (%d vs %d) → bin integration.",
#            n_model_in_range,
#            len(wav_obs),
#        )
#        return _resample_by_bin_integration(
#            wav_obs,
#            wav_model,
#            flux_model,
#            wav_bin_lower_model,
#            wav_bin_upper_model,
#            d_wav_model,
#        )
#    else:
#        # Comparable or sparser → cubic interpolation
#        logger.debug("Model comparable/sparser → cubic interpolation.")
#        f = interp1d(wav_model, flux_model, kind="cubic", bounds_error=False)
#        return f(wav_obs)


def _compute_bin_edges(
    wav: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bin lower edges, upper edges, and widths for a wavelength grid.

    Parameters
    ----------
    wav : np.ndarray
        Sorted wavelength array.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (bin_lower, bin_upper, bin_width)
    """
    wl = np.concatenate(([wav[0] - (wav[1] - wav[0])], wav))
    wh = np.concatenate((wav, [wav[-1] + (wav[-1] - wav[-2])]))
    wm = (wh + wl) / 2.0
    return wm[:-1], wm[1:], wm[1:] - wm[:-1]


def _resample_by_gaussian_kernel(
    wav_obs: np.ndarray,
    wav_model: np.ndarray,
    flux_model: np.ndarray,
    size_spec_channel: float,
    nsigma: int,
) -> np.ndarray:
    """Resample by weighted bin integration using a wavelength-dependent gaussian kernel (for dense model spectra).

    For each wavelength of wav_obs, sums the model flux contributions
    weighted by a wavelength-dependent gaussian kernel, within a window of +- nsigma*sigma around the local wavelength of wav_obs, where sigma is the standard deviation of the gaussian kernel.
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


def _resample_by_bin_integration(
    wav_obs: np.ndarray,
    wav_model: np.ndarray,
    flux_model: np.ndarray,
    wav_bin_lower_model: np.ndarray,
    wav_bin_upper_model: np.ndarray,
    d_wav_model: np.ndarray,
) -> np.ndarray:
    """Resample by weighted bin integration (for dense model spectra).

    For each observation channel, sums the model flux contributions
    weighted by the fractional overlap between model and observation bins.
    """
    obs_lower, obs_upper, d_wav_obs = _compute_bin_edges(wav_obs)
    resampled = np.zeros_like(wav_obs, dtype=np.float64)

    for i in range(len(wav_obs)):
        # Find model bins that overlap with this observation bin
        wu = wav_bin_upper_model - obs_lower[i]
        wl = obs_upper[i] - wav_bin_lower_model
        overlap_idx = np.where((wu > 0.0) & (wl > 0.0))[0]

        if len(overlap_idx) == 0:
            continue

        # Weighted sum: partial overlap at edges, full overlap in middle
        total = 0.0
        total += (wav_bin_upper_model[overlap_idx[0]] - obs_lower[i]) * flux_model[
            overlap_idx[0]
        ]
        total += (obs_upper[i] - wav_bin_lower_model[overlap_idx[-1]]) * flux_model[
            overlap_idx[-1]
        ]
        for j in range(1, len(overlap_idx) - 1):
            total += flux_model[overlap_idx[j]] * d_wav_model[overlap_idx[j]]

        resampled[i] = total / d_wav_obs[i]

    return resampled


# ---------------------------------------------------------------------------
# Uniform-disk visibility model
# ---------------------------------------------------------------------------


def uniform_disk_visibility(
    diameter_mas: float,
    baseline_m: float,
    wavelength_m: np.ndarray,
) -> np.ndarray:
    """Compute the visibility of a uniform disk.

    .. math::
        V = \\frac{2 J_1(z)}{z}, \\quad z = \\pi \\theta B / \\lambda

    Parameters
    ----------
    diameter_mas : float
        Angular diameter in milli-arcseconds.
    baseline_m : float
        Projected baseline length in metres.
    wavelength_m : np.ndarray
        Wavelength grid in metres.

    Returns
    -------
    np.ndarray
        Visibility amplitude array.
    """
    diameter_rad = diameter_mas / 1000.0 / 3600.0 * math.pi / 180.0
    spatial_freq = baseline_m / wavelength_m
    z = math.pi * diameter_rad * spatial_freq
    return 2.0 * j1(z) / z


def uniform_disk_visibility_error(
    diameter_mas: float,
    diameter_err_mas: float,
    baseline_m: float,
    wavelength_m: np.ndarray,
) -> np.ndarray:
    """Compute the uncertainty on the UD visibility due to diameter error.

    .. math::
        \\sigma_V = V \\times \\frac{\\sigma_\\theta}{\\theta}
        \\times \\left| \\frac{z J_2(z)}{J_1(z)} \\right|

    Parameters
    ----------
    diameter_mas, diameter_err_mas : float
        Diameter and its uncertainty in milli-arcseconds.
    baseline_m : float
        Projected baseline length in metres.
    wavelength_m : np.ndarray
        Wavelength grid in metres.

    Returns
    -------
    np.ndarray
        Visibility error array.
    """
    diameter_rad = diameter_mas / 1000.0 / 3600.0 * math.pi / 180.0
    diam_err_rad = diameter_err_mas / 1000.0 / 3600.0 * math.pi / 180.0
    spatial_freq = baseline_m / wavelength_m
    z = math.pi * diameter_rad * spatial_freq

    vis = 2.0 * j1(z) / z
    return vis * (diam_err_rad / diameter_rad) * np.abs(z * jv(2, z) / j1(z))


# ---------------------------------------------------------------------------
# Flux calibration application
# ---------------------------------------------------------------------------


def _flip_if_lband(arr: np.ndarray, band: str) -> np.ndarray:
    """Flip array for L-band data (wavelength order convention)."""
    if "L" in band:
        return np.flip(arr)
    return arr


def calibrate_total_flux(
    hdul_sci: fits.HDUList,
    hdul_cal: fits.HDUList,
    hdul_out: fits.HDUList,
    spectrum_cal_resampled: np.ndarray,
    airmass_correction: np.ndarray,
    band: str,
) -> None:
    """Apply total flux calibration to OI_FLUX table.

    Writes calibrated FLUXDATA and FLUXERR into *hdul_out* (in-place).

    Parameters
    ----------
    hdul_sci, hdul_cal : fits.HDUList
        Open science and calibrator FITS files.
    hdul_out : fits.HDUList
        Output file opened in update mode.
    spectrum_cal_resampled : np.ndarray
        Model spectrum resampled to the observation grid (Jy).
    airmass_correction : np.ndarray
        Airmass correction factor (or ones).
    band : str
        Band identifier (containing ``'L'`` triggers wavelength flip).
    """
    try:
        flux_sci = hdul_sci["OI_FLUX"].data["FLUXDATA"]
        flux_cal = hdul_cal["OI_FLUX"].data["FLUXDATA"]
    except KeyError:
        logger.warning("No OI_FLUX table found — skipping total flux calibration.")
        return

    n_exp_sci = len(flux_sci)
    n_exp_cal = len(flux_cal)
    if n_exp_sci != n_exp_cal:
        msg = (
            f"SCI and CAL have different number of OI_FLUX rows "
            f"({n_exp_sci} vs {n_exp_cal}). Cannot calibrate."
        )
        logger.error(msg)
        raise ValueError(msg)

    # Update units and calibration status
    hdul_out["OI_FLUX"].header["TUNIT5"] = "Jy"
    hdul_out["OI_FLUX"].header["TUNIT6"] = "Jy"
    hdul_out["OI_FLUX"].header["CALSTAT"] = "C"

    for j in range(n_exp_sci):
        f_sci = _flip_if_lband(flux_sci[j].copy(), band)
        f_cal = _flip_if_lband(flux_cal[j].copy(), band)
        ferr_sci = _flip_if_lband(hdul_sci["OI_FLUX"].data["FLUXERR"][j].copy(), band)
        ferr_cal = _flip_if_lband(hdul_cal["OI_FLUX"].data["FLUXERR"][j].copy(), band)

        # Calibrated flux: F_sci / F_cal × model × airmass_corr
        f_calibrated = f_sci / f_cal * spectrum_cal_resampled * airmass_correction

        # Error propagation (quadrature)
        with np.errstate(divide="ignore", invalid="ignore"):
            ferr_calibrated = (
                np.abs(f_sci / f_cal)
                * np.sqrt((ferr_sci / f_sci) ** 2 + (ferr_cal / f_cal) ** 2)
                * spectrum_cal_resampled
            )

        hdul_out["OI_FLUX"].data["FLUXDATA"][j] = _flip_if_lband(f_calibrated, band)
        hdul_out["OI_FLUX"].data["FLUXERR"][j] = _flip_if_lband(ferr_calibrated, band)

    logger.info("Total flux calibration applied (%d exposures).", n_exp_sci)


def _find_matching_baseline(
    sta_index_sci: np.ndarray,
    sta_indices_cal: np.ndarray,
) -> int | None:
    """Find the CAL row with matching station pair (order-independent).

    Parameters
    ----------
    sta_index_sci : np.ndarray
        Station index pair ``[sta1, sta2]`` from the SCI file.
    sta_indices_cal : np.ndarray
        Array of station index pairs from the CAL file (shape ``(N, 2)``).

    Returns
    -------
    int | None
        Index of the matching row in CAL, or ``None``.
    """
    s0, s1 = sta_index_sci[0], sta_index_sci[1]
    for i, pair in enumerate(sta_indices_cal):
        if (s0 == pair[0] and s1 == pair[1]) or (s0 == pair[1] and s1 == pair[0]):
            return i
    return None


def calibrate_correlated_flux(
    hdul_sci: fits.HDUList,
    hdul_cal: fits.HDUList,
    hdul_out: fits.HDUList,
    spectrum_cal_resampled: np.ndarray,
    airmass_correction: np.ndarray,
    diameter_mas: float,
    diameter_err_mas: float,
    wav_cal_m: np.ndarray,
    band: str,
) -> None:
    """Apply correlated flux calibration to OI_VIS table.

    For each baseline, computes the UD visibility of the calibrator
    and applies:
    ``CF_cal = CF_sci / CF_cal_obs × V_cal_UD × S_model × C_airmass``

    Writes calibrated VISAMP and VISAMPERR into *hdul_out* (in-place).

    Parameters
    ----------
    hdul_sci, hdul_cal : fits.HDUList
        Open science and calibrator FITS files.
    hdul_out : fits.HDUList
        Output file opened in update mode.
    spectrum_cal_resampled : np.ndarray
        Model spectrum resampled to the observation wavelength grid (Jy).
    airmass_correction : np.ndarray
        Airmass correction factor.
    diameter_mas, diameter_err_mas : float
        Calibrator uniform-disk diameter and error in mas.
    wav_cal_m : np.ndarray
        Calibrator wavelength grid in metres (for UD visibility computation).
    band : str
        Band identifier.
    """
    vis_data_sci = hdul_sci["OI_VIS"].data
    vis_data_cal = hdul_cal["OI_VIS"].data
    sta_indices_cal = vis_data_cal["STA_INDEX"]

    for j in range(len(vis_data_sci["VISAMP"])):
        sta_sci = vis_data_sci["STA_INDEX"][j]

        # Find matching baseline in calibrator
        idx_cal = _find_matching_baseline(sta_sci, sta_indices_cal)
        if idx_cal is None:
            logger.warning(
                "No matching baseline in CAL for STA_INDEX=%s — skipping.", sta_sci
            )
            continue

        # Raw correlated fluxes
        cf_sci = _flip_if_lband(vis_data_sci["VISAMP"][j].copy(), band)
        cf_cal = _flip_if_lband(vis_data_cal["VISAMP"][idx_cal].copy(), band)
        cferr_sci = _flip_if_lband(vis_data_sci["VISAMPERR"][j].copy(), band)
        cferr_cal = _flip_if_lband(vis_data_cal["VISAMPERR"][idx_cal].copy(), band)

        # Projected baseline
        uu = vis_data_cal["UCOORD"][idx_cal]
        vv = vis_data_cal["VCOORD"][idx_cal]
        baseline = np.sqrt(uu**2 + vv**2)

        # UD visibility of the calibrator
        vis_cal = uniform_disk_visibility(diameter_mas, baseline, wav_cal_m)
        vis_err_cal = uniform_disk_visibility_error(
            diameter_mas, diameter_err_mas, baseline, wav_cal_m
        )

        # Calibrated correlated flux
        cf_calibrated = (
            cf_sci / cf_cal * vis_cal * spectrum_cal_resampled * airmass_correction
        )

        # Error propagation (includes diameter uncertainty)
        with np.errstate(divide="ignore", invalid="ignore"):
            cferr_calibrated = cf_calibrated * np.sqrt(
                (cferr_sci / cf_sci) ** 2
                + (cferr_cal / cf_cal) ** 2
                + (vis_err_cal / vis_cal) ** 2
            )

        hdul_out["OI_VIS"].data["VISAMP"][j] = _flip_if_lband(cf_calibrated, band)
        hdul_out["OI_VIS"].data["VISAMPERR"][j] = _flip_if_lband(cferr_calibrated, band)

    # Update units
    hdul_out["OI_VIS"].header["TUNIT5"] = "Jy"
    hdul_out["OI_VIS"].header["TUNIT6"] = "Jy"

    logger.info(
        "Correlated flux calibration applied (%d baselines).", len(vis_data_sci)
    )


# ---------------------------------------------------------------------------
# Provenance header update
# ---------------------------------------------------------------------------


def write_provenance_headers(
    hdul_out: fits.HDUList,
    cal_name: str,
    ra_cal: float,
    dec_cal: float,
    airmass_cal: float,
    pwv_cal: float,
    seeing_cal: float,
    tau0_cal: float,
    tpl_start_cal: str,
    cal_spectrum: CalibratorSpectrum,
) -> None:
    """Write calibration provenance keywords into the output primary header.

    These follow the ESO HIERARCH convention used by the MATISSE DRS.

    Parameters
    ----------
    hdul_out : fits.HDUList
        Output FITS file opened in update mode.
    cal_name : str
        Calibrator name from OI_TARGET.
    ra_cal, dec_cal : float
        Calibrator coordinates from the OIFITS file [deg].
    airmass_cal : float
        Mean calibrator airmass.
    pwv_cal : float
        Mean precipitable water vapour [mm].
    seeing_cal : float
        Mean seeing [arcsec].
    tau0_cal : float
        Mean coherence time [s].
    tpl_start_cal : str
        Template start timestamp of the calibrator observation.
    cal_spectrum : class 'CalibratorSpectrum'
        The calibrator spectrum object used for calibration (for provenance).
    """

    hdr = hdul_out[0].header
    hdr["HIERARCH ESO PRO CAL NAME"] = cal_name
    hdr["HIERARCH ESO PRO CAL RA"] = (ra_cal, "[deg]")
    hdr["HIERARCH ESO PRO CAL DEC"] = (dec_cal, "[deg]")
    hdr["HIERARCH ESO PRO CAL AIRM"] = airmass_cal
    hdr["HIERARCH ESO PRO CAL IWV"] = pwv_cal
    hdr["HIERARCH ESO PRO CAL FWHM"] = (seeing_cal, "[arcsec]")
    hdr["HIERARCH ESO PRO CAL TAU0"] = (tau0_cal, "Coherence time [s]")
    hdr["HIERARCH ESO PRO CAL TPL START"] = tpl_start_cal
    hdr["HIERARCH ESO PRO CAL DB NAME"] = (cal_spectrum.name, "Name in cal database")
    hdr["HIERARCH ESO PRO CAL DB DBNAME"] = (cal_spectrum.database, "Cal database name")
    hdr["HIERARCH ESO PRO CAL DB RA"] = (cal_spectrum.ra_deg, "[deg]")
    hdr["HIERARCH ESO PRO CAL DB DEC"] = (cal_spectrum.dec_deg, "[deg]")
    hdr["HIERARCH ESO PRO CAL DB DIAM"] = (
        cal_spectrum.diameter_mas,
        "Calibrator diameter [mas]",
    )
    hdr["HIERARCH ESO PRO CAL DB ERRDIAM"] = (
        round(cal_spectrum.diameter_err_mas, 4),
        "Error in calibrator diameter [mas]",
    )
    hdr["HIERARCH ESO PRO CAL DB SEP"] = (
        round(cal_spectrum.separation_arcsec, 4),
        "Sep input-calDB [arcsec]",
    )
    hdr.set("HIERARCH ESO PRO CATG", "TARGET_FLUXCAL_INT", comment="")

    logger.debug("Provenance headers written.")
