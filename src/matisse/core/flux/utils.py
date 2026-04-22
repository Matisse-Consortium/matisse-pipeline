"""
Spectral utility helpers for MATISSE flux calibration.

These functions extract instrument-specific spectral parameters
from FITS headers (detector type, dispersion mode, spectral binning)
and provide simple numerical helpers.

"""

from __future__ import annotations

import logging

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detector / dispersion identification
# ---------------------------------------------------------------------------

# Mapping: (detector_chip, dispersion_name) → Δλ in nm
_DLAMBDA_TABLE: dict[tuple[str, str], float] = {
    ("AQUARIUS", "LOW"): 30.0,  # N band, LOW resolution
    ("AQUARIUS", "HIGH"): 3.0,  # N band, HIGH resolution
    ("HAWAII", "LOW"): 8.0,  # LM band, LOW resolution
    ("HAWAII", "MED"): 0.6,  # LM band, MED resolution
    # HIGH and HIGH+ modes are not yet characterized
}

# Mapping: (detector_chip, dispersion_name) → polynomial coefficients for Δλ(λ)
# Used by transform_spectrum_to_real_spectral_resolution()
# Coefficients are for numpy.polynomial.polynomial.polyval (ascending order)
_DL_COEFFS_TABLE: dict[tuple[str, str], list[float]] = {
    ("AQUARIUS", "LOW"): [0.10600484, 0.01502548, 0.00294806, -0.00021434],
    ("AQUARIUS", "HIGH"): [
        -8.02282965e-05,
        3.83260266e-03,
        7.60090459e-05,
        -4.30753848e-07,
    ],
    ("HAWAII", "LOW"): [0.09200542, -0.03281159, 0.02166703, -0.00309248],
    ("HAWAII", "MED"): [
        2.73866174e-10,
        2.00286100e-03,
        1.33829137e-06,
        -4.46578231e-10,
    ],
}


def _identify_detector_dispersion(header: fits.Header) -> tuple[str, str]:
    """Identify detector family and dispersion mode from FITS header.

    Parameters
    ----------
    header : fits.Header
        Primary HDU header.

    Returns
    -------
    tuple[str, str]
        (detector_family, dispersion_mode) e.g. ("HAWAII", "LOW").

    Raises
    ------
    ValueError
        If the detector chip name is not recognized.
    """
    chip_name = header["HIERARCH ESO DET CHIP NAME"]

    if "AQUARIUS" in chip_name:
        detector = "AQUARIUS"
        dispname = header["HIERARCH ESO INS DIN NAME"]  # N band keyword
    elif "HAWAII" in chip_name:
        detector = "HAWAII"
        dispname = header["HIERARCH ESO INS DIL NAME"]  # LM band keyword
    else:
        msg = f"Unknown detector chip: {chip_name}"
        raise ValueError(msg)

    # Extract the resolution keyword (LOW / MED / HIGH / HIGH+)
    for mode in ("LOW", "MED", "HIGH"):
        if mode in dispname:
            return detector, mode

    msg = f"Unknown dispersion mode: {dispname}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Public spectral helpers
# ---------------------------------------------------------------------------


def get_dlambda(hdul: fits.HDUList) -> float:
    """Return the spectral channel width Δλ in nm.

    Parameters
    ----------
    hdul : fits.HDUList
        Opened FITS file (needs primary header).

    Returns
    -------
    float
        Δλ in nm. ``nan`` if the mode is not characterized.
    """
    try:
        detector, mode = _identify_detector_dispersion(hdul[0].header)
    except ValueError:
        return float("nan")

    return _DLAMBDA_TABLE.get((detector, mode), float("nan"))


def get_dl_coeffs(hdul: fits.HDUList) -> list[float]:
    """Return the polynomial coefficients for Δλ(λ).

    These coefficients are used with ``numpy.polynomial.polynomial.polyval``
    to compute the spectral channel width as a function of wavelength.

    Parameters
    ----------
    hdul : fits.HDUList
        Opened FITS file (needs primary header).

    Returns
    -------
    list[float]
        Polynomial coefficients [c0, c1, c2, c3] for polyval.

    Raises
    ------
    ValueError
        If the detector/mode combination has no known coefficients.
    """
    detector, mode = _identify_detector_dispersion(hdul[0].header)
    key = (detector, mode)
    if key not in _DL_COEFFS_TABLE:
        msg = f"No Δλ coefficients for {detector}/{mode}"
        raise ValueError(msg)
    return _DL_COEFFS_TABLE[key]


def get_spectral_average(hdul: fits.HDUList) -> float:
    """Extract the spectral average parameter from reduction recipe headers.

    The DRS stores recipe parameters as numbered PARAM keywords in the
    primary header. This scans them to find ``spectralAverage``.

    Parameters
    ----------
    hdul : fits.HDUList
        Opened FITS file.

    Returns
    -------
    float
        Spectral binning value, or ``nan`` if not found.
    """
    header = hdul[0].header
    spectral_average = float("nan")

    for i in range(1, 20):
        name_key = f"HIERARCH ESO PRO REC1 PARAM{i} NAME"
        if name_key in header and (
            "spectralAverage" in header[name_key]
            or "spectralBinning" in header[name_key]
        ):
            value_key = f"HIERARCH ESO PRO REC1 PARAM{i} VALUE"
            spectral_average = float(header[value_key])
            logger.debug("Spectral average = %.1f (from PARAM%d)", spectral_average, i)
            break

    return spectral_average


def find_nearest_idx(array: list[float] | np.ndarray, value: float) -> int:
    """Return the index of the element closest to *value*.

    Parameters
    ----------
    array : array-like
        1-D array of values.
    value : float
        Target value.

    Returns
    -------
    int
        Index of the nearest element.
    """
    arr = np.asarray(array)
    return int(np.abs(arr - value).argmin())
