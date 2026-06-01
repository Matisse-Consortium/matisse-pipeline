"""
Calibrator model spectrum lookup for MATISSE flux calibration.

This module searches for a synthetic model spectrum of the spectrophotometric
calibrator in two sources (tried in order):

1. **STARSFLUX** — online catalog (MDFC, Vizier II/361) with FITS spectra
   hosted at Leiden University.
2. **Local databases** — FITS files shipped with the MATISSE DRS:
   ``vBoekelDatabase.fits``, ``calib_spec_db_v10.fits``,
   ``calib_spec_db_v10_supplement.fits``.
"""

from __future__ import annotations

import logging
import math
import os
import socket
import urllib.error
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

logger = logging.getLogger(__name__)


def _is_remote_lookup_error(error: Exception) -> bool:
    """Return True when an exception comes from an optional remote lookup.

    STARSFLUX access is best-effort. Transient network failures must not abort
    flux calibration because the workflow can fall back to local databases.
    """

    if isinstance(
        error,
        (
            TimeoutError,
            socket.timeout,
            urllib.error.URLError,
            ConnectionError,
            OSError,
        ),
    ):
        return True

    message = str(error).lower()
    remote_error_markers = (
        "timed out",
        "timeout",
        "connection",
        "temporary failure",
        "name or service not known",
        "network is unreachable",
        "remote end closed connection",
    )
    return any(marker in message for marker in remote_error_markers)


# ---------------------------------------------------------------------------
# Data structure for a calibrator spectrum lookup result
# ---------------------------------------------------------------------------


@dataclass
class CalibratorSpectrum:
    """Result of a calibrator model spectrum lookup."""

    name: str
    """Name of the calibrator as found in the database."""

    diameter_mas: float
    """Uniform-disk diameter in milli-arcseconds."""

    diameter_err_mas: float
    """Uncertainty on the diameter in milli-arcseconds."""

    wavelength: np.ndarray
    """Model wavelength grid (units depend on the database: m or µm)."""

    flux: np.ndarray
    """Model flux array (Jy)."""

    database: str
    """Name/path of the database where the match was found."""

    separation_arcsec: float
    """Angular separation between input coordinates and database match."""

    ra_deg: float
    """RA of the calibrator in the database [deg]."""

    dec_deg: float
    """Dec of the calibrator in the database [deg]."""


# ---------------------------------------------------------------------------
# STARSFLUX (online) lookup
# ---------------------------------------------------------------------------

# _STARSFLUX_BASE_URL = "https://home.strw.leidenuniv.nl/~gamez/catalog/fitsfiles/Aug20"
_STARSFLUX_BASE_URL = (
    "https://home.strw.leidenuniv.nl/~gamez/catalog/fitsfilesFinal/Dec30"
)


def lookup_starsflux(
    cal_name: str,
    ra_deg: float,
    dec_deg: float,
) -> CalibratorSpectrum | None:
    """Search the MDFC / STARSFLUX online catalog for a calibrator spectrum.

    Queries Vizier catalog **II/361** by object name, then downloads the
    corresponding FITS spectrum from the Leiden server.

    Parameters
    ----------
    cal_name : str
        Calibrator name (from ``OI_TARGET``).
    ra_deg, dec_deg : float
        Calibrator coordinates in degrees (ICRS).
    match_radius : float
        Matching radius in arcseconds (currently unused — query is by name).

    Returns
    -------
    CalibratorSpectrum | None
        The matched spectrum, or ``None`` if not found.
    """
    try:
        from astroquery.vizier import Vizier
    except ImportError:
        logger.warning("astroquery not installed — STARSFLUX lookup skipped.")
        return None

    logger.info("Checking if '%s' is in MDFC / STARSFLUX catalog...", cal_name)

    try:
        result = Vizier.query_object(cal_name, catalog="II/361")
        if not result or len(result) == 0:
            logger.info("Calibrator '%s' not found in MDFC catalog.", cal_name)
            return None

        # Clean up the Vizier name to build the URL
        cal_name_vizier = result[0]["Name"][0].replace(" ", "").replace("*", "")
        url = f"{_STARSFLUX_BASE_URL}{cal_name_vizier}.fits"
        logger.info("Downloading spectrum from %s", url)

        # Open the FITS STARFLUX database file and extract the spectrum and diameter
        # _STARSFLUX_BASE_URL is the base URL from Leiden observatory.
        with fits.open(url) as hdu:
            wavelength = hdu[3].data["Wavelength"] * 1e-6  # m
            spectrum = hdu[0].data  # Jy
            diam = hdu[0].header["ANGRMAS"]  # mas
            diam_err = diam * 0.1  # 10% assumed uncertainty

        c_db = SkyCoord(
            result[0]["RAJ2000"][0], result[0]["DEJ2000"][0], unit=(u.hourangle, u.deg)
        )

        # Compute separation for provenance
        c_input = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
        sep_arcsec = c_input.separation(c_db).arcsecond

        logger.info(
            "Found '%s' in STARSFLUX (diam=%.2f mas, sep=%.1f\")",
            cal_name_vizier,
            diam,
            sep_arcsec,
        )

        return CalibratorSpectrum(
            name=cal_name_vizier,
            diameter_mas=float(diam),
            diameter_err_mas=float(diam_err),
            wavelength=wavelength,
            flux=spectrum,
            database=url,
            separation_arcsec=float(sep_arcsec),
            ra_deg=float(c_db.ra.deg),
            dec_deg=float(c_db.dec.deg),
        )

    except urllib.error.URLError as e:
        logger.info(
            "Calibrator '%s' not available in STARSFLUX (%s).",
            cal_name,
            e.reason,
        )
        return None
    except Exception as exc:
        if _is_remote_lookup_error(exc):
            logger.warning(
                "STARSFLUX lookup failed for '%s' (%s: %s). Falling back to local databases.",
                cal_name,
                type(exc).__name__,
                exc,
            )
            return None
        raise


# ---------------------------------------------------------------------------
# Local database lookup
# ---------------------------------------------------------------------------


def _read_diameter_vboekel(
    caldb: fits.HDUList,
    idx: int,
    is_old_format: bool,
) -> tuple[float, float]:
    """Extract diameter from the vBoekel database."""
    if is_old_format:
        diam = 1000.0 * caldb[-2].data["DIAMETER"][idx]
        diam_err = 1000.0 * caldb[-2].data["DIAMETER_ERR"][idx]
    else:
        diam = 1000.0 * caldb["DIAMETERS"].data["DIAMETER"][idx]
        diam_err = 1000.0 * caldb["DIAMETERS"].data["DIAMETER_ERR"][idx]
    return diam, diam_err


def _read_diameter_calib_spec(
    caldb: fits.HDUList,
    idx: int,
    band: str,
) -> tuple[float, float]:
    """Extract diameter from calib_spec_db, trying multiple columns as fallback.

    The priority order follows the legacy code:
    1. UDDL_est / UDDN_est (estimated UD diameter for L or N band)
    2. diam_midi
    3. diam_cohen
    4. UDD_meas
    5. diam_gaia (10% assumed error)
    """
    sources = caldb["SOURCES"].data

    # Band-dependent primary estimate
    if "L" in band:
        diam = sources["UDDL_est"][idx]
    else:
        diam = sources["UDDN_est"][idx]
    diam_err = sources["e_diam_est"][idx]

    # Fallback chain
    fallback_pairs = [
        ("diam_midi", "e_diam_midi"),
        ("diam_cohen", "e_diam_cohen"),
        ("UDD_meas", "e_diam_meas"),
    ]
    for col_diam, col_err in fallback_pairs:
        if math.isnan(diam):
            diam = sources[col_diam][idx]
            diam_err = sources[col_err][idx]

    if math.isnan(diam):
        diam = sources["diam_gaia"][idx]
        diam_err = diam * 0.1

    return float(diam), float(diam_err)


def lookup_local_database(
    database_path: Path | str,
    ra_deg: float,
    dec_deg: float,
    match_radius: float = 15.0,
    band: str = "L",
) -> CalibratorSpectrum | None:
    """Search a single local calibrator database for a matching spectrum.

    Supported databases:
    - ``vBoekelDatabase.fits`` (current and old format)
    - ``calib_spec_db_v10.fits``, ``calib_spec_db_v10_supplement.fits``

    Parameters
    ----------
    database_path : Path | str
        Full path to the FITS database file.
    ra_deg, dec_deg : float
        Calibrator coordinates in degrees (ICRS).
    match_radius : float
        Maximum separation in arcseconds to accept a match.
    band : str
        Spectral band hint (``'L'``, ``'N'``, ``'IR-LM'``, ``'IR-N'``).

    Returns
    -------
    CalibratorSpectrum | None
        The matched spectrum, or ``None`` if no source within *match_radius*.
    """
    database_path = str(database_path)
    caldb_file = os.path.basename(database_path)

    try:
        caldb = fits.open(database_path)
    except FileNotFoundError:
        logger.warning("Database file not found: %s", database_path)
        return None

    # --- Identify column names depending on database format ---
    is_vboekel = "vBoekelDatabase" in caldb_file
    is_old_format = "fitsold" in caldb_file
    is_calib_spec = "calib_spec_db" in caldb_file

    if is_vboekel:
        if is_old_format:
            sources_ext = caldb[8]
        else:
            sources_ext = caldb["SOURCES"]
        name_col, ra_col, dec_col = "NAME", "RAEPP", "DECEPP"
    elif is_calib_spec:
        sources_ext = caldb["SOURCES"]
        name_col, ra_col, dec_col = "name", "ra", "dec"
    else:
        sources_ext = caldb["SOURCES"]
        name_col, ra_col, dec_col = "NAME", "RAEPP", "DECEPP"

    cal_ra_lst = sources_ext.data[ra_col]
    cal_dec_lst = sources_ext.data[dec_col]
    cal_name_lst = sources_ext.data[name_col]

    # --- Cross-match by angular separation ---
    c_cal = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    c_lst = SkyCoord(cal_ra_lst * u.deg, cal_dec_lst * u.deg, frame="icrs")

    sep = c_cal.separation(c_lst)
    min_idx = int(np.nanargmin(sep))
    min_sep_deg = sep[min_idx].value
    min_sep_arcsec = min_sep_deg * 3600.0

    if min_sep_deg >= match_radius / 3600.0:
        logger.info(
            'No match in %s (closest: %s at %.1f")',
            caldb_file,
            cal_name_lst[min_idx],
            min_sep_arcsec,
        )
        caldb.close()
        return None

    logger.info(
        "Found '%s' in %s (sep=%.2f\")",
        cal_name_lst[min_idx],
        caldb_file,
        min_sep_arcsec,
    )

    # --- Extract diameter ---
    if is_vboekel:
        diam, diam_err = _read_diameter_vboekel(caldb, min_idx, is_old_format)
        offset = 9
    elif is_calib_spec:
        diam, diam_err = _read_diameter_calib_spec(caldb, min_idx, band)
        offset = 2
    else:
        diam, diam_err = _read_diameter_vboekel(caldb, min_idx, is_old_format=False)
        offset = 9

    # --- Extract model spectrum ---
    spec_ext = caldb[min_idx + offset]
    wavelength = spec_ext.data["WAVELENGTH"]  # m (for local databases)
    flux = spec_ext.data["FLUX"]  # Jy
    spec_name = spec_ext.header.get("NAME", cal_name_lst[min_idx])

    caldb.close()

    return CalibratorSpectrum(
        name=spec_name,
        diameter_mas=diam,
        diameter_err_mas=diam_err,
        wavelength=wavelength,
        flux=flux,
        database=caldb_file,
        separation_arcsec=min_sep_arcsec,
        ra_deg=float(cal_ra_lst[min_idx]),
        dec_deg=float(cal_dec_lst[min_idx]),
    )


# ---------------------------------------------------------------------------
# Top-level lookup orchestrator
# ---------------------------------------------------------------------------


def lookup_calibrator_spectrum(
    cal_name: str,
    ra_deg: float,
    dec_deg: float,
    cal_database_paths: list[Path | str],
    match_radius: float = 15.0,
    band: str = "L",
) -> CalibratorSpectrum | None:
    """Search for a calibrator model spectrum across all available sources.

    The search order is:
    1. STARSFLUX online catalog (Vizier II/361)
    2. Local FITS databases (in the order given)

    Parameters
    ----------
    cal_name : str
        Calibrator target name from the OIFITS header.
    ra_deg, dec_deg : float
        Calibrator coordinates in degrees (ICRS).
    cal_database_paths : list[Path | str]
        Ordered list of local database FITS files to search.
    match_radius : float
        Maximum angular separation in arcseconds.
    band : str
        Spectral band hint (``'L'``, ``'N'``, ``'IR-LM'``, ``'IR-N'``).

    Returns
    -------
    CalibratorSpectrum | None
        The first matching spectrum found, or ``None`` if no match.
    """
    # 1. Try STARSFLUX online catalog first
    result = lookup_starsflux(cal_name, ra_deg, dec_deg)
    if result is not None:
        return result

    # lookup_starsflux already logged why it failed; continue to local databases
    logger.info("Trying local databases for '%s'...", cal_name)

    # 2. Try local databases in order
    for db_path in cal_database_paths:
        result = lookup_local_database(db_path, ra_deg, dec_deg, match_radius, band)
        if result is not None:
            return result

    logger.error(
        "Calibrator '%s' not found in STARSFLUX nor in any local database.", cal_name
    )
    return None
