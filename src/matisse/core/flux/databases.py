"""
Calibrator spectral databases download manager for MATISSE flux calibration.

Databases are hosted on Zenodo and downloaded/cached on first use via
`pooch <https://www.fatiando.org/pooch/>`_.  The local cache lives in the
platform-appropriate app-data directory (``~/.cache/matisse/cal_databases``
on Linux/macOS).

You can override the download by pointing the environment variable
``MATISSE_CAL_DB_PATH`` to a directory that already contains the FITS files::

    export MATISSE_CAL_DB_PATH=/path/to/local/databases

The module exposes a single public function:

.. code-block:: python

    from matisse.core.flux.databases import get_cal_databases_dir

    db_dir = get_cal_databases_dir()  # Path, guaranteed to exist

Version history is tracked automatically via the Zenodo concept record.
New versions published on Zenodo are picked up automatically without
any code change (the concept record ID ``_ZENODO_CONCEPT_RECORD_ID``
always resolves to the latest version via the Zenodo API).
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version & Zenodo metadata
# ---------------------------------------------------------------------------

#: Zenodo *concept* record ID — stable across all versions of the deposit.
#: The Zenodo API resolves this to the latest published version automatically.
_ZENODO_CONCEPT_RECORD_ID = "19004354"

#: Archive filename on Zenodo (single tar.gz containing all FITS databases).
_ARCHIVE_NAME = "matisse_spectrum_database.tar.gz"

#: Expected FITS database files inside the archive (used for validation).
_DB_FILES: list[str] = [
    "vBoekelDatabase.fits",
    "calib_spec_db_v10.fits",
    "calib_spec_db_v10_supplement.fits",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_cal_databases_dir() -> Path:
    """Return the directory that contains the calibrator FITS databases.

    Resolution order
    ----------------
    1. ``MATISSE_CAL_DB_PATH`` environment variable (any absolute path).
    2. ``pooch`` cache – downloads from Zenodo on first call, then reuses.

    Returns
    -------
    Path
        Absolute path to a directory containing the three database files.

    Raises
    ------
    RuntimeError
        If no databases can be found or downloaded.
    """
    # ------------------------------------------------------------------
    # 1. Explicit local override
    # ------------------------------------------------------------------
    env_path = os.environ.get("MATISSE_CAL_DB_PATH")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.is_dir():
            logger.info("Using calibrator databases from MATISSE_CAL_DB_PATH: %s", p)
            return p
        logger.warning("MATISSE_CAL_DB_PATH='%s' does not exist — ignoring.", env_path)

    # ------------------------------------------------------------------
    # 2. Pooch download / cache from Zenodo
    # ------------------------------------------------------------------
    try:
        db_dir = _ensure_pooch_cache()
        return db_dir
    except Exception as exc:
        logger.error("Failed to obtain calibrator databases: %s", exc)

    raise RuntimeError(
        "Calibrator spectral databases not found.\n\n"
        "The databases are hosted on Zenodo and downloaded automatically,\n"
        "but the download failed (no internet? Zenodo down?).\n\n"
        "You can download them manually from:\n"
        f"  https://zenodo.org/records/{_ZENODO_CONCEPT_RECORD_ID}\n\n"
        "Then point MATISSE to the extracted directory:\n"
        "  export MATISSE_CAL_DB_PATH=/path/to/extracted/databases\n"
    )


def _pooch_is_available() -> bool:
    """Return True if ``pooch`` can be imported."""
    try:
        import pooch  # noqa: F401

        return True
    except ImportError:
        return False


def _resolve_latest_zenodo_record() -> tuple[str, str]:
    """Query the Zenodo API to resolve the latest version of the concept record.

    Returns
    -------
    tuple[str, str]
        ``(record_id, version)`` of the latest published version.

    Raises
    ------
    RuntimeError
        If the API call fails or returns unexpected data.
    """
    api_url = f"https://zenodo.org/api/records/{_ZENODO_CONCEPT_RECORD_ID}"
    try:
        try:
            import certifi

            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ssl_ctx = ssl.create_default_context()
        req = urllib.request.Request(api_url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30, context=ssl_ctx) as resp:  # noqa: S310
            data = json.loads(resp.read())
    except Exception as exc:
        raise RuntimeError(
            f"Failed to query Zenodo API for concept record "
            f"{_ZENODO_CONCEPT_RECORD_ID}: {exc}"
        ) from exc

    record_id = str(data["id"])
    version = data.get("metadata", {}).get("version", record_id)
    # Normalize: strip leading "v" so callers can format consistently
    version = version.lstrip("v")
    logger.info("Zenodo: latest database version is %s (record %s)", version, record_id)
    return record_id, version


def _find_cached_databases() -> Path | None:
    """Look for already-extracted databases in the pooch cache directory.

    Returns the directory containing the FITS files, or ``None`` if nothing
    is cached yet.
    """
    try:
        import pooch
    except ImportError:
        return None

    cache_dir = pooch.os_cache("matisse") / "cal_databases"
    if not cache_dir.is_dir():
        return None

    # Scan version directories (most recent first)
    for vdir in sorted(cache_dir.iterdir(), reverse=True):
        candidate = vdir / f"{_ARCHIVE_NAME}.untar"
        if candidate.is_dir():
            found = list(candidate.rglob("*.fits"))
            if found:
                return found[0].parent
    return None


def _ensure_pooch_cache() -> Path:
    """Download the archive from Zenodo, extract it, and return the cache dir.

    If databases are already cached locally, they are returned immediately
    without any network access.  Otherwise the *concept record ID* is
    resolved via the Zenodo API to find the latest version, and the archive
    is downloaded and extracted.
    """
    # 1. Return from cache if already available (works offline)
    cached = _find_cached_databases()
    if cached is not None:
        logger.info("Using cached calibrator databases: %s", cached)
        return cached

    # 2. Need to download — resolve latest version from Zenodo API
    try:
        import pooch
    except ImportError as exc:
        raise ImportError(
            "pooch is required to download calibrator databases.  "
            "Install it with:  pip install pooch"
        ) from exc

    record_id, version = _resolve_latest_zenodo_record()
    base_url = f"https://zenodo.org/records/{record_id}/files"

    cache_dir = pooch.os_cache("matisse") / "cal_databases"

    retriever = pooch.create(
        path=cache_dir,
        base_url=base_url + "/",
        version=f"v{version}",
        version_dev="main",
        registry={_ARCHIVE_NAME: None},
        retry_if_failed=3,
    )

    extracted = retriever.fetch(_ARCHIVE_NAME, processor=pooch.Untar())

    # Find the directory containing the extracted FITS files
    db_dir = _locate_extracted_fits(extracted)
    return db_dir


def _locate_extracted_fits(extracted_paths: list[str]) -> Path:
    """Find the directory containing the FITS databases among extracted paths."""
    fits_paths = [Path(p) for p in extracted_paths if p.endswith(".fits")]
    if not fits_paths:
        raise RuntimeError(
            f"No FITS files found in extracted archive '{_ARCHIVE_NAME}'."
        )
    # All FITS files should be in the same directory
    db_dir = fits_paths[0].parent
    return db_dir


# ---------------------------------------------------------------------------
# CLI helper: download all files eagerly (matisse doctor / prefetch)
# ---------------------------------------------------------------------------


def prefetch_databases() -> Path:
    """Download all calibrator databases to the pooch cache and return the dir.

    Raises ``RuntimeError`` if the download fails.
    Useful to call from ``matisse doctor`` or a setup script.
    """
    return _ensure_pooch_cache()


def database_status() -> dict[str, str]:
    """Return a dict mapping each database filename to its local status.

    Possible statuses: ``"cached"``, ``"missing"``,
    ``"missing (pooch not installed)"``, ``"local_override"``.
    Useful for ``matisse doctor``.
    """
    env_path = os.environ.get("MATISSE_CAL_DB_PATH")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        return {
            fname: "local_override" if (p / fname).exists() else "missing"
            for fname in _DB_FILES
        }

    extract_dir = _find_cached_databases()
    pooch_available = extract_dir is not None or _pooch_is_available()

    status: dict[str, str] = {}
    for fname in _DB_FILES:
        in_cache = extract_dir is not None and (extract_dir / fname).exists()

        if in_cache:
            status[fname] = "cached"
        elif not pooch_available:
            status[fname] = "missing (pooch not installed)"
        else:
            status[fname] = "missing"

    return status
