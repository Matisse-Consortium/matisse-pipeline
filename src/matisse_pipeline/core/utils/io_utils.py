import glob
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from astropy.io import fits

from matisse_pipeline.core.lib_auto_pipeline import matisse_type
from matisse_pipeline.core.utils.log_utils import log


# --- Input resolution helpers ---
def _is_json_list(text: str) -> bool:
    """Return True if string looks like a JSON list."""
    t = text.strip()
    return t.startswith("[") and t.endswith("]")


def _read_list_file(path: Path) -> list[str]:
    """Read a text file containing one path per line, skipping blanks and comments."""
    lines: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def resolve_raw_input(raw_spec: str | Sequence[str]) -> tuple[list[Path], str]:
    """
    Normalize 'raw_spec' into a list of FITS file paths and detect the source type.

    Accepted forms:
      - Directory path (glob 'MATIS*.fits')
      - Single .fits file
      - JSON list string '["/path/a.fits", "/path/b.fits"]'
      - Text file (.lst, .list, .txt)
      - Python list/tuple (already resolved)
      - Glob pattern ('/data/*.fits')

    Returns
    -------
    files : list[Path]
        Valid FITS files.
    """
    paths: list[Path] = []
    source: str = "unknown"

    # Case 1: already a list-like (Sequence[str])
    if isinstance(raw_spec, (list, tuple)):
        paths = [Path(p) for p in raw_spec]
        source = "explicit Python list"

    # Case 2: string input
    else:
        raw_str = str(raw_spec).strip()
        p = Path(raw_str)

        if p.is_dir():
            # Directory -> glob for MATIS*.fits
            paths = [Path(x) for x in glob.glob(str(p / "MATIS*.fits"))]
            source = "directory glob (MATIS*.fits)"
        elif p.is_file():
            if p.suffix.lower() == ".fits":
                paths = [p]
                source = "single FITS file"
            elif p.suffix.lower() in {".lst", ".list", ".txt"}:
                # Text file with one path per line
                items = _read_list_file(p)
                paths = [Path(x) for x in items]
                source = f"text file list ({p.name})"
            else:
                # Fallback: if user gave a plain file, try to treat it as a list file
                items = _read_list_file(p)
                paths = [Path(x) for x in items]
                source = f"custom file list ({p.name})"

        elif _is_json_list(raw_str):
            try:
                items = json.loads(raw_str)
                if not isinstance(items, list):
                    raise ValueError("JSON payload is not a list.")
                paths = [Path(x) for x in items]
                source = "JSON list"
            except Exception as err:
                log.error("Failed to parse JSON list for raw files.")
                raise ValueError("Invalid JSON list for raw files.") from err
        else:
            # As a last resort, treat as a glob pattern
            paths = [Path(x) for x in glob.glob(raw_str)]
            source = "glob pattern"

    # Validate existence
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Some raw files do not exist: {missing[:3]}{' ...' if len(missing) > 3 else ''}"
        )

    # Keep only FITS files
    fits = [p for p in paths if p.suffix.lower() == ".fits"]
    if not fits:
        raise FileNotFoundError(
            "No FITS files found in the provided raw specification."
        )

    # De-duplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in fits:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique, source


def check_for_calib_file(allhdr):
    for hdr in allhdr:
        tagCalib = matisse_type(hdr)
        print(tagCalib)


def open_oifits(oi_file):
    try:
        hdu = fits.open(oi_file)
    except OSError:
        print("Unable to read fits file: " + oi_file)
        return {}

    hdr = hdu[0].header

    wl = hdu["OI_WAVELENGTH"].data["EFF_WAVE"]
    dic = {"WLEN": wl}

    dic["HDR"] = hdr
    dic["file"] = oi_file
    try:
        dic["SEEING"] = (
            hdr["HIERARCH ESO ISS AMBI FWHM START"]
            + hdr["HIERARCH ESO ISS AMBI FWHM END"]
        ) / 2.0
    except Exception:
        dic["SEEING"] = 0

    try:
        dic["TAU0"] = (
            hdr["HIERARCH ESO ISS AMBI TAU0 START"]
            + hdr["HIERARCH ESO ISS AMBI TAU0 END"]
        ) / 2.0
    except Exception:
        dic["TAU0"] = 0

    target_name = hdu["OI_TARGET"].data["TARGET"][0]
    if not target_name:
        try:
            target_name = hdr["HIERARCH ESO OBS TARG NAME"]
        except KeyError:
            print("Target name not found.")
            target_name = ""

    dic["TARGET"] = target_name

    # Fix eventual bad target identification
    # dic['TARGET'] = resolve_target(dic)

    try:
        catg = hdr["ESO PRO CATG"]
        if catg == "TARGET_RAW_INT":
            target_category = "SCI"
        else:
            target_category = "CAL"
        # target_category = hdu['OI_TARGET'].data['CATEGORY'][0]  # "CAL" or "SCI"
    except KeyError:
        print("Target category not found.")
        target_category = "CAL"
    dic["CATEGORY"] = target_category
    try:
        dateobs = hdr["DATE-OBS"]
    except KeyError:
        dateobs = ""
    dic["DATEOBS"] = dateobs
    try:
        det_name = hdr["HIERARCH ESO DET CHIP NAME"]
    except KeyError:
        print("Detector name not found.")
        det_name = ""

    if det_name == "AQUARIUS":
        band = "N"
    elif det_name == "HAWAII-2RG":
        band = "LM"
    elif hdr["FILTER1"] == "H_band":
        band = "H"
        print("I found the mircx band!")
    else:
        band = ""

    dic["BAND"] = band
    try:
        if det_name == "AQUARIUS":
            dispersion_name = hdr["HIERARCH ESO INS DIN NAME"]
        else:
            dispersion_name = hdr["HIERARCH ESO INS DIL NAME"]
    except KeyError:
        print("Dispersion name not found.")
        dispersion_name = ""
    dic["DISP"] = dispersion_name
    try:
        DIT = hdr["HIERARCH ESO DET SEQ1 DIT"]  # (s)
    except KeyError:
        DIT = np.nan
        print("DIT not found")
    dic["DIT"] = DIT
    try:
        BCD1 = hdr["HIERARCH ESO INS BCD1 NAME"]
        BCD2 = hdr["HIERARCH ESO INS BCD2 NAME"]
    except KeyError:
        BCD1 = ""
        BCD2 = ""
        print("BCD NAME not found")
    dic["BCD1NAME"] = BCD1
    dic["BCD2NAME"] = BCD2
    try:
        dic["TEL_NAME"] = hdu["OI_ARRAY"].data["TEL_NAME"]
        dic["STA_NAME"] = hdu["OI_ARRAY"].data["STA_NAME"]
        dic["STA_INDEX"] = hdu["OI_ARRAY"].data["STA_INDEX"]
    except KeyError:
        dic["TEL_NAME"] = {}
        dic["STA_NAME"] = {}
        dic["STA_INDEX"] = {}
        print("Key in table OI_ARRAY not found")
    try:
        dic["VIS"] = {}
        dic["VIS"]["VISAMP"] = hdu["OI_VIS"].data["VISAMP"]
        dic["VIS"]["VISAMPERR"] = hdu["OI_VIS"].data["VISAMPERR"]
        dic["VIS"]["DPHI"] = hdu["OI_VIS"].data["VISPHI"]
        dic["VIS"]["DPHIERR"] = hdu["OI_VIS"].data["VISPHIERR"]
        dic["VIS"]["FLAG"] = hdu["OI_VIS"].data["FLAG"]
        try:
            dic["VIS"]["CFLUX"] = hdu["OI_VIS"].data["CFXAMP"]
            dic["VIS"]["CFLUXERR"] = hdu["OI_VIS"].data["CFXAMPERR"]
        except Exception:
            pass
            # print("WARNING: No correlated fluxes in this OI_VIS table!")

        dic["VIS"]["U"] = hdu["OI_VIS"].data["UCOORD"]
        dic["VIS"]["V"] = hdu["OI_VIS"].data["VCOORD"]
        dic["VIS"]["TIME"] = hdu["OI_VIS"].data["MJD"]
        if dic["VIS"]["TIME"][0] < 50000:
            dic["VIS"]["TIME"] = np.full(len(hdu["OI_VIS"].data["MJD"]), hdr["MJD-OBS"])
        dic["VIS"]["STA_INDEX"] = hdu["OI_VIS"].data["STA_INDEX"]
    except Exception:
        pass
        # print("WARNING: No OI_VIS table!")

    try:
        dic["VIS2"] = {}
        dic["VIS2"]["VIS2"] = hdu["OI_VIS2"].data["VIS2DATA"]
        dic["VIS2"]["VIS2ERR"] = hdu["OI_VIS2"].data["VIS2ERR"]
        dic["VIS2"]["U"] = hdu["OI_VIS2"].data["UCOORD"]
        dic["VIS2"]["V"] = hdu["OI_VIS2"].data["VCOORD"]
        dic["VIS2"]["TIME"] = hdu["OI_VIS2"].data["MJD"]
        dic["VIS2"]["FLAG"] = hdu["OI_VIS2"].data["FLAG"]
        if dic["VIS2"]["TIME"][0] < 50000:
            print("WARNING: incoherent MJD, picking it up from header")
            print(np.shape(hdu["OI_VIS2"].data["MJD"]))
            dic["VIS2"]["TIME"] = np.full(
                len(hdu["OI_VIS"].data["MJD"]), hdr["MJD-OBS"]
            )
            print(np.shape(np.full(len(hdu["OI_VIS"].data["MJD"]), hdr["MJD-OBS"])))
        dic["VIS2"]["STA_INDEX"] = hdu["OI_VIS2"].data["STA_INDEX"]
    except Exception:
        pass
        # print("WARNING: No OI_VIS2 table!")

    try:
        dic["TF2"] = {}
        dic["TF2"]["TF2"] = hdu["TF2"].data["TF2"]
        dic["TF2"]["TF2ERR"] = hdu["TF2"].data["TF2ERR"]
        # dic['TF2']['U']       = hdu['OI_TF2'].data['UCOORD']
        # dic['TF2']['V']       = hdu['OI_TF2'].data['VCOORD']
        dic["TF2"]["TIME"] = hdu["TF2"].data["MJD"]
        if dic["TF2"]["TIME"][0] < 50000:
            dic["TF2"]["TIME"] = np.full(len(hdu["OI_VIS"].data["MJD"]), hdr["MJD-OBS"])
        dic["TF2"]["STA_INDEX"] = hdu["TF2"].data["STA_INDEX"]
    except Exception:
        pass
        # print("WARNING: No OI_TF2 table!")

    try:
        dic["T3"] = {}
        dic["T3"]["T3AMP"] = hdu["OI_T3"].data["T3AMP"]
        dic["T3"]["T3AMPERR"] = hdu["OI_T3"].data["T3AMPERR"]
        dic["T3"]["CLOS"] = hdu["OI_T3"].data["T3PHI"]
        dic["T3"]["CLOSERR"] = hdu["OI_T3"].data["T3PHIERR"]
        dic["T3"]["U1"] = hdu["OI_T3"].data["U1COORD"]
        dic["T3"]["V1"] = hdu["OI_T3"].data["V1COORD"]
        dic["T3"]["U2"] = hdu["OI_T3"].data["U2COORD"]
        dic["T3"]["V2"] = hdu["OI_T3"].data["V2COORD"]
        dic["T3"]["TIME"] = hdu["OI_T3"].data["MJD"]
        dic["T3"]["FLAG"] = hdu["OI_T3"].data["FLAG"]
        if dic["T3"]["TIME"][0] < 50000:
            dic["T3"]["TIME"] = np.full(len(hdu["OI_VIS"].data["MJD"]), hdr["MJD-OBS"])
        dic["T3"]["STA_INDEX"] = hdu["OI_T3"].data["STA_INDEX"]
    except Exception:
        pass
        # print("WARNING: No OI_T3 table!")

    try:
        dic["FLUX"] = {}
        dic["FLUX"]["FLUX"] = hdu["OI_FLUX"].data["FLUXDATA"]
        dic["FLUX"]["FLUXERR"] = hdu["OI_FLUX"].data["FLUXERR"]
        dic["FLUX"]["TIME"] = hdu["OI_FLUX"].data["MJD"]
        dic["FLUX"]["FLAG"] = hdu["OI_FLUX"].data["FLAG"]
        if dic["FLUX"]["TIME"][0] < 50000:
            dic["FLUX"]["TIME"] = np.full(
                len(hdu["OI_VIS"].data["MJD"]), hdr["MJD-OBS"]
            )
        dic["FLUX"]["STA_INDEX"] = hdu["OI_FLUX"].data["STA_INDEX"]
    except Exception:
        pass
        # print("WARNING: No OI_FLUX table!")

    return dic
