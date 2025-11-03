"""
OIFITS file reader with robust error handling and structured data access.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)


@dataclass
class OIFitsData:
    """
    Structured representation of OIFITS file data.

    All fields are optional and will be None if not found in the file.
    """

    # Required file metadata (no defaults)
    file_path: Path
    header: fits.Header
    wavelength: np.ndarray

    # Observing conditions
    seeing: float = 0.0
    tau0: float = 0.0

    # Target information
    target_name: str = ""
    category: str = "CAL"  # "SCI" or "CAL"
    date_obs: str = ""

    # Instrument configuration
    band: str = ""  # "N", "LM", "H", etc.
    detector_name: str = ""
    dispersion_name: str = ""
    dit: float = field(default=float("nan"))
    bcd1_name: str = ""
    bcd2_name: str = ""

    # Array configuration
    tel_name: np.ndarray = field(default_factory=lambda: np.array([]))
    sta_name: np.ndarray = field(default_factory=lambda: np.array([]))
    sta_index: np.ndarray = field(default_factory=lambda: np.array([]))

    # Interferometric data tables (all optional)
    vis: dict[str, np.ndarray] | None = None
    vis2: dict[str, np.ndarray] | None = None
    t3: dict[str, np.ndarray] | None = None
    flux: dict[str, np.ndarray] | None = None
    tf2: dict[str, np.ndarray] | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to legacy dictionary format for backward compatibility.

        Returns:
            Dictionary with the same structure as the old open_oifits function.
        """
        result = {
            "file": str(self.file_path),
            "HDR": self.header,
            "WLEN": self.wavelength,
            "SEEING": self.seeing,
            "TAU0": self.tau0,
            "TARGET": self.target_name,
            "CATEGORY": self.category,
            "DATEOBS": self.date_obs,
            "BAND": self.band,
            "DISP": self.dispersion_name,
            "DIT": self.dit,
            "BCD1NAME": self.bcd1_name,
            "BCD2NAME": self.bcd2_name,
            "TEL_NAME": self.tel_name if self.tel_name.size > 0 else {},
            "STA_NAME": self.sta_name if self.sta_name.size > 0 else {},
            "STA_INDEX": self.sta_index if self.sta_index.size > 0 else {},
        }

        # Add optional tables only if present
        if self.vis is not None:
            result["VIS"] = self.vis
        if self.vis2 is not None:
            result["VIS2"] = self.vis2
        if self.t3 is not None:
            result["T3"] = self.t3
        if self.flux is not None:
            result["FLUX"] = self.flux
        if self.tf2 is not None:
            result["TF2"] = self.tf2

        return result


class OIFitsReader:
    """
    Reader for OIFITS files with robust error handling.

    Example:
        >>> reader = OIFitsReader(Path("myfile.fits"))
        >>> data = reader.read()
        >>> if
        ...     print(f"Target: {data.target_name}, Band: {data.band}")
    """

    def __init__(self, file_path: Path | str):
        """
        Initialize reader for an OIFITS file.

        Args:
            file_path: Path to the OIFITS file.
        """
        self.file_path = Path(file_path)
        self._hdu: fits.HDUList | None = None

    def _ensure_hdu(self) -> fits.HDUList:
        """
        Return the open HDU list, raising if it is unavailable.
        """
        if self._hdu is None:
            raise RuntimeError(
                "FITS HDU not loaded. Call read() successfully before accessing data."
            )
        return self._hdu

    def read(self) -> OIFitsData | None:
        """
        Read and parse the OIFITS file.

        Returns:
            OIFitsData object or None if file cannot be read.
        """
        try:
            self._hdu = fits.open(self.file_path)
        except OSError as e:
            logger.error(f"Unable to read FITS file: {self.file_path} - {e}")
            return None

        try:
            data = self._parse_all_data()
            return data
        except Exception as e:
            logger.exception(f"Error parsing OIFITS file {self.file_path}: {e}")
            return None
        finally:
            if self._hdu is not None:
                self._hdu.close()

    def _parse_all_data(self) -> OIFitsData:
        """Parse all data from the OIFITS file."""
        hdu = self._ensure_hdu()
        hdr = hdu[0].header

        data = OIFitsData(
            file_path=self.file_path,
            header=hdr,
            wavelength=self._read_wavelength(),
            seeing=self._read_seeing(hdr),
            tau0=self._read_tau0(hdr),
            target_name=self._read_target_name(),
            category=self._read_category(hdr),
            date_obs=self._read_date_obs(hdr),
            band=self._read_band(hdr),
            detector_name=self._read_detector_name(hdr),
            dispersion_name=self._read_dispersion_name(hdr),
            dit=self._read_dit(hdr),
            bcd1_name=self._read_bcd1_name(hdr),
            bcd2_name=self._read_bcd2_name(hdr),
            tel_name=self._read_array_data("TEL_NAME"),
            sta_name=self._read_array_data("STA_NAME"),
            sta_index=self._read_array_data("STA_INDEX"),
            vis=self._read_vis_table(),
            vis2=self._read_vis2_table(),
            t3=self._read_t3_table(),
            flux=self._read_flux_table(),
            tf2=self._read_tf2_table(),
        )

        return data

    def _read_wavelength(self) -> np.ndarray:
        """Extract wavelength array from OI_WAVELENGTH extension."""
        try:
            hdu = self._ensure_hdu()
            table = hdu["OI_WAVELENGTH"]
            table_data = cast("fits.FITS_rec | None", table.data)
            if table_data is None:
                raise KeyError("OI_WAVELENGTH data missing")
            return np.asarray(table_data["EFF_WAVE"])
        except KeyError:
            logger.warning(f"OI_WAVELENGTH table not found in {self.file_path}")
            return np.array([])

    def _read_seeing(self, hdr: fits.Header) -> float:
        """Extract average seeing from header."""
        try:
            start = hdr["HIERARCH ESO ISS AMBI FWHM START"]
            end = hdr["HIERARCH ESO ISS AMBI FWHM END"]
            return (start + end) / 2.0
        except KeyError:
            return 0.0

    def _read_tau0(self, hdr: fits.Header) -> float:
        """Extract average coherence time from header."""
        try:
            start = hdr["HIERARCH ESO ISS AMBI TAU0 START"]
            end = hdr["HIERARCH ESO ISS AMBI TAU0 END"]
            return (start + end) / 2.0
        except KeyError:
            return 0.0

    def _read_target_name(self) -> str:
        """Extract target name with fallback strategy."""
        try:
            hdu = self._ensure_hdu()
            table = cast("fits.FITS_rec | None", hdu["OI_TARGET"].data)
            if table is not None:
                target = table["TARGET"][0]
                if target:
                    return str(target)
        except (KeyError, IndexError):
            pass

        # Fallback to header
        try:
            return self._ensure_hdu()[0].header["HIERARCH ESO OBS TARG NAME"]
        except KeyError:
            logger.warning(f"Target name not found in {self.file_path}")
            return ""

    def _read_category(self, hdr: fits.Header) -> str:
        """Determine if observation is science (SCI) or calibrator (CAL)."""
        try:
            catg = hdr["ESO PRO CATG"]
            return "SCI" if catg == "TARGET_RAW_INT" else "CAL"
        except KeyError:
            logger.debug(f"Target category not found in {self.file_path}, assuming CAL")
            return "CAL"

    def _read_date_obs(self, hdr: fits.Header) -> str:
        """Extract observation date."""
        try:
            return hdr["DATE-OBS"]
        except KeyError:
            return ""

    def _read_detector_name(self, hdr: fits.Header) -> str:
        """Extract detector name."""
        try:
            return hdr["HIERARCH ESO DET CHIP NAME"]
        except KeyError:
            logger.debug(f"Detector name not found in {self.file_path}")
            return ""

    def _read_band(self, hdr: fits.Header) -> str:
        """
        Determine observation band from detector or filter.

        Returns:
            Band identifier: "N", "LM", "H", or empty string.
        """
        det_name = self._read_detector_name(hdr)

        if det_name == "AQUARIUS":
            return "N"
        elif det_name == "HAWAII-2RG":
            return "LM"

        # Check for MIRCX H-band
        try:
            if hdr["FILTER1"] == "H_band":
                logger.info(f"Found MIRCX H-band observation in {self.file_path}")
                return "H"
        except KeyError:
            pass

        return ""

    def _read_dispersion_name(self, hdr: fits.Header) -> str:
        """Extract disperser name based on detector."""
        det_name = self._read_detector_name(hdr)

        try:
            if det_name == "AQUARIUS":
                return hdr["HIERARCH ESO INS DIN NAME"]
            else:
                return hdr["HIERARCH ESO INS DIL NAME"]
        except KeyError:
            logger.debug(f"Dispersion name not found in {self.file_path}")
            return ""

    def _read_dit(self, hdr: fits.Header) -> float:
        """Extract detector integration time (DIT) in seconds."""
        try:
            return hdr["HIERARCH ESO DET SEQ1 DIT"]
        except KeyError:
            logger.debug(f"DIT not found in {self.file_path}")
            return float("nan")

    def _read_bcd1_name(self, hdr: fits.Header) -> str:
        """Extract BCD1 (Beam Combiner Device 1) name."""
        try:
            return hdr["HIERARCH ESO INS BCD1 NAME"]
        except KeyError:
            logger.debug(f"BCD1 name not found in {self.file_path}")
            return ""

    def _read_bcd2_name(self, hdr: fits.Header) -> str:
        """Extract BCD2 (Beam Combiner Device 2) name."""
        try:
            return hdr["HIERARCH ESO INS BCD2 NAME"]
        except KeyError:
            logger.debug(f"BCD2 name not found in {self.file_path}")
            return ""

    def _read_array_data(self, column_name: str) -> np.ndarray:
        """Read data from OI_ARRAY extension."""
        try:
            hdu = self._ensure_hdu()
            array_ext = hdu["OI_ARRAY"]
            table = cast("fits.FITS_rec | None", array_ext.data)
            if table is None:
                raise KeyError("OI_ARRAY data missing")
            return np.asarray(table[column_name])
        except KeyError:
            logger.debug(f"Column '{column_name}' not found in OI_ARRAY table")
            return np.array([])

    def _fix_mjd_if_needed(self, mjd: np.ndarray) -> np.ndarray:
        """
        Fix MJD values if they are incoherent (< 50000).

        Falls back to MJD-OBS from primary header.
        """
        if len(mjd) > 0 and mjd[0] < 50000:
            logger.warning(
                f"Incoherent MJD values detected in {self.file_path}, "
                "using MJD-OBS from header"
            )
            try:
                hdu = self._ensure_hdu()
                mjd_obs = hdu[0].header["MJD-OBS"]
                return np.full(len(mjd), mjd_obs)
            except KeyError:
                logger.error("MJD-OBS not found in header, cannot fix MJD")
        return mjd

    def _read_vis_table(self) -> dict[str, np.ndarray] | None:
        """Read OI_VIS extension (complex visibility)."""
        try:
            hdu = self._ensure_hdu()
            vis_ext = hdu["OI_VIS"]
            table = cast("fits.FITS_rec | None", vis_ext.data)
            if table is None:
                return None
            vis_data = {
                "VISAMP": np.asarray(table["VISAMP"]),
                "VISAMPERR": np.asarray(table["VISAMPERR"]),
                "DPHI": np.asarray(table["VISPHI"]),
                "DPHIERR": np.asarray(table["VISPHIERR"]),
                "FLAG": np.asarray(table["FLAG"]),
                "U": np.asarray(table["UCOORD"]),
                "V": np.asarray(table["VCOORD"]),
                "TIME": self._fix_mjd_if_needed(np.asarray(table["MJD"])),
                "STA_INDEX": np.asarray(table["STA_INDEX"]),
            }

            # Optional correlated flux
            if table.dtype.names and "CFXAMP" in table.dtype.names:
                vis_data["CFLUX"] = np.asarray(table["CFXAMP"])
                vis_data["CFLUXERR"] = np.asarray(table["CFXAMPERR"])

            return vis_data
        except KeyError:
            logger.debug(f"No OI_VIS table in {self.file_path}")
            return None

    def _read_vis2_table(self) -> dict[str, np.ndarray] | None:
        """Read OI_VIS2 extension (squared visibility)."""
        try:
            hdu = self._ensure_hdu()
            vis2_ext = hdu["OI_VIS2"]
            table = cast("fits.FITS_rec | None", vis2_ext.data)
            if table is None:
                return None
            return {
                "VIS2": np.asarray(table["VIS2DATA"]),
                "VIS2ERR": np.asarray(table["VIS2ERR"]),
                "U": np.asarray(table["UCOORD"]),
                "V": np.asarray(table["VCOORD"]),
                "TIME": self._fix_mjd_if_needed(np.asarray(table["MJD"])),
                "FLAG": np.asarray(table["FLAG"]),
                "STA_INDEX": np.asarray(table["STA_INDEX"]),
            }
        except KeyError:
            logger.debug(f"No OI_VIS2 table in {self.file_path}")
            return None

    def _read_t3_table(self) -> dict[str, np.ndarray] | None:
        """Read OI_T3 extension (closure phase and triple amplitude)."""
        try:
            hdu = self._ensure_hdu()
            t3_ext = hdu["OI_T3"]
            table = cast("fits.FITS_rec | None", t3_ext.data)
            if table is None:
                return None
            return {
                "T3AMP": np.asarray(table["T3AMP"]),
                "T3AMPERR": np.asarray(table["T3AMPERR"]),
                "CLOS": np.asarray(table["T3PHI"]),
                "CLOSERR": np.asarray(table["T3PHIERR"]),
                "U1": np.asarray(table["U1COORD"]),
                "V1": np.asarray(table["V1COORD"]),
                "U2": np.asarray(table["U2COORD"]),
                "V2": np.asarray(table["V2COORD"]),
                "TIME": self._fix_mjd_if_needed(np.asarray(table["MJD"])),
                "FLAG": np.asarray(table["FLAG"]),
                "STA_INDEX": np.asarray(table["STA_INDEX"]),
            }
        except KeyError:
            logger.debug(f"No OI_T3 table in {self.file_path}")
            return None

    def _read_flux_table(self) -> dict[str, np.ndarray] | None:
        """Read OI_FLUX extension."""
        try:
            hdu = self._ensure_hdu()
            flux_ext = hdu["OI_FLUX"]
            table = cast("fits.FITS_rec | None", flux_ext.data)
            if table is None:
                return None
            return {
                "FLUX": np.asarray(table["FLUXDATA"]),
                "FLUXERR": np.asarray(table["FLUXERR"]),
                "TIME": self._fix_mjd_if_needed(np.asarray(table["MJD"])),
                "FLAG": np.asarray(table["FLAG"]),
                "STA_INDEX": np.asarray(table["STA_INDEX"]),
            }
        except KeyError:
            logger.debug(f"No OI_FLUX table in {self.file_path}")
            return None

    def _read_tf2_table(self) -> dict[str, np.ndarray] | None:
        """Read TF2 extension (transfer function)."""
        try:
            hdu = self._ensure_hdu()
            tf2_ext = hdu["TF2"]
            table = cast("fits.FITS_rec | None", tf2_ext.data)
            if table is None:
                return None
            return {
                "TF2": np.asarray(table["TF2"]),
                "TF2ERR": np.asarray(table["TF2ERR"]),
                "TIME": self._fix_mjd_if_needed(np.asarray(table["MJD"])),
                "STA_INDEX": np.asarray(table["STA_INDEX"]),
            }
        except KeyError:
            logger.debug(f"No TF2 table in {self.file_path}")
            return None


# Backward compatibility function
def open_oifits(oi_file: str | Path) -> dict[str, Any]:
    """
    Read an OIFITS file and return data as a dictionary.

    This function maintains backward compatibility with the legacy interface.
    For new code, prefer using OIFitsReader directly.

    Args:
        oi_file: Path to the OIFITS file.

    Returns:
        Dictionary containing OIFITS data, or empty dict if file cannot be read.
    """
    reader = OIFitsReader(oi_file)
    result = reader.read()

    if result is None:
        return {}

    return result.to_dict()
