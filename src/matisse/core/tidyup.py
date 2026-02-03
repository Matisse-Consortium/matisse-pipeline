import os
from fnmatch import fnmatch
from pathlib import Path
from shutil import copy2

from astropy.io import fits
from rich.progress import Progress

from matisse.core.utils.log_utils import log


def change_oifits_filename(oifits: Path) -> None:
    """
    Rename a MATISSE OIFITS file based on its FITS header metadata.

    This function replicates the behavior of the original `mat_tidyupOiFits.py`
    script, using modern Python (pathlib, logging, exceptions handling).

    Args:
        oifits: Path to the OIFITS file.
    """
    direc = oifits.parent

    try:
        # Read FITS header
        hdu = fits.getheader(oifits)

        # Check if it's a valid MATISSE OIFITS file
        catg = hdu.get("HIERARCH ESO PRO CATG", "")
        if catg not in ("CALIB_RAW_INT", "TARGET_RAW_INT"):
            return

        # Extract header keywords safely
        targ = (
            hdu.get("HIERARCH ESO OBS TARG NAME") or hdu.get("OBJECT", "UNKNOWN")
        ).replace(" ", "")

        # Station configuration
        try:
            stations_config = (
                hdu["HIERARCH ESO ISS CONF STATION1"]
                + hdu["HIERARCH ESO ISS CONF STATION2"]
                + hdu["HIERARCH ESO ISS CONF STATION3"]
                + hdu["HIERARCH ESO ISS CONF STATION4"]
            )
        except Exception:
            stations_config = "noConf"

        chip_type = hdu.get("HIERARCH ESO DET CHIP TYPE", "unknown")

        if chip_type == "IR-LM":
            resol = hdu.get("HIERARCH ESO INS DIL NAME", "noRes")
        elif chip_type == "IR-N":
            resol = hdu.get("HIERARCH ESO INS DIN NAME", "noRes")
        else:
            resol = "noRes"

        # Chopping mode
        chop = hdu.get("HIERARCH ESO ISS CHOP ST", "F")
        chop_mode = "noChop" if chop == "F" else "Chop"

        tpl_start = hdu.get("HIERARCH ESO TPL START", "noDate").replace(":", "")

        bcd1 = hdu.get("HIERARCH ESO INS BCD1 NAME", "noBCD1")
        bcd2 = hdu.get("HIERARCH ESO INS BCD2 NAME", "noBCD2")

        # Build new filename
        new_name = (
            f"{tpl_start}_{targ}_{stations_config}_{chip_type}_"
            f"{resol}_{bcd1}_{bcd2}_{chop_mode}.fits"
        )

        new_path = direc / new_name

        # Rename file on disk
        os.rename(oifits, new_path)
        log.debug(f"Renamed {oifits.name} → {new_path.name}")

    except Exception as e:
        log.warning(f"Could not process {oifits.name}: {e}")
        return


def tidyup_path(input_dir: Path) -> None:
    """
    Replicates legacy behavior:
      - If 'path' is a file: rename it in place (no backup directory).
      - If 'path' is a directory:
          * Create '<cwd>/<basename(path)>_OIFITS' if missing
          * Walk the directory recursively
          * Skip files matching SKIP_PATTERNS
          * For files ending with 'fits', if header CATG is CALIB_RAW_INT or TARGET_RAW_INT:
              - copy to the backup directory
              - then rename there using header metadata
    """
    SKIP_PATTERNS = [
        "TARGET_CAL_0*",
        "OBJ_CORR_FLUX_0*",
        "OI_OPDWVPO_*",
        "PHOT_BEAMS_*",
        "CALIB_CAL_0*",
        "RAW_DPHASE_*",
        "matis_eop*",
        "nrjReal*",
        "DSPtarget*",
        "nrjImag*",
        "fringePeak*",
        "BSreal*",
        "BSimag*",
    ]

    if input_dir.is_file():
        change_oifits_filename(input_dir)
        return

    if not input_dir.is_dir():
        log.warning(f"Path not found: {input_dir}")
        return

    # Legacy creates the backup dir in the CURRENT WORKING DIRECTORY, not inside 'path'
    cwd = Path.cwd()
    backup_dir = cwd / f"{input_dir.resolve().name}_OIFITS"
    if backup_dir.exists():
        log.info(f"{backup_dir} already exists.")
    else:
        log.info(f"Creating directory {backup_dir}")
        backup_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"OIFITS files will be copied into {backup_dir}")

    # Collect recursively all FITS files
    fits_files: list[Path] = [
        p
        for p in input_dir.rglob("*.fits")
        if not any(fnmatch(p.name, pat) for pat in SKIP_PATTERNS)
    ]
    if len(fits_files) == 0:
        log.info(f"No OIFITS/FITS files found in {input_dir}")
        return

    log.info(f"Number of files to treat: {len(fits_files)}")

    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Renaming OIFITS files...", total=len(fits_files)
        )
        for file in fits_files:
            fifil = file.name

            # Copy, then rename inside backup dir — only for the two CATG values
            try:
                hdr = fits.getheader(str(file))
                catg = hdr.get("HIERARCH ESO PRO CATG")
                if catg in ("CALIB_RAW_INT", "TARGET_RAW_INT"):
                    dst = backup_dir / fifil
                    copy2(str(file), str(dst))
                    change_oifits_filename(dst)
            except Exception:
                continue
            progress.advance(task)
