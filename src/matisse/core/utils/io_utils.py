import glob
import json
from collections.abc import Sequence
from pathlib import Path

from matisse.core.lib_auto_pipeline import matisse_type
from matisse.core.utils.log_utils import log


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
