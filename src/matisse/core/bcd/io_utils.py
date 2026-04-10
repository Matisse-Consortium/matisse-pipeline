from importlib.resources import files
from pathlib import Path

import pandas as pd

_CALIBRATION_DATA = files("matisse.core.bcd") / "master_mn_calibration"


def load_bcd_corrections(
    bcd_mode: str,
    corrections_dir: Path | None = None,
) -> pd.DataFrame:
    """Load BCD polynomial correction coefficients from a CSV file.

    Uses the master calibration data bundled with the package by default.

    Parameters
    ----------
    bcd_mode:
        BCD configuration mode (e.g. ``"IN_IN"``, ``"IN_OUT"``, ``"OUT_IN"``).
    corrections_dir:
        Path to an external directory containing correction CSV files.
        When *None* (default), the data shipped with the package is used.
    """
    filename = f"bcd_{bcd_mode}_poly_coeffs.csv"
    if corrections_dir is None:
        with (_CALIBRATION_DATA / filename).open() as fh:
            return pd.read_csv(fh)
    csv_file = Path(corrections_dir) / filename

    if not csv_file.exists():
        raise FileNotFoundError(f"Correction file not found: {csv_file}")
    return pd.read_csv(csv_file)
