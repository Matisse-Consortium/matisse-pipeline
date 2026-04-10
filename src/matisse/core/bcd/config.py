"""Configuration for BCD correction computation."""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BCDConfig:
    """Configuration for BCD correction computation."""

    # BCD mode
    bcd_mode: str = "IN_IN"

    # Extension type
    extension: str = "OI_VIS2"

    # Spectral configuration
    band: str = "LM"
    resolution: str = "LOW"
    spectral_binlen: int = 118

    # Wavelength range for BCD average computation (in meters)
    # -> used to check the overall behavior in L band (average+std)
    wavelength_low: float = 3.3e-6
    wavelength_high: float = 3.8e-6

    # Filtering
    correlated_flux: bool = False
    outlier_threshold: float = 1.5
    tau0_min: float | None = None  # Minimum coherence time in ms (None = no filter)

    # Fit magic numbers
    poly_order: int = 1

    # Output
    output_dir: Path = Path(".")

    def __post_init__(self):
        """Set derived parameters and validate."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.extension == "OI_VIS":
            self.vis_column = "visamp"
        elif self.extension == "OI_VIS2":
            self.vis_column = "vis2data"
        else:
            raise ValueError(f"Unknown extension: {self.extension}")

        logger.info(
            f"BCD Configuration: {self.bcd_mode}, {self.band}, {self.resolution}"
        )
        logger.info(f"Output directory: {self.output_dir}")


# Baseline mappings
BCD_BASELINE_MAP: dict[str, list[int]] = {
    "OUT_OUT": [0, 1, 2, 3, 4, 5],
    "OUT_IN": [0, 1, 4, 5, 2, 3],
    "IN_OUT": [0, 1, 3, 2, 5, 4],
    "IN_IN": [0, 1, 5, 4, 3, 2],
}

BCD_CP_MAP: dict[str, list[int]] = {
    "OUT_OUT": [0, 1, 2, 3],
    "OUT_IN": [3, 1, 2, 0],
    "IN_OUT": [0, 2, 1, 3],
    "IN_IN": [3, 2, 1, 0],
}

BASELINE_PAIRS: dict[str, list[list[int]]] = {
    "OUT_IN": [[2, 4], [3, 5]],
    "IN_OUT": [[2, 3], [4, 5]],
    "IN_IN": [[2, 5], [3, 4]],
}

BCD_CP_SIGN_MAP: dict[str, list[int]] = {
    "OUT_OUT": [1, 1, 1, 1],  # OUT-OUT (0)
    "OUT_IN": [1, -1, -1, 1],  # OUT-IN  (1)
    "IN_OUT": [-1, 1, 1, -1],  # IN-OUT  (2)
    "IN_IN": [-1, -1, -1, -1],  # IN-IN   (3)
}

BCD_MODES_TO_CORRECT = ["IN_IN", "IN_OUT", "OUT_IN"]
