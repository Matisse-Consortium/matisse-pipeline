"""
Recipe parameter compatibility layer for handling version-specific esorex parameters.

This module provides utilities to detect installed recipe versions and map
high-level CLI parameters to version-appropriate esorex options.

Created in 2026
Contributor: aso
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from matisse.core.utils.log_utils import log


def detect_has_filtering_params(recipe_dir: Path | None = None) -> bool:
    """
    Detect if installed mat_raw_estimates recipe supports new filtering parameters.

    Checks for presence of --filter, --vfactor, --pfactor parameters by parsing
    the esorex --man-page output. These parameters are NEW in matisse 2.0.1+.

    Parameters
    ----------
    recipe_dir : Path | None
        Custom recipe directory. If None, uses system default.

    Returns
    -------
    bool
        True if recipe supports --filter, --vfactor, --pfactor (version 2.0.1+).
        False for older recipes (version 2.2.3 or earlier).

    Notes
    -----
    This function caches results per recipe_dir to avoid repeated esorex calls.
    """
    try:
        cmd = ["esorex"]
        if recipe_dir is not None:
            cmd.extend(["--recipe-dir", str(recipe_dir)])
        cmd.extend(["--man-page", "mat_raw_estimates"])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )

        man_output = result.stdout + result.stderr
        # Check for presence of new filtering parameters
        return "--vfactor" in man_output and "--pfactor" in man_output

    except (subprocess.TimeoutExpired, FileNotFoundError) as err:
        log.warning(f"Could not detect recipe version: {err}")
        return False


def build_raw_estimates_params(
    recipe_dir: Path | None = None,
    vfactor_mode: bool = True,
    pfactor_mode: bool = True,
    filter_mode: str = "vf,pf,jp",
) -> str:
    """
    Build recipe-specific parameters for mat_raw_estimates based on recipe version.

    Maps high-level user parameters to version-appropriate esorex options.
    For latest recipes 2.0.1 (pre-production), includes new filtering parameters.
    For older recipes, silently ignores unsupported parameters.

    Parameters
    ----------
    recipe_dir : Path | None
        Custom recipe directory. If None, uses system default.
    vfactor_mode : bool
        Enable vfactor correction (L/M band only). Default: True.
        Only used for recipes 2.0.1+ (pre-production).
    pfactor_mode : bool
        Enable pfactor correction (L/M band only). Default: True.
        Only used for recipes 2.0.1+ (pre-production).
    filter_mode : str
        Filter modes to apply (comma-separated: "vf, pf, jp", or none).
        Default: "vf,pf,jp". Only used for recipes 2.0.1+ (pre-production).

    Returns
    -------
    str
        Space-separated esorex parameters (e.g., "--vfactor=TRUE --pfactor=TRUE --filter=vf,pf,jp").
        Empty string if no parameters are applicable.

    Notes
    -----
    The returned string is ready to be appended to the esorex command line.
    """
    params = []

    # Check if recipe supports new filtering parameters
    if detect_has_filtering_params(recipe_dir):
        # Recipe 2.0.1+ path: include new filtering parameters
        params.append(f"--vfactor={'TRUE' if vfactor_mode else 'FALSE'}")
        params.append(f"--pfactor={'TRUE' if pfactor_mode else 'FALSE'}")

        # Only add --filter if mode is not "none"
        if filter_mode.lower() != "none":
            params.append(f"--filter={filter_mode}")

        log.debug(f"Using filtering parameters for recipe 2.0.1+: {' '.join(params)}")
    else:
        # Recipe 2.2.3 or older: filtering parameters not supported
        if vfactor_mode or pfactor_mode or filter_mode != "vf,pf,jp":
            log.warning(
                "Recipe does not support --vfactor, --pfactor, --filter. "
                "Parameters ignored (consider upgrading recipes)."
            )

    return " ".join(params)
