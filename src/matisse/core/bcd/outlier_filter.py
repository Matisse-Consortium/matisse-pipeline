"""Outlier filtering for visibility data."""

import numpy as np

from matisse.types import FloatArray


def filter_outliers_custom(
    data: FloatArray,
    wavelength_range: tuple[int, int],
    threshold: float = 1.5,
) -> tuple[FloatArray, int]:
    """
    Custom sigma-clipping for visibility data per baseline.

    This implements the specific outlier rejection used in MagNum.py:
    - Rejects measurements > threshold * (2nd most extreme deviation)
    - Applied per baseline separately
    - Two iterations of filtering

    Parameters
    ----------
    data : np.ndarray
        Visibility data (nexposures, nbaselines, nwavelengths)
    wavelength_range : tuple of int
        (wlow_idx, whigh_idx) for averaging window
    threshold : float
        Threshold multiplier for outlier detection

    Returns
    -------
    np.ndarray
        Data with outliers set to NaN
    """
    data_filtered = data.copy()
    wlow, whigh = (int(wavelength_range[0]), int(wavelength_range[1]))

    n_outliers = 0
    # Average over wavelength window for outlier detection
    averaged = np.nanmean(data_filtered[:, :, wlow:whigh], axis=-1)
    if averaged.shape[0] <= 2:
        return data_filtered, n_outliers

    # Filter each baseline separately
    for baseline in range(data_filtered.shape[1]):
        baseline_avg = averaged[:, baseline]

        # First iteration
        mean_val = np.nanmean(baseline_avg)
        sorted_idx = np.argsort(np.abs(baseline_avg - mean_val))

        # Check most extreme value
        if np.abs(baseline_avg[sorted_idx[-1]] - mean_val) >= threshold * np.abs(
            baseline_avg[sorted_idx[-2]] - mean_val
        ):
            data_filtered[sorted_idx[-1], baseline] = np.nan
            averaged[sorted_idx[-1], baseline] = np.nan
            n_outliers += 1

        # Second iteration if enough data points
        if averaged.shape[0] > 3:
            mean_val = np.nanmean(averaged[:, baseline])
            sorted_idx = np.argsort(np.abs(averaged[:, baseline] - mean_val))

            if np.abs(
                averaged[sorted_idx[-2], baseline] - mean_val
            ) >= threshold * np.abs(averaged[sorted_idx[-3], baseline] - mean_val):
                data_filtered[sorted_idx[-2], baseline] = np.nan
                n_outliers += 1

    return data_filtered, n_outliers
