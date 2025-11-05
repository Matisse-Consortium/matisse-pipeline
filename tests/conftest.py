"""
Global pytest configuration for MATISSE pipeline tests.

This fixture automatically cleans up any temporary 'IterX' directories
(created during CLI or pipeline execution) after each test run, ensuring
that no residual data remains in the working directory.
"""

import shutil
from itertools import combinations
from pathlib import Path
from unittest.mock import MagicMock, Mock

import numpy as np
import plotly.graph_objects as go
import pytest


@pytest.fixture(autouse=True)
def cleanup_iter_dirs():
    """
    Automatically remove 'Iter1' to 'Iter4' directories after each test.

    This cleanup runs even if the test fails, and ignores errors if
    directories are missing or locked.
    """
    yield  # Run the test first

    for i in range(1, 5):  # Iter1 to Iter4
        for suffix in ("", "_OIFITS"):
            iter_dir = Path(f"Iter{i}{suffix}")
            if iter_dir.exists():
                shutil.rmtree(iter_dir, ignore_errors=True)


@pytest.fixture
def base_mock_data():
    """
    Minimal mock data for testing basic processing functions
    (baseline lists, closure phases, and metadata).
    """
    data = {
        "STA_INDEX": [1, 2, 3],
        "STA_NAME": ["A0", "B1", "C2"],
        "VIS2": {
            "STA_INDEX": np.array([[1, 2], [1, 3], [2, 3]]),
        },
        "T3": {
            "STA_INDEX": np.array([[1, 2, 3]]),
        },
        # Used for metadata creation
        "file": "test.oifits",
        "TARGET": "TestStar",
        "CATEGORY": "SCIENCE",
        "DATEOBS": "2025-01-01T00:00:00",
        "DIT": 0.1,
        "DISP": "HIGH",
        "BAND": "L",
        "BCD1NAME": "IN",
        "BCD2NAME": "OUT",
        "TEL_NAME": ["UT1", "UT2", "UT3"],
        "HDR": {"HIERARCH ESO ISS CHOP ST": "F"},
    }
    return data


@pytest.fixture
def full_mock_data():
    """
    Extended mock data for testing plotting and higher-level functions.
    Contains 4 telescopes, 6 baselines, and 4 closure-phase triplets.
    """
    rng = np.random.default_rng(42)
    sta_index = [1, 2, 3, 4]
    sta_names = ["A0", "B2", "C2", "D0"]
    tel_names = ["AT1", "AT2", "AT3", "AT4"]
    n_tel = len(sta_index)
    n_wl = 12

    wlen = np.linspace(3.0, 4.0, n_wl)
    baseline_pairs = np.array(list(combinations(sta_index, 2)))
    n_baselines = baseline_pairs.shape[0]
    triplet_indices = np.array(list(combinations(sta_index, 3)))
    n_triplets = triplet_indices.shape[0]

    flux_values = [rng.random(n_wl) * (i + 1) for i in range(n_tel)]
    vis2_values = [rng.random(n_wl) for _ in range(n_baselines)]
    vis2_err = [rng.random(n_wl) * 0.05 for _ in range(n_baselines)]
    vis_flags = [np.zeros(n_wl, dtype=bool) for _ in range(n_baselines)]

    visamp_values = [np.clip(v + 0.05, 0, 1) for v in vis2_values]
    visamp_err = [rng.random(n_wl) * 0.05 for _ in range(n_baselines)]
    dphi_values = [rng.random(n_wl) * 30 - 15 for _ in range(n_baselines)]
    dphi_err = [rng.random(n_wl) for _ in range(n_baselines)]

    clos_values = [rng.random(n_wl) * 60 - 30 for _ in range(n_triplets)]
    clos_err = [rng.random(n_wl) * 2 for _ in range(n_triplets)]
    clos_flags = [np.zeros(n_wl, dtype=bool) for _ in range(n_triplets)]

    data = {
        "file": "test_full.oifits",
        "TARGET": "TestTarget",
        "CATEGORY": "SCI",
        "DATEOBS": "2025-01-01T01:23:45",
        "DIT": 0.2,
        "DISP": "MED",
        "BAND": "N",
        "BCD1NAME": "IN",
        "BCD2NAME": "OUT",
        "TEL_NAME": tel_names,
        "STA_INDEX": sta_index,
        "STA_NAME": sta_names,
        "SEEING": 0.6,
        "TAU0": 4.0,
        "WLEN": wlen,
        "HDR": {"HIERARCH ESO ISS CHOP ST": "T"},
        "FLUX": {
            "FLUX": flux_values,
            "STA_INDEX": sta_index,
            "FLAG": [np.zeros(n_wl, dtype=bool) for _ in range(n_tel)],
        },
        "VIS2": {
            "STA_INDEX": baseline_pairs,
            "VIS2": vis2_values,
            "VIS2ERR": vis2_err,
            "FLAG": vis_flags,
            "U": np.linspace(-60, 60, n_baselines),
            "V": np.linspace(-40, 40, n_baselines),
        },
        "VIS": {
            "STA_INDEX": baseline_pairs,
            "VISAMP": visamp_values,
            "VISAMPERR": visamp_err,
            "DPHI": dphi_values,
            "DPHIERR": dphi_err,
            "FLAG": [np.zeros(n_wl, dtype=bool) for _ in range(n_baselines)],
        },
        "T3": {
            "STA_INDEX": triplet_indices,
            "CLOS": clos_values,
            "CLOSERR": clos_err,
            "FLAG": clos_flags,
        },
    }
    return data


@pytest.fixture
def mock_fig():
    """Mocked Plotly Figure with realistic 8x3  layout and fixed subplot axes."""
    fig = MagicMock(spec=go.Figure)
    fig._grid_ref = {"rows": 8, "cols": 3}

    # Validation logic
    def mock_add_trace(trace, row=None, col=None, **kwargs):
        rows, cols = fig._grid_ref["rows"], fig._grid_ref["cols"]
        if not (1 <= row <= rows):
            raise ValueError(f"Invalid row index {row}")
        if not (1 <= col <= cols):
            raise ValueError(f"Invalid col index {col}")
        if not hasattr(fig, "_added_traces"):
            fig._added_traces = []
        fig._added_traces.append((trace, row, col))
        return trace

    fig.add_trace.side_effect = mock_add_trace

    # Create mocks for subplot axes with fixed attributes
    mock_xaxis = Mock()
    mock_xaxis.configure_mock(anchor="x2")
    mock_yaxis = Mock()
    mock_yaxis.configure_mock(anchor="y2")
    fig.get_subplot.return_value = (mock_xaxis, mock_yaxis)

    # Layout placeholders
    fig.layout.shapes = []
    fig.layout.annotations = []

    return fig


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def real_obs_target(data_dir: Path) -> Path:
    """Return the path to the target real observation FITS file."""
    return data_dir / "MATIS_target_raw.fits"


@pytest.fixture(scope="session")
def real_obs_calib(data_dir: Path) -> Path:
    """Return the path to the calibrator real observation FITS file."""
    return data_dir / "MATIS_calib_raw.fits"


@pytest.fixture(scope="session")
def real_obs_sky(data_dir: Path) -> Path:
    """Return the path to the sky real observation FITS file."""
    return data_dir / "MATIS_sky_raw.fits"


@pytest.fixture(scope="session")
def real_oifits(data_dir: Path) -> Path:
    """Return the path to the real reduced observation FITS file."""
    return data_dir / "real_data1.fits"
