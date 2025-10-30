"""
Global pytest configuration for MATISSE pipeline tests.

This fixture automatically cleans up any temporary 'IterX' directories
(created during CLI or pipeline execution) after each test run, ensuring
that no residual data remains in the working directory.
"""

import shutil
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
    }
    return data


@pytest.fixture
def full_mock_data(base_mock_data):
    """
    Extended mock data for testing plotting and higher-level functions.
    Contains 3 telescopes, 3 baselines, and 1 closure-phase triplet.
    """
    data = base_mock_data.copy()
    n_wl = 10  # number of wavelengths
    n_bl = 3  # number of baselines
    n_cp = 1  # number of closure-phase triplets
    n_tel = 3  # number of telescopes

    data.update(
        {
            "WLEN": np.linspace(3.0, 4.0, n_wl),
            "FLUX": {
                "FLUX": [np.random.rand(n_wl) for _ in range(n_tel)],
                "STA_INDEX": [1, 2, 3],
            },
            "VIS2": {
                "STA_INDEX": np.array([[1, 2], [1, 3], [2, 3]]),
                "VIS2": [np.random.rand(n_wl) for _ in range(n_bl)],
                "VIS2ERR": [np.random.rand(n_wl) * 0.1 for _ in range(n_bl)],
                "FLAG": [np.zeros(n_wl, dtype=bool) for _ in range(n_bl)],
                "U": np.random.rand(n_bl),
                "V": np.random.rand(n_bl),
            },
            "VIS": {
                "DPHI": [np.random.rand(n_wl) * 10 for _ in range(n_bl)],
                "DPHIERR": [np.random.rand(n_wl) for _ in range(n_bl)],
                "FLAG": [np.zeros(n_wl, dtype=bool) for _ in range(n_bl)],
                "VISAMP": [np.random.rand(n_wl) for _ in range(n_bl)],
                "VISAMPERR": [np.random.rand(n_wl) * 0.1 for _ in range(n_bl)],
            },
            "T3": {
                "STA_INDEX": np.array([[1, 2, 3]]),
                "CLOS": [np.random.rand(n_wl) * 20 - 10 for _ in range(n_cp)],
                "CLOSERR": [np.random.rand(n_wl) for _ in range(n_cp)],
                "FLAG": [np.zeros(n_wl, dtype=bool) for _ in range(n_cp)],
            },
            # Used by make_static_matisse_plot
            "SEEING": 0.8,
            "TAU0": 5.0,
        }
    )
    return data


@pytest.fixture
def mock_fig2():
    """Fixture for a mocked Plotly Figure object."""
    fig = MagicMock(spec=go.Figure)

    # Simulate subplot retrieval for compatibility with helper functions
    mock_xaxis = Mock()
    mock_xaxis.anchor = "x2"
    mock_yaxis = Mock()
    mock_yaxis.anchor = "y2"
    fig.get_subplot.return_value = (mock_xaxis, mock_yaxis)

    # Simulate layout structure expected in add_photometric_bands
    fig.layout.shapes = []
    fig.layout.annotations = []

    return fig


@pytest.fixture
def mock_fig():
    """Mocked Plotly Figure with realistic 8×3 layout and fixed subplot axes."""
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
