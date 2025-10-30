import numpy as np
import plotly.graph_objects as go

from matisse_pipeline.viewer import viewer_plotly as vp


def test_build_blname_list_minimal(base_mock_data):
    """
    It should return an array of 'STA-STA' strings with one item per baseline.
    The function uses data['STA_INDEX']['VIS2'] and data['STA_NAME'].
    """

    n_telescopes = len(base_mock_data["STA_NAME"])
    expected_n_bl = n_telescopes * (n_telescopes - 1) // 2
    blnames = vp.build_blname_list(base_mock_data)

    # basic shape/type checks
    assert isinstance(blnames, list)
    assert len(blnames) == expected_n_bl

    # each name should contain a dash and valid station labels
    valid_labels = set(map(str, base_mock_data["STA_NAME"]))
    for name in blnames:
        assert "-" in name
        left, right = name.split("-")
        assert left in valid_labels
        assert right in valid_labels
        assert left != right  # no self-baseline


def test_build_cpname_list(base_mock_data):
    """
    Ensure build_cpname_list returns an array of valid 'STA-STA-STA'
    closure-phase triplet names based on the mock OIFITS-like structure.
    """
    result = vp.build_cpname_list(base_mock_data)

    n_telescopes = len(base_mock_data["STA_NAME"])
    expected_n_cp = n_telescopes * (n_telescopes - 1) * (n_telescopes - 2) // 6
    assert len(result) == expected_n_cp
    assert isinstance(result, list)
    assert all(isinstance(x, str) for x in result)

    # each name should contain a dash and valid station labels
    valid_labels = set(map(str, base_mock_data["STA_NAME"]))
    for name in result:
        assert "-" in name
        left, middle, right = name.split("-")
        assert left in valid_labels
        assert middle in valid_labels
        assert right in valid_labels
        assert left != right  # no self-baseline


def test_mix_colors_for_closure_with_baselines():
    """
    Ensure mix_colors_for_closure returns one mixed color per closure name,
    derived from the colors of its associated baselines.
    """
    # Baseline names and colors
    bl_names = np.array(["A0-G1", "G1-K0", "A0-K0"])
    bl_colors = ["#ff0000", "#00ff00", "#0000ff"]  # red, green, blue

    # One closure phase: A0-G1-K0 (formed by those three baselines)
    cp_names = np.array(["A0-G1-K0"])

    result = vp.mix_colors_for_closure(bl_colors, bl_names, cp_names)

    # Must return a list/array of strings (one per closure)
    assert isinstance(result, (list, np.ndarray))
    assert len(result) == len(cp_names)
    assert all(isinstance(c, str) for c in result)

    # Output color should be distinct and valid
    mixed_color = result[0]
    assert mixed_color.startswith("#") or mixed_color.startswith("rgb")
    assert mixed_color not in bl_colors  # blended color should differ

    # Deterministic behavior
    result_repeat = vp.mix_colors_for_closure(bl_colors, bl_names, cp_names)
    assert result == result_repeat


def test_mix_colors_fallback():
    bl_colors = ["#FF0000"]
    baselines = ["A-B"]
    closures = ["C-D-E"]

    result = vp.mix_colors_for_closure(bl_colors, baselines, closures)
    assert result == ["#161313"]  # Fallback color


def test_add_photometric_bands_adds_filled_scatter_traces(mock_fig):
    """
    vp.add_photometric_bands should add 3 filled Scatter traces
    (L, M, N bands) to the figure and return the same figure object.
    """
    ymin, ymax = 0.0, 1.0
    row, col = 1, 1

    out = vp.add_photometric_bands(mock_fig, ymin, ymax, row, col)

    # Returns the same figure-like object
    assert out is mock_fig

    # Exactly 3 traces added (L/M/N)
    assert mock_fig.add_trace.call_count == 3

    # Inspect the first call for structure and kwargs (row/col)
    first_call = mock_fig.add_trace.call_args_list[0]
    args, kwargs = first_call

    # Positional arg 0 should be a go.Scatter
    assert isinstance(args[0], go.Scatter)

    # The scatter should be a filled polygon
    s = args[0]
    assert s.fill == "toself"
    assert s.mode == "none"
    assert s.opacity == 0.3
    assert s.hoverinfo == "skip"
    assert isinstance(s.fillcolor, str) and s.fillcolor  # hex color present
    assert isinstance(s.name, str)  # e.g., "L-band"

    # Row/col routed to add_trace
    assert kwargs.get("row") == row
    assert kwargs.get("col") == col


def test_add_photometric_bands_respects_y_bounds(mock_fig):
    """
    The y-coordinates of the filled polygons must use the provided ymin/ymax.
    """
    ymin, ymax = -2.5, 3.5
    row, col = 2, 3

    vp.add_photometric_bands(mock_fig, ymin, ymax, row, col)

    # Check that each call uses [ymin, ymin, ymax, ymax, ymin]
    for call in mock_fig.add_trace.call_args_list:
        s = call[0][0]
        y = list(s.y)
        assert y == [ymin, ymin, ymax, ymax, ymin]


def test_get_subplot_axes_returns_expected_tuple(mock_fig):
    """
    Ensure vp.get_subplot_axes calls fig.get_subplot and returns the expected
    (xaxis, yaxis) tuple.
    """
    row, col = 2, 2
    xaxis, yaxis = vp.get_subplot_axes(mock_fig, row, col)

    mock_fig.get_subplot.assert_called_once_with(row, col)
    assert xaxis == "x2"
    assert yaxis == "y2"


def test_create_meta_from_mock_data(base_mock_data):
    """
    Ensure vp.create_meta builds a proper metadata structure
    containing all essential fields from the OIFITS-like mock data.
    """
    meta = vp.create_meta(base_mock_data)

    # Type check
    assert isinstance(meta, (dict, object))

    # Must contain at least these essential fields
    expected_keys = [
        "filename",
        "target",
        "date",
        "disp",
        "dit",
        "band",
        "bcd",
    ]

    for key in expected_keys:
        assert key in meta, f"Missing key: {key}"

    # Check that values match input
    assert meta["filename"] == base_mock_data["file"]
    assert meta["target"] == base_mock_data["TARGET"]
    assert meta["band"] == base_mock_data["BAND"]

    # Ensure all fields are accessible
    for key in meta:
        _ = meta[key]


def test_create_meta_with_incomplete_data():
    """
    If required fields are missing, vp.create_meta should either
    raise a clear exception or fill defaults.
    """
    incomplete = {"TARGET": "Unknown"}
    try:
        result = vp.create_meta(incomplete)
        assert isinstance(result, dict)
    except Exception as e:
        assert isinstance(e, (KeyError, ValueError))
