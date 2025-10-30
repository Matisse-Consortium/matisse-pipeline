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


def test_make_table_builds_filtered_table_and_annotation(mock_fig):
    """
    make_table should add a single Table trace with filtered rows
    and create a matching annotation banner.
    """
    data = {"seeing": 0.8, "tau0": 5.0}
    title = "Quality check"
    color = "#ABCDEF"
    x_annot = 0.42
    y_annot = 0.91

    result = vp.make_table(
        mock_fig,
        data=data,
        title=title,
        color=color,
        row=1,
        col=1,
        x_annot=x_annot,
        y_annot=y_annot,
        keys=["tau0"],  # case-insensitive filtering
    )

    assert result is mock_fig
    mock_fig.add_trace.assert_called_once()
    table_trace = mock_fig.add_trace.call_args.args[0]
    assert isinstance(table_trace, go.Table)

    kwargs = mock_fig.add_trace.call_args.kwargs
    assert kwargs["row"] == 1
    assert kwargs["col"] == 1

    param_values = list(table_trace.cells.values[0])
    value_values = list(table_trace.cells.values[1])
    assert param_values == ["<b>TAU0</b>"]  # uppercase + bold formatting
    assert value_values == [5.0]  # filtered row only
    assert table_trace.header.line.color == color
    assert table_trace.cells.line.color == color

    mock_fig.add_annotation.assert_called_once()
    _, annotation_kwargs = mock_fig.add_annotation.call_args
    assert annotation_kwargs["text"] == f"<b>{title}</b>"
    assert annotation_kwargs["bgcolor"] == color
    assert annotation_kwargs["x"] == x_annot
    assert annotation_kwargs["y"] == y_annot


def test_make_title_adds_annotation(mock_fig):
    """
    make_title must inject a styled annotation using meta information.
    """
    meta = {"target": "Vega", "config": "A0-B2-C1", "date": "2025-01-02"}
    color = "#112233"

    vp.make_title(mock_fig, meta, color=color)

    mock_fig.add_annotation.assert_called_once()
    _, kwargs = mock_fig.add_annotation.call_args
    assert color in kwargs["text"]
    assert "Vega" in kwargs["text"]
    assert "A0-B2-C1" in kwargs["text"]
    assert "2025-01-02" in kwargs["text"]
    assert kwargs["bgcolor"] == "rgba(255,255,255,0.9)"
    assert kwargs["bordercolor"] == color
    assert kwargs["xref"] == "paper"
    assert kwargs["yref"] == "paper"


def test_plot_spectrum_adds_flux_traces(mock_fig):
    """
    plot_spectrum should add photometric bands, then append one Scatter trace
    per station with the expected naming convention.
    """
    wl = np.array([3.0, 3.5, 4.0])
    flux_a = np.array([1.0, 2.0, 3.0])
    flux_b = np.array([0.5, 1.5, 2.5])
    data = {
        "WLEN": wl,
        "STA_INDEX": [1, 2],
        "STA_NAME": ["STA1", "STA2"],
        "TEL_NAME": ["UT1", "UT2"],
        "FLUX": {
            "FLUX": [flux_a, flux_b],
            "STA_INDEX": [1, 2],
        },
    }

    vp.plot_spectrum(mock_fig, data)

    # Three photometric bands + two flux traces appended
    assert hasattr(mock_fig, "_added_traces")
    assert len(mock_fig._added_traces) == 5

    band_traces = mock_fig._added_traces[:3]
    for trace, row, col in band_traces:
        assert isinstance(trace, go.Scatter)
        assert row == 2
        assert col == 1
        y_vals = list(trace.y)
        assert y_vals[0] == y_vals[1] == min(flux_b.min(), flux_a.min())
        assert y_vals[2] == y_vals[3] == max(flux_b.max(), flux_a.max())
        assert y_vals[4] == y_vals[0]

    flux_traces = mock_fig._added_traces[3:]
    assert len(flux_traces) == 2

    names = {trace.name for trace, _, _ in flux_traces}
    assert names == {"UT1-STA1", "UT2-STA2"}

    for trace, row, col in flux_traces:
        assert isinstance(trace, go.Scatter)
        assert row == 2
        assert col == 1
        assert np.allclose(np.array(trace.x), wl * 1e6)

    _, y_kwargs = mock_fig.update_yaxes.call_args
    assert y_kwargs["title"] == "Flux (arbitrary units)"
    assert y_kwargs["row"] == 2
    assert y_kwargs["col"] == 1
    assert y_kwargs["domain"] == [0.58, 0.80]

    _, x_kwargs = mock_fig.update_xaxes.call_args
    assert x_kwargs["title"] == "Wavelength (µm)"
    assert x_kwargs["title_standoff"] == 10
    assert x_kwargs["row"] == 2
    assert x_kwargs["col"] == 1
    assert x_kwargs["range"] == [wl[-1] * 1e6, wl[0] * 1e6]
