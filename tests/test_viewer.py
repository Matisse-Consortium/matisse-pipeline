import shutil
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
from astropy.io import fits

from matisse.viewer import viewer_plotly as vp


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


def test_add_photometric_bands_adds_vrects(mock_fig):
    """
    vp.add_photometric_bands should add 3 rect shapes (L, M, N bands)
    via fig.add_shape using explicit xaxis_id/yaxis_id,
    plus 3 annotations for band labels.
    """
    vp.add_photometric_bands(mock_fig, xaxis_id="x2", yaxis_id="y2")

    # Exactly 3 add_shape calls (L / M / N bands)
    assert mock_fig.add_shape.call_count == 3
    # No Scatter traces added
    assert mock_fig.add_trace.call_count == 0

    # Each call must use explicit xref/yref (no row/col)
    for call in mock_fig.add_shape.call_args_list:
        _, kwargs = call
        assert "row" not in kwargs
        assert "col" not in kwargs
        assert kwargs.get("xref") == "x2"
        assert kwargs.get("yref") == "y2 domain"
        assert kwargs.get("layer") == "below"
        assert kwargs.get("line_width") == 0


def test_add_photometric_bands_correct_x_bounds(mock_fig):
    """
    The x0/x1 limits of each vrect must match the known band boundaries.
    """
    vp.add_photometric_bands(mock_fig, xaxis_id="x2", yaxis_id="y2")

    expected = [(3.2, 4.1), (4.5, 5.0), (8.0, 13.0)]
    actual = [
        (call[1]["x0"], call[1]["x1"]) for call in mock_fig.add_shape.call_args_list
    ]
    assert actual == expected


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
        "chopping",
    ]

    for key in expected_keys:
        assert key in meta, f"Missing key: {key}"

    # Check that values match input
    assert meta["filename"] == base_mock_data["file"]
    assert meta["target"] == base_mock_data["TARGET"]
    assert meta["band"] == base_mock_data["BAND"]
    assert meta["chopping"] == "No"
    assert meta["cat"] == base_mock_data["CATEGORY"]
    assert meta["date"] == base_mock_data["DATEOBS"]
    assert meta["disp"] == base_mock_data["DISP"]
    assert meta["bcd"] == "IN-OUT"
    assert meta["tel"] == base_mock_data["TEL_NAME"]
    assert meta["config"] == "A0-B1-C2"
    assert meta["dit"].value == pytest.approx(base_mock_data["DIT"])
    assert str(meta["dit"].unit) == "s"

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
    assert param_values == ["<b>TAU0 </b>"]  # filtered row only
    value_values = list(table_trace.cells.values[1])
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

    result = vp.plot_spectrum(mock_fig, data)
    assert result is True

    # Two flux traces appended (bands use add_shape, not add_trace)
    assert hasattr(mock_fig, "_added_traces")
    assert len(mock_fig._added_traces) == 2
    assert mock_fig.add_shape.call_count == 3
    # 3 band-label annotations ("NO FLUX DATA" not called)
    assert mock_fig.add_annotation.call_count == 3
    annotation_texts = [
        call[1]["text"] for call in mock_fig.add_annotation.call_args_list
    ]
    assert all("NO FLUX DATA" not in t for t in annotation_texts)

    names = [trace.name for trace, _, _ in mock_fig._added_traces]
    assert names == ["UT1-STA1", "UT2-STA2"]

    # Each trace carries its own flux row against the wavelengths in microns.
    for (trace, row, col), flux in zip(
        mock_fig._added_traces, [flux_a, flux_b], strict=True
    ):
        assert isinstance(trace, go.Scatter)
        assert row == 2
        assert col == 1
        np.testing.assert_allclose(np.asarray(trace.x, dtype=float), wl * 1e6)
        np.testing.assert_allclose(np.asarray(trace.y, dtype=float), flux)


def test_plot_spectrum_returns_false_when_flux_missing(mock_fig):
    """
    If FLUX data is absent, plot_spectrum should skip plotting and return False.
    A Scatter text trace (NO FLUX DATA) is still added at (mid_wl, 0.5); xaxis
    keeps the real wl_range (synced with visibility plots); yaxis is fixed [0,1]
    so the text is always vertically centred.
    """
    wl = np.array([3.0, 3.5, 4.0])
    data = {
        "WLEN": wl,
        "STA_INDEX": [1, 2],
        "STA_NAME": ["STA1", "STA2"],
        "TEL_NAME": ["UT1", "UT2"],
    }

    result = vp.plot_spectrum(mock_fig, data)

    assert result is False

    mock_fig.add_trace.assert_called_once()
    trace_args = mock_fig.add_trace.call_args
    trace = trace_args.args[0]
    assert isinstance(trace, go.Scatter)
    assert trace.mode == "text"
    assert "NO FLUX DATA" in trace.text[0]
    # x = mid-wavelength (keeps xaxis2 synced), y = 0.5 (vertically centred)
    expected_mid = float((wl.min() + wl.max()) / 2) * 1e6
    assert trace.x[0] == pytest.approx(expected_mid)
    assert trace.y[0] == 0.5
    # xaxis range = wl_range; yaxis range = [0, 1]
    mock_fig.update_xaxes.assert_called_once()
    mock_fig.update_yaxes.assert_called_once()
    _, xkw = mock_fig.update_xaxes.call_args
    _, ykw = mock_fig.update_yaxes.call_args
    assert xkw["range"] == pytest.approx([wl.min() * 1e6, wl.max() * 1e6])
    assert ykw["range"] == [0, 1]


def test_plot_spectrum_n_band_no_flux_uses_flux_ylabel(mock_fig):
    """
    When band is N and flux is absent, the spectrum ylabel should be
    'Flux (arbitrary units)'; xaxis keeps wl_range (not [0,1]); yaxis [0,1].
    """
    wl = np.array([8.0, 10.0, 13.0])
    data = {
        "BAND": "N",
        "WLEN": wl,
        "STA_INDEX": [1, 2],
        "STA_NAME": ["STA1", "STA2"],
        "TEL_NAME": ["UT1", "UT2"],
    }

    result = vp.plot_spectrum(mock_fig, data)

    assert result is False
    mock_fig.update_xaxes.assert_called_once()
    mock_fig.update_yaxes.assert_called_once()
    _, xkw = mock_fig.update_xaxes.call_args
    _, ykw = mock_fig.update_yaxes.call_args
    assert xkw["range"] == pytest.approx([wl.min() * 1e6, wl.max() * 1e6])
    assert ykw["range"] == [0, 1]
    assert ykw["title"] == "Flux (arbitrary units)"


def test_make_vltiplot_mini_returns_lengths_and_layout(mock_fig):
    """
    make_vltiplot_mini should draw station markers, optional highlighted telescopes,
    baseline segments, and return their lengths.
    """
    tels = ["A0", "U1"]
    baseline_names = ["A0-U1"]
    baseline_colors = ["#abcdef"]

    lengths = vp.make_vltiplot_mini(
        mock_fig,
        row=2,
        col=2,
        title="VLTI Layout",
        color="#EEEEEE",
        tels=tels,
        baseline_names=baseline_names,
        baseline_colors=baseline_colors,
    )

    assert "A0-U1" in lengths
    assert lengths["A0-U1"] > 0

    # Ensure the dedicated baseline trace has been added
    matching = [
        trace
        for trace, _, _ in getattr(mock_fig, "_added_traces", [])
        if trace.name == "A0-U1" and trace.mode == "lines"
    ]
    assert matching, "Expected a line trace for baseline A0-U1"

    mock_fig.add_annotation.assert_called_once()
    args, kwargs = mock_fig.add_annotation.call_args
    assert kwargs["text"] == "<b>VLTI LAYOUT</b>"


def test_make_vltiplot_mini_no_tels(mock_fig):
    """
    Test the case with no tels to highlight.
    """
    baseline_names = ["A0-U1"]
    baseline_colors = ["#abcdef"]

    lengths = vp.make_vltiplot_mini(
        mock_fig,
        row=2,
        col=2,
        title="VLTI Layout",
        color="#EEEEEE",
        baseline_names=baseline_names,
        baseline_colors=baseline_colors,
    )

    assert len(lengths) == 1

    mock_fig.add_annotation.assert_called_once()
    args, kwargs = mock_fig.add_annotation.call_args
    assert kwargs["text"] == "<b>VLTI LAYOUT</b>"


def test_make_uvplot_adds_symmetrical_points(mock_fig):
    """
    make_uvplot should add +uv and -uv markers and configure axes symmetrically.
    """
    u_coords = [10.0, -12.0]
    v_coords = [5.0, 8.0]
    names = ["A0-G0", "A0-U1"]
    colors = ["#111111", "#222222"]
    lengths = {"A0-G0": 20.0, "A0-U1": 30.0}

    vp.make_uvplot(
        mock_fig,
        u_coords=u_coords,
        v_coords=v_coords,
        array_type="UT",
        baseline_names=names,
        baseline_colors=colors,
        bl_lengths=lengths,
    )

    assert hasattr(mock_fig, "_added_traces")
    assert len(mock_fig._added_traces) == 4  # symmetric points

    first_trace, row, col = mock_fig._added_traces[0]
    assert isinstance(first_trace, go.Scatter)
    assert first_trace.legendgroup == "uv"
    assert first_trace.showlegend is True
    assert row == 2 and col == 3
    assert first_trace.marker.color == "#111111"
    assert first_trace.name == "A0-G0 (20.0 m)"

    # (u, v) then its symmetric (-u, -v) counterpart, baseline after baseline.
    expected_points = [(10.0, 5.0), (-10.0, -5.0), (-12.0, 8.0), (12.0, -8.0)]
    for (trace, _, _), (x_val, y_val) in zip(
        mock_fig._added_traces, expected_points, strict=True
    ):
        np.testing.assert_allclose(np.asarray(trace.x, dtype=float), [x_val])
        np.testing.assert_allclose(np.asarray(trace.y, dtype=float), [y_val])

    # Check axes configuration
    _, kwargs_x = mock_fig.update_xaxes.call_args
    assert kwargs_x["title"] == "U (m)"
    assert kwargs_x["domain"] == [0.75, 1.0]
    y_kwargs = next(
        kwargs
        for _, kwargs in mock_fig.update_yaxes.call_args_list
        if "title" in kwargs
    )
    assert y_kwargs["title"] == "V (m)"
    assert y_kwargs["domain"] == [0.52, 0.8]


def test_plot_obs_groups_single_group_with_errors(mock_fig):
    """
    plot_obs_groups should handle one exposure group and attach error bars when requested.
    """
    wl = np.array([3.0, 3.5, 4.0])
    vis2 = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
    vis2_err = np.array([[0.01, 0.02, 0.03], [0.02, 0.02, 0.02]])
    flags = np.array([[False, True, False], [False, False, False]])
    data = {
        "WLEN": wl,
        "VIS2": {
            "VIS2": vis2,
            "VIS2ERR": vis2_err,
            "FLAG": flags,
        },
    }
    baseline_names = ["A-B", "C-D"]
    colors = ["#FF0000", "#00FF00"]

    vp.plot_obs_groups(
        mock_fig,
        data,
        baseline_names=baseline_names,
        baseline_colors=colors,
        show_errors=True,
        obs_range=[0, 1],
    )

    traces = getattr(mock_fig, "_added_traces", [])
    assert len(traces) == len(baseline_names)
    for idx, ((trace, row, col), color) in enumerate(zip(traces, colors, strict=True)):
        valid = ~flags[idx]
        assert isinstance(trace, go.Scatter)
        assert trace.name == baseline_names[idx]
        assert trace.line.color == color
        assert row == 3 + idx
        assert col == 1
        assert len(trace.y) == int(np.sum(valid))
        np.testing.assert_allclose(np.asarray(trace.x, dtype=float), wl[valid] * 1e6)
        np.testing.assert_allclose(np.asarray(trace.y, dtype=float), vis2[idx][valid])
        np.testing.assert_allclose(
            np.asarray(trace.error_y.array, dtype=float), vis2_err[idx][valid]
        )

    # Ensure y-axis last call used provided range
    _, y_kwargs = mock_fig.update_yaxes.call_args
    assert y_kwargs["range"] == [0, 1]


def test_plot_obs_groups_with_vis_amplitude(mock_fig):
    """
    When obs_name='V', the function must use VISAMP data and tag traces with legendgroup 'V'.
    """
    wl = np.array([3.0, 3.5, 4.0])
    vis_amp = np.array([[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]])
    vis_amp_err = np.array([[0.05, 0.05, 0.04], [0.02, 0.02, 0.03]])
    flags = np.array([[False, False, True], [False, True, False]])
    data = {
        "WLEN": wl,
        "VIS": {
            "VISAMP": vis_amp,
            "VISAMPERR": vis_amp_err,
            "FLAG": flags,
        },
    }
    baseline_names = ["A-B", "B-C"]
    colors = ["#AA0000", "#00AA00"]

    vp.plot_obs_groups(
        mock_fig,
        data,
        baseline_names=baseline_names,
        baseline_colors=colors,
        show_errors=True,
        obs_name="V",
        obs_range=[0, 1],
    )

    traces = getattr(mock_fig, "_added_traces", [])
    assert len(traces) == len(baseline_names)
    for idx, ((trace, row, col), color) in enumerate(zip(traces, colors, strict=True)):
        valid = ~flags[idx]
        assert trace.legendgroup == "V"
        assert trace.name == baseline_names[idx]
        assert trace.line.color == color
        assert row == 3 + idx and col == 1
        assert len(trace.y) == int(np.sum(valid))
        np.testing.assert_allclose(np.asarray(trace.x, dtype=float), wl[valid] * 1e6)
        np.testing.assert_allclose(
            np.asarray(trace.y, dtype=float), vis_amp[idx][valid]
        )
        np.testing.assert_allclose(
            np.asarray(trace.error_y.array, dtype=float), vis_amp_err[idx][valid]
        )


def test_plot_closure_groups_multiple_groups(mock_fig):
    """
    plot_closure_groups should iterate across triplets and exposure groups with fading opacity.
    """
    wl = np.array([3.0, 3.5, 4.0])
    n_triplets = 2
    n_groups = 4
    n_series = n_triplets * n_groups
    base = np.linspace(-30, 30, wl.size)
    clos = np.stack([base + i for i in range(n_series)], axis=0)
    clos_err = np.full_like(clos, 0.5)
    flags = np.zeros_like(clos, dtype=bool)

    data = {"WLEN": wl, "T3": {"CLOS": clos, "CLOSERR": clos_err, "FLAG": flags}}
    triplet_names = ["ABC", "BCD"]
    triplet_colors = ["#FFAA00", "#AA00FF"]

    vp.plot_closure_groups(
        mock_fig,
        data,
        triplet_names=triplet_names,
        triplet_colors=triplet_colors,
        show_errors=True,
        obs_range=[-90, 90],
    )

    traces = getattr(mock_fig, "_added_traces", [])
    assert len(traces) == n_series
    # First two traces correspond to first triplet, first group, etc.
    first_trace, row, col = traces[0]
    assert row == 4
    assert col == 3
    assert isinstance(first_trace, go.Scatter)
    assert first_trace.line.color == triplet_colors[0]
    assert first_trace.opacity == 1.0
    assert first_trace.error_y is not None

    last_trace, _, _ = traces[-1]
    assert last_trace.opacity == 0.1  # final alpha level for 4th group

    # Triplet-major ordering: traces[t * n_groups + g] holds clos[g * n_triplets + t].
    for t_idx in range(n_triplets):
        for g_idx in range(n_groups):
            trace, trace_row, trace_col = traces[t_idx * n_groups + g_idx]
            assert trace.name == triplet_names[t_idx]
            assert trace_row == 4 + t_idx and trace_col == 3
            np.testing.assert_allclose(np.asarray(trace.x, dtype=float), wl * 1e6)
            np.testing.assert_allclose(
                np.asarray(trace.y, dtype=float), clos[g_idx * n_triplets + t_idx]
            )

    _, y_kwargs = mock_fig.update_yaxes.call_args
    assert y_kwargs["range"] == [-90, 90]


def test_make_static_matisse_plot_constructs_full_figure(full_mock_data):
    """
    make_static_matisse_plot should integrate all components into a single Plotly figure.
    """
    baseline_names = vp.build_blname_list(full_mock_data)
    cp_names = vp.build_cpname_list(full_mock_data)
    assert len(baseline_names) == 6
    assert len(cp_names) == 4

    fig = vp.make_static_matisse_plot(full_mock_data)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert any(
        annot.text and "Meta Information" in annot.text
        for annot in fig.layout.annotations
    )

    def _axis_domain(axis_id: str):
        if not axis_id or axis_id == "x":
            layout_key = "xaxis"
        else:
            layout_key = f"xaxis{axis_id[1:]}"
        axis = getattr(fig.layout, layout_key)
        return axis.domain

    v2_traces = [tr for tr in fig.data if getattr(tr, "legendgroup", "") == "V2"]
    assert len(v2_traces) == 6
    for tr in v2_traces:
        dom = _axis_domain(tr.xaxis)
        assert dom[0] < 0.35  # first column

    dphi_traces = [tr for tr in fig.data if getattr(tr, "legendgroup", "") == "dphi"]
    assert len(dphi_traces) == 6
    for tr in dphi_traces:
        dom = _axis_domain(tr.xaxis)
        assert 0.33 < dom[0] < 0.68  # middle column

    cphase_traces = [
        tr for tr in fig.data if getattr(tr, "legendgroup", "") == "CPHASE"
    ]
    assert len(cphase_traces) == 4
    for tr in cphase_traces:
        dom = _axis_domain(tr.xaxis)
        assert dom[0] > 0.66  # last column


def test_make_static_matisse_plot_mix_color_false_defaults(full_mock_data):
    """
    With mix_color=False (default), closure traces must use the static palette.
    """
    fig = vp.make_static_matisse_plot(full_mock_data, mix_color=False)
    cphase_traces = [
        tr for tr in fig.data if getattr(tr, "legendgroup", "") == "CPHASE"
    ]
    expected_colors = ["#FAA050", "#E0C9A6", "#BE7C7E", "#26AEB8"]
    assert len(cphase_traces) == len(expected_colors)
    assert [trace.line.color for trace in cphase_traces] == expected_colors


def test_make_static_matisse_plot_mix_color_true(full_mock_data):
    """
    With mix_color=True, closure traces use blended baseline colors.
    """
    fig = vp.make_static_matisse_plot(full_mock_data, mix_color=True)
    cphase_traces = [
        tr for tr in fig.data if getattr(tr, "legendgroup", "") == "CPHASE"
    ]
    assert len(cphase_traces) == len(full_mock_data["T3"]["STA_INDEX"])
    static_colors = {"#FAA050", "#E0C9A6", "#BE7C7E", "#26AEB8"}
    for trace in cphase_traces:
        assert trace.line.color not in static_colors


def test_show_plot_writes_html(tmp_path):
    """
    show_plot should produce an HTML file when using the real write_html helper.
    """
    fig = go.Figure()
    outfile = tmp_path / "viewer.html"

    result = vp.show_plot(fig, filename=str(outfile), auto_open=False)

    assert result is fig
    assert outfile.exists()
    content = outfile.read_text(encoding="utf-8")
    assert content.startswith("<html>")
    assert "plotly" in content.lower()


# ---------- _extract_bcd_from_filename ----------


@pytest.mark.parametrize(
    "stem, expected",
    [
        ("fake_calib_IR-LM_LOW_IN_IN_noChop_noBCD", "IN_IN"),
        ("fake_calib_IR-LM_LOW_OUT_OUT_noChop_noBCD", "OUT_OUT"),
        ("fake_calib_IR-LM_LOW_IN_OUT_noChop_noBCD", "IN_OUT"),
        ("fake_calib_IR-LM_LOW_OUT_IN_noChop_noBCD", "OUT_IN"),
        # Edge: multiple matches → last one wins
        ("fake_IN_OUT_calib_IR-LM_LOW_OUT_OUT_noChop_noBCD", "OUT_OUT"),
        # No BCD pattern at all
        ("some_random_file", None),
        # Case insensitivity
        ("fake_in_in_noBCD", "IN_IN"),
    ],
)
def test_extract_bcd_from_filename(stem: str, expected: str | None):
    assert vp._extract_bcd_from_filename(stem) == expected


# ---------------------------------------------------------------------------
# find_siblings_all_bands — directory scanning edge cases
# ---------------------------------------------------------------------------


@pytest.fixture
def sibling_dir(viewer_dir, tmp_path):
    """A writable copy of the viewer fixture directory (LM IN_IN, LM IN_OUT, N IN_IN)."""
    target = tmp_path / "obs"
    shutil.copytree(viewer_dir, target)
    return target


def _lm_in_in(directory: Path) -> Path:
    """Return the LM IN_IN reference file inside a sibling directory."""
    return directory / "fake1_IR-LM_LOW_IN_IN_noChop.fits"


def _copy_with(source: Path, destination: Path, **cards) -> Path:
    """Copy a FITS file and overwrite or delete primary-header keywords."""
    shutil.copy(source, destination)
    with fits.open(destination, mode="update") as hdul:
        for key, value in cards.items():
            if value is None:
                del hdul[0].header[key]
            else:
                hdul[0].header[key] = value
    return destination


def test_find_siblings_groups_fixture_files_by_band(sibling_dir):
    """The three reference files are grouped into LM (two BCDs) and N (one)."""
    siblings = vp.find_siblings_all_bands(_lm_in_in(sibling_dir))

    assert sorted(siblings) == ["LM", "N"]
    assert sorted(siblings["LM"]) == ["IN_IN", "IN_OUT"]
    assert list(siblings["N"]) == ["IN_IN"]


def test_find_siblings_ignores_appledouble_sidecars(sibling_dir):
    """macOS '._' sidecar files are skipped entirely."""
    (sibling_dir / "._fake1_IR-LM_LOW_IN_IN_noChop.fits").write_text("sidecar")

    siblings = vp.find_siblings_all_bands(_lm_in_in(sibling_dir))

    assert sorted(siblings["LM"]) == ["IN_IN", "IN_OUT"]


def test_find_siblings_skips_unreadable_files(sibling_dir):
    """A file that is not valid FITS is skipped instead of aborting the scan."""
    (sibling_dir / "broken.fits").write_text("definitely not FITS")

    siblings = vp.find_siblings_all_bands(_lm_in_in(sibling_dir))

    assert sorted(siblings["LM"]) == ["IN_IN", "IN_OUT"]


def test_find_siblings_skips_other_templates(sibling_dir):
    """A file from a different observing template is not treated as a sibling."""
    _copy_with(
        _lm_in_in(sibling_dir),
        sibling_dir / "other_IR-LM_LOW_OUT_OUT_noChop.fits",
        **{"HIERARCH ESO TPL START": "1999-01-01T00:00:00"},
    )

    siblings = vp.find_siblings_all_bands(_lm_in_in(sibling_dir))

    assert sorted(siblings["LM"]) == ["IN_IN", "IN_OUT"]


def test_find_siblings_skips_unknown_detectors(sibling_dir):
    """A file whose detector maps to no band is dropped."""
    _copy_with(
        _lm_in_in(sibling_dir),
        sibling_dir / "weird_IR-LM_LOW_OUT_OUT_noChop.fits",
        **{"HIERARCH ESO DET CHIP NAME": "SOME-OTHER-CHIP"},
    )

    siblings = vp.find_siblings_all_bands(_lm_in_in(sibling_dir))

    assert sorted(siblings["LM"]) == ["IN_IN", "IN_OUT"]


def test_find_siblings_recovers_bcd_of_nobcd_files_from_filename(sibling_dir):
    """A _noBCD product keeps its original BCD key, taken from the filename."""
    _copy_with(
        _lm_in_in(sibling_dir),
        sibling_dir / "fake1_IR-LM_LOW_OUT_OUT_noChop_noBCD.fits",
        **{
            "HIERARCH ESO INS BCD1 NAME": "OUT",
            "HIERARCH ESO INS BCD2 NAME": "OUT",
        },
    )

    siblings = vp.find_siblings_all_bands(_lm_in_in(sibling_dir))

    assert "OUT_OUT" in siblings["LM"]
    assert siblings["LM"]["OUT_OUT"].name.endswith("_noBCD.fits")


def test_find_siblings_falls_back_to_stem_without_bcd_keywords(sibling_dir):
    """A file with no BCD metadata is keyed by its filename stem."""
    calibrated = _copy_with(
        _lm_in_in(sibling_dir),
        sibling_dir / "calibrated_product.fits",
        **{
            "HIERARCH ESO INS BCD1 NAME": None,
            "HIERARCH ESO INS BCD2 NAME": None,
        },
    )

    siblings = vp.find_siblings_all_bands(_lm_in_in(sibling_dir))

    assert siblings["LM"]["calibrated_product"] == calibrated


def test_find_siblings_exposes_chop_tag_for_duplicate_bcd(sibling_dir):
    """Two files sharing a BCD mode are disambiguated by their chopping status."""
    _copy_with(
        _lm_in_in(sibling_dir),
        sibling_dir / "fake1_IR-LM_LOW_IN_IN_Chop.fits",
        **{"HIERARCH ESO ISS CHOP ST": "T"},
    )

    siblings = vp.find_siblings_all_bands(_lm_in_in(sibling_dir))

    assert "IN_IN_NOCHOP" in siblings["LM"]
    assert "IN_IN_CHOP" in siblings["LM"]


def test_find_siblings_numbers_remaining_duplicates(sibling_dir):
    """Files identical in BCD and chopping get a numeric suffix."""
    _copy_with(
        _lm_in_in(sibling_dir), sibling_dir / "fake1_IR-LM_LOW_IN_IN_noChop_bis.fits"
    )

    siblings = vp.find_siblings_all_bands(_lm_in_in(sibling_dir))

    assert "IN_IN_NOCHOP" in siblings["LM"]
    assert "IN_IN_NOCHOP_2" in siblings["LM"]


def test_find_siblings_without_readable_reference_scans_every_file(sibling_dir):
    """An unreadable reference file disables template filtering and keeps all files."""
    broken = sibling_dir / "broken.fits"
    broken.write_text("definitely not FITS")

    siblings = vp.find_siblings_all_bands(broken)

    assert sorted(siblings["LM"]) == ["IN_IN", "IN_OUT"]
    assert list(siblings["N"]) == ["IN_IN"]


# ---------------------------------------------------------------------------
# make_multi_bcd_plot — fallbacks
# ---------------------------------------------------------------------------


def _static_trace_count(path: Path) -> int:
    """Number of traces in the single-dataset figure built from one file."""
    from matisse.core.utils.oifits_reader import open_oifits

    return len(vp.make_static_matisse_plot(open_oifits(str(path))).data)


def test_make_multi_bcd_plot_falls_back_to_static_for_a_lone_file(viewer_dir, tmp_path):
    """With a single discoverable file the plain single-dataset figure is returned."""
    lonely = tmp_path / "fake1_IR-LM_LOW_IN_IN_noChop.fits"
    shutil.copy(viewer_dir / "fake1_IR-LM_LOW_IN_IN_noChop.fits", lonely)

    fig = vp.make_multi_bcd_plot(lonely)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == _static_trace_count(lonely)


def test_make_multi_bcd_plot_handles_unreadable_reference_file(sibling_dir):
    """An unreadable reference file still loads every sibling into the figure."""
    broken = sibling_dir / "broken.fits"
    broken.write_text("definitely not FITS")

    fig = vp.make_multi_bcd_plot(broken)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > _static_trace_count(_lm_in_in(sibling_dir))


def test_make_multi_bcd_plot_recovers_chop_suffixed_primary(sibling_dir):
    """When chopping suffixes the keys, the reference file is still resolved."""
    _copy_with(
        _lm_in_in(sibling_dir),
        sibling_dir / "fake1_IR-LM_LOW_IN_IN_Chop.fits",
        **{"HIERARCH ESO ISS CHOP ST": "T"},
    )

    fig = vp.make_multi_bcd_plot(_lm_in_in(sibling_dir))

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > _static_trace_count(_lm_in_in(sibling_dir))


# ---------------------------------------------------------------------------
# plot_spectrum — missing / degenerate flux blocks
# ---------------------------------------------------------------------------


def test_plot_spectrum_annotates_when_flux_block_is_absent(mock_fig, full_mock_data):
    """A dataset without a FLUX block gets a 'NO FLUX DATA' marker instead."""
    data = dict(full_mock_data)
    del data["FLUX"]

    plot_spectrum_result = vp.plot_spectrum(mock_fig, data)

    texts = [
        list(t.text) for t, _, _ in mock_fig._added_traces if getattr(t, "text", None)
    ]
    assert ["<b>NO FLUX DATA</b>"] in texts
    assert plot_spectrum_result is False


def test_plot_spectrum_annotates_when_flux_block_is_not_a_mapping(
    mock_fig, full_mock_data
):
    """A FLUX block missing its sub-keys also degrades to the marker."""
    data = dict(full_mock_data)
    data["FLUX"] = {"SOMETHING_ELSE": []}

    plot_spectrum_result = vp.plot_spectrum(mock_fig, data)

    texts = [
        list(t.text) for t, _, _ in mock_fig._added_traces if getattr(t, "text", None)
    ]
    assert ["<b>NO FLUX DATA</b>"] in texts
    assert plot_spectrum_result is False


def test_plot_spectrum_annotates_when_flux_table_is_empty(mock_fig, full_mock_data):
    """An empty flux table produces the marker rather than an empty panel."""
    data = dict(full_mock_data)
    data["FLUX"] = {"FLUX": [], "STA_INDEX": [], "FLAG": []}

    plot_spectrum_result = vp.plot_spectrum(mock_fig, data)

    texts = [
        list(t.text) for t, _, _ in mock_fig._added_traces if getattr(t, "text", None)
    ]
    assert ["<b>NO FLUX DATA</b>"] in texts
    assert plot_spectrum_result is False


def test_plot_spectrum_marker_falls_back_when_wavelengths_are_unusable(
    mock_fig, full_mock_data
):
    """An unusable wavelength array still yields a centred marker."""
    data = dict(full_mock_data)
    del data["FLUX"]
    data["WLEN"] = "not-an-array"

    assert vp.plot_spectrum(mock_fig, data) is False

    marker = next(
        t
        for t, _, _ in mock_fig._added_traces
        if list(getattr(t, "text", None) or []) == ["<b>NO FLUX DATA</b>"]
    )
    assert list(marker.x) == [0.5]


def test_plot_spectrum_cycles_colours_beyond_four_telescopes(mock_fig, full_mock_data):
    """More than four photometric channels reuse the telescope colour cycle."""
    data = dict(full_mock_data)
    flux = full_mock_data["FLUX"]
    data["STA_INDEX"] = list(full_mock_data["STA_INDEX"]) * 2
    data["STA_NAME"] = list(full_mock_data["STA_NAME"]) * 2
    data["TEL_NAME"] = list(full_mock_data["TEL_NAME"]) * 2
    data["FLUX"] = {
        "FLUX": list(flux["FLUX"]) * 2,
        "STA_INDEX": list(flux["STA_INDEX"]) * 2,
        "FLAG": list(flux["FLAG"]) * 2,
    }

    assert vp.plot_spectrum(mock_fig, data) is not False
    assert len(mock_fig._added_traces) >= 8


# ---------------------------------------------------------------------------
# make_static_matisse_plot — correlated-flux datasets (no photometry, no VIS2)
# ---------------------------------------------------------------------------


def _corrflux_data(full_mock_data: dict) -> dict:
    """A correlated-flux dataset: no photometry, no VIS2, uv coordinates on VIS."""
    data = dict(full_mock_data)
    data.pop("FLUX", None)
    data["VIS"] = dict(full_mock_data["VIS"])
    data["VIS"]["U"] = full_mock_data["VIS2"]["U"]
    data["VIS"]["V"] = full_mock_data["VIS2"]["V"]
    data["VIS2"] = None
    return data


def test_make_static_matisse_plot_handles_correlated_flux_data(full_mock_data):
    """A dataset with only correlated flux still produces a complete figure."""
    fig = vp.make_static_matisse_plot(_corrflux_data(full_mock_data))

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_make_static_matisse_plot_handles_flux_block_without_stations(full_mock_data):
    """A FLUX block missing STA_INDEX does not break the station bookkeeping."""
    data = dict(full_mock_data)
    data["FLUX"] = {"FLUX": full_mock_data["FLUX"]["FLUX"]}

    fig = vp.make_static_matisse_plot(data)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_make_static_matisse_plot_keeps_each_baseline_series(full_mock_data):
    """The assembled figure carries each baseline/triplet's own input series."""
    fig = vp.make_static_matisse_plot(full_mock_data)

    n_wl = len(full_mock_data["WLEN"])

    def curves_named(name: str) -> list[np.ndarray]:
        return [
            np.asarray(trace.y, dtype=float)
            for trace in fig.data
            if isinstance(trace, go.Scatter)
            and trace.name == name
            and trace.y is not None
            and len(trace.y) == n_wl
        ]

    for b_idx, bname in enumerate(vp.build_blname_list(full_mock_data)):
        curves = curves_named(bname)
        assert any(
            np.allclose(curve, full_mock_data["VIS2"]["VIS2"][b_idx])
            for curve in curves
        )
        assert any(
            np.allclose(curve, full_mock_data["VIS"]["DPHI"][b_idx]) for curve in curves
        )

    for t_idx, tname in enumerate(vp.build_cpname_list(full_mock_data)):
        assert any(
            np.allclose(curve, full_mock_data["T3"]["CLOS"][t_idx])
            for curve in curves_named(tname)
        )

    for i, flux in enumerate(full_mock_data["FLUX"]["FLUX"]):
        name = f"{full_mock_data['TEL_NAME'][i]}-{full_mock_data['STA_NAME'][i]}"
        assert any(np.allclose(curve, flux) for curve in curves_named(name))
