import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from astropy import units as u
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

sta_name_list = np.array(
    [
        "A0",
        "A1",
        "B0",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "C0",
        "C1",
        "C2",
        "C3",
        "D0",
        "D1",
        "D2",
        "E0",
        "G0",
        "G1",
        "G2",
        "H0",
        "I1",
        "J1",
        "J2",
        "J3",
        "J4",
        "J5",
        "J6",
        "K0",
        "L0",
        "M0",
        "U1",
        "U2",
        "U3",
        "U4",
    ]
)

sta_pos_list = np.array(
    [
        [-32.001, -48.013, -14.642, -55.812],
        [-32.001, -64.021, -9.434, -70.949],
        [-23.991, -48.019, -7.065, -53.212],
        [-23.991, -64.011, -1.863, -68.334],
        [-23.991, -72.011, 0.739, -75.899],
        [-23.991, -80.029, 3.348, -83.481],
        [-23.991, -88.013, 5.945, -91.030],
        [-23.991, -96.012, 8.547, -98.594],
        [-16.002, -48.013, 0.487, -50.607],
        [-16.002, -64.011, 5.691, -65.735],
        [-16.002, -72.019, 8.296, -73.307],
        [-16.002, -80.010, 10.896, -80.864],
        [0.010, -48.012, 15.628, -45.397],
        [0.010, -80.015, 26.039, -75.660],
        [0.010, -96.012, 31.243, -90.787],
        [16.011, -48.016, 30.760, -40.196],
        [32.017, -48.0172, 45.896, -34.990],
        [32.020, -112.010, 66.716, -95.501],
        [31.995, -24.003, 38.063, -12.289],
        [64.015, -48.007, 76.150, -24.572],
        [72.001, -87.997, 96.711, -59.789],
        [88.016, -71.992, 106.648, -39.444],
        [88.016, -96.005, 114.460, -62.151],
        [88.016, 7.996, 80.628, 36.193],
        [88.016, 23.993, 75.424, 51.320],
        [88.016, 47.987, 67.618, 74.009],
        [88.016, 71.990, 59.810, 96.706],
        [96.002, -48.006, 106.397, -14.165],
        [104.021, -47.998, 113.977, -11.549],
        [112.013, -48.000, 121.535, -8.951],
        [-16.000, -16.000, -9.925, -20.335],
        [24.000, 24.000, 14.887, 30.502],
        [64.0013, 47.9725, 44.915, 66.183],
        [112.000, 8.000, 103.306, 43.999],
    ]
)

TEL_COLORS = ["#A2C8E8", "#F8C8DC", "#FBE8A6", "#C7E5B4"]
baseline_colors = ["#9B59B6", "#7C889B", "#F4D03F", "#FF6F61", "#2ECC71", "#1E90FF"]
cp_colors = ["#FAA050", "#B8AC6D", "#BE7C7E", "#26AEB8"]

# Color palette (ESO-like)
QC_COLORS = {
    "excellent": "rgba(0,200,0,0.6)",  # dark green
    "good": "rgba(144,238,144,0.6)",  # light green
    "avg": "rgba(255,215,0,0.6)",  # gold
    "bad": "rgba(255,99,71,0.6)",  # tomato
    "blank": "rgba(255,255,255,0.8)",  # default
}

# Thresholds per metric (units: seeing["], tau0[ms], airmass[-], wind_speed[m/s])
QC_THRESHOLDS = {
    "seeing": [(0.6, "excellent"), (0.8, "good"), (1.2, "avg"), (float("inf"), "bad")],
    "tau0": [
        (2.0, "bad"),
        (4.0, "avg"),
        (6.0, "good"),
        (float("-inf"), "excellent"),
    ],  # handled in code (reverse)
    "airmass": [(1.2, "excellent"), (1.5, "good"), (2.0, "avg"), (float("inf"), "bad")],
    "wind_speed": [
        (8.0, "excellent"),
        (12.0, "good"),
        (15.0, "avg"),
        (float("inf"), "bad"),
    ],
    "humidity": [(50, "excellent"), (70, "good"), (80, "avg"), (float("inf"), "bad")],
}


def symbol_for_mode(param):
    """Return an emoji or symbol for a given observing mode/flag."""
    symbols = {
        "bcd": "🔁",
        "chopping": "🔄",
        "dit": "🧭",
        "disp": "🔷",
    }
    return symbols.get(param.lower(), "")


def _qc_color(value, metric):
    """Return a background color for a single cell value and metric."""
    if value is None:
        return QC_COLORS["blank"]
    try:
        if hasattr(value, "value"):
            v = float(value.value)
        else:
            v = float(value)
    except Exception:
        return QC_COLORS["blank"]

    # tau0 is 'higher is better'; others listed as increasing->worse (except we encoded each directly)
    if metric == "tau0":
        # reverse walk (higher is better)
        if v >= 6.0:
            return QC_COLORS["excellent"]
        if v >= 4.0:
            return QC_COLORS["good"]
        if v >= 2.0:
            return QC_COLORS["avg"]
        return QC_COLORS["bad"]

    # generic increasing->worse rule using QC_THRESHOLDS bins
    for thr, label in QC_THRESHOLDS[metric]:
        if v <= thr:
            return QC_COLORS[label]
    return QC_COLORS["bad"]


def build_blname_list(data: dict[str, Any]) -> list:
    """
    Build an array of baseline names (e.g., "A0-G1", "K0-J1") from the OIFITS-like data structure.

    Parameters
    ----------
    data : dict
        Dictionary containing interferometric data, including:
        - `STA_INDEX` : list or array of station indices.
        - `STA_NAME`  : list of corresponding station names.
        - `VIS2["STA_INDEX"]` or `VIS["STA_INDEX"]` : 2D array of shape (n_baselines, 2) giving
          the station index pairs for each baseline.

    Returns
    -------
    list
        List of baseline names (e.g., ["A0-G1", "K0-J1", "G1-I1", ...]).
    """
    sta_index = data["STA_INDEX"]
    n_telescopes = len(sta_index)
    n_baseline = n_telescopes * (n_telescopes - 1) // 2

    # Mapping from station index to station name
    ref_station = {sta_index[i]: data["STA_NAME"][i] for i in range(n_telescopes)}

    # Get all pairs of station indices for each baseline
    # Try VIS2 first, fallback to VIS if unavailable (for correlated flux mode)
    if data.get("VIS2") is not None:
        all_sta_index = data["VIS2"]["STA_INDEX"][:n_baseline]
    else:
        all_sta_index = data["VIS"]["STA_INDEX"][:n_baseline]

    # Initialize a fixed-size array of strings
    baseline_names = np.empty(n_baseline, dtype="U16")

    for i, pair in enumerate(all_sta_index):
        bl1, bl2 = pair
        baseline_names[i] = f"{ref_station[bl1]}-{ref_station[bl2]}"

    return list(baseline_names)


def build_cpname_list(data: dict[str, Any]) -> list:
    """
    Build an array of triplet names (e.g., "A0-G1-J1") from the OIFITS-like data structure.

    Parameters
    ----------
    data : dict
        Dictionary containing interferometric data, including:
        - `STA_INDEX` : list or array of station indices.
        - `STA_NAME`  : list of corresponding station names.
        - `T3["STA_INDEX"]` : 2D array of shape (n_triplet, 3) giving
          the station index triplet for each closure phase.

    Returns
    -------
    np.ndarray
        Array of triple names.
    """
    sta_index = data["STA_INDEX"]
    n_telescopes = len(sta_index)
    n_cp = n_telescopes * (n_telescopes - 1) * (n_telescopes - 2) // 6

    # Mapping from station index to station name
    ref_station = {sta_index[i]: data["STA_NAME"][i] for i in range(n_telescopes)}

    # Get all pairs of station indices for each baseline
    all_sta_index = data["T3"]["STA_INDEX"][:n_cp]

    # Initialize a fixed-size array of strings
    triplet_names = np.empty(n_cp, dtype="U16")
    for i, triplet in enumerate(all_sta_index):
        bl1, bl2, bl3 = triplet
        triplet_names[i] = f"{ref_station[bl1]}-{ref_station[bl2]}-{ref_station[bl3]}"

    return list(triplet_names)


def mix_colors_for_closure(baseline_colors, baselines, closures):
    """
    Generate closure colors as the mean RGB value of the baselines that compose them.

    Parameters
    ----------
    baseline_colors : list[str]
        Hex colors of the baselines.
    baselines : list[str]
        Baseline names like ["A0-B1", "B1-C1", ...].
    closures : list[str]
        Closure names like ["A0-B1-C1", "A0-B2-C1", ...].

    Returns
    -------
    dict[str, str]
        Mapping {closure_name: hex_color}.
    """

    # Convert baseline colors to RGB arrays (0–1)
    baseline_rgb = np.array([mcolors.to_rgb(c) for c in baseline_colors])

    # Create a quick lookup: baseline_name -> color vector
    baseline_color_map = dict(zip(baselines, baseline_rgb, strict=True))

    closure_colors = {}
    for closure in closures:
        stations = closure.split("-")

        # Identify baselines involved in this closure (3 choose 2 = 3 baselines)
        combo_names = [
            f"{stations[i]}-{stations[j]}"
            if f"{stations[i]}-{stations[j]}" in baseline_color_map
            else f"{stations[j]}-{stations[i]}"
            for i in range(3)
            for j in range(i + 1, 3)
        ]

        # Gather their colors (only the ones existing)
        colors = np.array(
            [baseline_color_map[b] for b in combo_names if b in baseline_color_map]
        )

        # Compute mean RGB
        if len(colors) > 0:
            mean_rgb = colors.mean(axis=0)
            closure_colors[closure] = mcolors.to_hex(mean_rgb)
        else:
            closure_colors[closure] = "#161313"  # fallback black if none found

    return [closure_colors[x] for x in closure_colors]


def add_photometric_bands(fig, xaxis_id: str = "x2", yaxis_id: str = "y2") -> None:
    """
    Shade the standard near/mid-IR photometric bands on a spectrum subplot
    using ``add_vrect`` (layout shapes, not traces).

    Using shapes avoids inflating ``fig.data`` and prevents interference with
    the multi-BCD trace-visibility toggling.

    Parameters
    ----------
    xaxis_id, yaxis_id : str
        Plotly axis IDs for the target subplot, e.g. ``"x2"`` / ``"y2"``.
        Pass the axis *ID* directly (not the anchor value returned by
        ``fig.get_subplot``) so that Plotly's ``_subplot_not_empty`` guard —
        which crashes on mixed table/xy subplot grids — is bypassed entirely.
    """
    bands = [
        (3.2, 4.1, "#DAE5EB", "L"),
        (4.5, 5.0, "#DAE5EB", "M"),
        (8.0, 13.0, "#DAE5EB", "N"),
    ]

    for x0, x1, color, label in bands:
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=0,
            y1=1,
            fillcolor=color,
            opacity=0.2,
            layer="below",
            line_width=0,
            xref=xaxis_id,
            yref=f"{yaxis_id} domain",
        )
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=1.0,
            xref=xaxis_id,
            yref=f"{yaxis_id} domain",
            text=f"{label}",
            showarrow=False,
            yanchor="top",
            font=dict(size=9, color="#666"),
        )


def get_subplot_axes(fig, row: int, col: int) -> tuple[str, str]:
    """
    Return the correct axis references (xref, yref) for a given subplot in a Plotly figure.
    """
    xaxis, yaxis = fig.get_subplot(row, col)
    return xaxis.anchor, yaxis.anchor


def make_vltiplot_mini(
    fig,
    row: int,
    col: int,
    title: str = "VLTI Layout",
    color: str = "#E8F2FF",
    tels: list[Any] | None = None,
    baseline_names: list[Any] | None = None,
    baseline_colors: list[Any] | None = None,
    add_annotation: bool = True,
):
    """
    Add a minimalist VLTI layout under the colored title band.
    Only points (ATs blue, UTs black).
    """
    if tels is None:
        tels = []

    x = sta_pos_list[:, 2]
    y = sta_pos_list[:, 3]

    ATs = sta_name_list[:-4]
    UTs = sta_name_list[-4:]
    AT_mask = np.isin(sta_name_list, ATs)
    UT_mask = np.isin(sta_name_list, UTs)

    for i in range(len(x[AT_mask])):
        if ATs[i] not in tels:
            fig.add_trace(
                go.Scatter(
                    x=[x[AT_mask][i]],
                    y=[y[AT_mask][i]],
                    mode="markers",
                    name=ATs[i],
                    showlegend=False,
                    textposition="top center",
                    marker=dict(size=5, color="#007BFF", symbol="circle"),
                    textfont=dict(size=6),
                ),
                row=row,
                col=col,
            )

    for i in range(len(x[UT_mask])):
        if UTs[i] not in tels:
            fig.add_trace(
                go.Scatter(
                    x=[x[UT_mask][i]],
                    y=[y[UT_mask][i]],
                    name=UTs[i],
                    mode="markers",
                    marker=dict(size=9, color="black", symbol="square"),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    if len(tels) > 0:
        if len(tels) > 4:
            custom_tel_colors = TEL_COLORS * (len(tels) // 4)
        else:
            custom_tel_colors = TEL_COLORS

        for i in range(len(tels)):
            tel = np.where(sta_name_list == tels[i])
            telx = sta_pos_list[tel, 2]
            tely = sta_pos_list[tel, 3]
            fig.add_trace(
                go.Scatter(
                    x=[telx[0, 0]],
                    y=[tely[0, 0]],
                    mode="markers",
                    zorder=3,
                    showlegend=False,
                    name=f"{tels[i]}",
                    marker=dict(size=11, color=custom_tel_colors[i]),
                    text=f"{tels[i]}",
                    hoverinfo="text",
                ),
                row=row,
                col=col,
            )

    if baseline_names and baseline_colors:
        bl_lenghts = {}
        for bl, bcolor in zip(baseline_names, baseline_colors, strict=True):
            tel1 = bl.split("-")[0]
            tel2 = bl.split("-")[1]
            dash = "solid"
            if bl == "B2-D0":
                dash = "dot"

            zorder = None
            if bl == "B2-D0":
                zorder = 2
            if tel1 in sta_name_list and tel2 in sta_name_list:
                i1 = np.where(sta_name_list == tel1)[0][0]
                i2 = np.where(sta_name_list == tel2)[0][0]
                bl_lenghts[bl] = np.hypot(x[i2] - x[i1], y[i2] - y[i1])
                fig.add_trace(
                    go.Scatter(
                        x=[x[i1], x[i2]],
                        y=[y[i1], y[i2]],
                        name=bl,
                        mode="lines",
                        line=dict(width=4, color=bcolor, dash=dash),
                        showlegend=False,
                        zorder=zorder,
                    ),
                    row=row,
                    col=col,
                )

    # --- Remove grid, ticks, axes
    fig.update_xaxes(
        visible=False,
        row=row,
        col=col,
    )
    fig.update_yaxes(
        visible=False,
        row=row,
        col=col,
    )

    # --- Force square aspect ratio
    fig.update_xaxes(
        scaleanchor=f"y{row}{col}",
        scaleratio=1,
        constrain="domain",
        domain=[0.42, 0.58],  # largeur de l’encart
        visible=False,
        row=row,
        col=col,
    )
    fig.update_yaxes(
        scaleanchor=f"x{row}{col}",
        scaleratio=1,
        constrain="domain",
        domain=np.array([0.64, 0.92]) - 0.05,  # hauteur et position haute
        visible=False,
        row=row,
        col=col,
    )
    # --- Add title band above
    if add_annotation:
        fig.add_annotation(
            text=f"<b>{title.upper()}</b>",
            xref="paper",
            yref="paper",
            x={1: 0.16, 2: 0.4, 3: 0.84}.get(col, 0.2),
            y=0.88,
            showarrow=False,
            font=dict(size=12, color="black"),
            align="center",
            bgcolor=color,
            bordercolor=color,
            borderwidth=1,
            borderpad=4,
        )

    return bl_lenghts


def create_meta(data):
    meta = {}
    meta["filename"] = data["file"]
    meta["target"] = data["TARGET"]
    meta["cat"] = data["CATEGORY"]
    meta["date"] = data["DATEOBS"]
    meta["dit"] = data["DIT"] * u.s
    meta["disp"] = data["DISP"]
    meta["band"] = data["BAND"]
    meta["bcd"] = f"{data['BCD1NAME']}-{data['BCD2NAME']}"
    meta["tel"] = data["TEL_NAME"]
    config = data["STA_NAME"][0]
    for bl in data["STA_NAME"][1:]:
        config += "-" + bl
    meta["config"] = config

    # Chopping mode
    chop = data["HDR"].get("HIERARCH ESO ISS CHOP ST", "F")
    chop_mode = "No" if chop == "F" else "Yes"
    meta["chopping"] = chop_mode
    return meta


def make_table(
    fig,
    data,
    title: str,
    color: str,
    row: int,
    col: int,
    x_annot: float,
    y_annot: float,
    keys: list[str] | None = None,
    fill_colors: list[list[str]] | None = None,
    add_annotation: bool = True,
):
    """
    Add a formatted table with a colored header (annotation) and borders to a Plotly figure.

    Parameters
    ----------
    fig : go.Figure
        The target Plotly figure (created with make_subplots).
    data : dict | pd.DataFrame
        Key/value pairs or DataFrame with 2 columns.
    title : str
        Title displayed above the table.
    color : str
        Header (annotation) background color.
    row, col : int
        Table position in the subplot grid.
    x_annot, y_annot : float
        Position of the annotation (paper coordinates).
    """

    # --- Convert to DataFrame ---
    if isinstance(data, dict):
        df = pd.DataFrame(list(data.items()), columns=["Parameter", "Value"])
    else:
        raise TypeError("Input must be dict.")

    # --- Default background colors ---
    if fill_colors is None:
        n_rows = len(df)
        fill_colors = [
            ["rgba(255,255,255,0.8)"] * n_rows,  # col 1
            ["rgba(255,255,255,0.8)"] * n_rows,  # col 2
        ]

    # --- Filter by selected keys (case-insensitive) ---
    if keys is not None:
        keyset = {k.lower() for k in keys}
        df = df[df["Parameter"].str.lower().isin(keyset)]

    # Add symbols
    df["Parameter"] = [f"{x} {symbol_for_mode(x)}" for x in df["Parameter"]]

    # --- Style for bold uppercase parameter names ---
    df["Parameter"] = df["Parameter"].apply(lambda x: f"<b>{x.upper()}</b>")

    values_unit = [
        f"{x.value} {x.unit}" if hasattr(x, "unit") else x for x in df["Value"]
    ]
    # --- Add table (no header, borders visible) ---
    fig.add_trace(
        go.Table(
            header=dict(
                values=["", ""],  # no header
                fill_color="#FFFFFF",
                line_color=color,  # borders same color as header
                height=0,
            ),
            cells=dict(
                values=[df["Parameter"], values_unit],
                fill_color=fill_colors,
                align=["left", "left"],
                font=dict(size=8, color="black"),
                height=20,
                line_color=color,
            ),
            columnwidth=[0.45, 0.55],
        ),
        row=row,
        col=col,
    )

    # --- Add annotation as header band ---
    if add_annotation:
        fig.add_annotation(
            text=f"<b>{title}</b>",
            xref="paper",
            yref="paper",
            x=x_annot,
            y=y_annot,
            showarrow=False,
            font=dict(size=11, color="black"),
            align="center",
            bgcolor=color,
            bordercolor=color,
            borderwidth=1,
            borderpad=4,
        )

    return fig


def _make_title_text(meta: dict, color: str = "navy") -> str:
    """Return the HTML string used in the figure title annotation."""
    target = meta.get("target", "Unknown")
    config = meta.get("config", "A0-B2-D0-C1")
    date = meta.get("date", "")
    return (
        f"<b><span style='color:{color};font-size:20px;'>{target}</span></b><br>"
        f"<span style='font-size:14px;'>Configuration: <b>{config}</b></span><br>"
        f"<span style='font-size:12px;color:#555;'>Date: {date}</span>"
    )


def make_title(fig, meta, color="navy"):
    fig.add_annotation(
        text=_make_title_text(meta, color),
        x=0.5,
        y=1.02,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="center",
        font=dict(family="Computer Modern, Times New Roman", size=15),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=color,
        borderwidth=1,
    )


def plot_spectrum(fig, data, flux_range: list[float] | None = None):
    ylabel = "Flux (arbitrary units)"

    def _annotate_missing_flux() -> bool:
        # Add as a trace (not a layout annotation) so BCD combo visibility
        # toggling via fig.data[i].visible hides/shows it correctly.
        # x = mid-wavelength (keeps xaxis2 synced with visibility plots).
        # y = 0.5 with yaxis2.range=[0,1] → always vertically centred.
        try:
            lam = np.ascontiguousarray(data["WLEN"], dtype=np.float64) * 1e6
            mid_wl = float((lam.min() + lam.max()) / 2)
            wl_range = [float(lam.min()), float(lam.max())]
        except Exception:
            mid_wl = 0.5
            wl_range = [0, 1]
        fig.add_trace(
            go.Scatter(
                x=[mid_wl],
                y=[0.5],
                mode="text",
                text=["<b>NO FLUX DATA</b>"],
                textfont=dict(size=14, color="#555"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )
        fig.update_xaxes(range=wl_range, showticklabels=False, row=2, col=1)
        fig.update_yaxes(
            domain=[0.52, 0.80],
            title=ylabel,
            range=[0, 1],
            showticklabels=False,
            row=2,
            col=1,
        )
        return False

    try:
        flux_block = data["FLUX"]
    except (KeyError, TypeError):
        return _annotate_missing_flux()

    try:
        table_flux = flux_block["FLUX"]
        sta_index = flux_block["STA_INDEX"]
    except (KeyError, TypeError):
        return _annotate_missing_flux()

    lam = np.ascontiguousarray(data["WLEN"], dtype=np.float64) * 1e6
    ref_station = {
        data["STA_INDEX"][i]: data["STA_NAME"][i] for i in range(len(data["STA_INDEX"]))
    }
    ref_telescope = {
        data["STA_INDEX"][i]: data["TEL_NAME"][i] for i in range(len(data["STA_INDEX"]))
    }

    all_flux_values = []
    for i in range(len(table_flux)):
        flux = np.ascontiguousarray(table_flux[i], dtype=np.float64)
        all_flux_values.append(flux)

    if not all_flux_values:
        return _annotate_missing_flux()

    if len(table_flux) > 4:
        custom_tel_colors = TEL_COLORS * (len(table_flux) // 4)
    else:
        custom_tel_colors = TEL_COLORS

    included_tel_names_in_legend = []
    for i in range(len(table_flux)):
        flux = np.ascontiguousarray(table_flux[i], dtype=np.float64)
        name_label = f"{ref_telescope[sta_index[i]]}-{ref_station[sta_index[i]]}"

        if name_label not in included_tel_names_in_legend:
            included_tel_names_in_legend.append(name_label)
            show_in_legend = True
        else:
            show_in_legend = False  # hide duplicate legend entries

        fig.add_trace(
            go.Scatter(
                x=lam,
                y=flux,
                mode="lines",
                name=name_label,
                line=dict(color=custom_tel_colors[i]),
                legendgroup="spectre",
                legendgrouptitle_text="Telescopes",
                showlegend=show_in_legend,
            ),
            row=2,
            col=1,
        )

    # Add vrect bands AFTER flux traces so the subplot is non-empty
    # (Plotly's add_vrect(row, col) requires at least one trace in the cell)
    # Resolve xaxis/yaxis IDs from _grid_ref to bypass the subplot-empty
    # guard, which crashes when the mixed table/xy grid contains Table traces.
    _xid, _yid = "x2", "y2"  # spectrum is always row=2, col=1 in our fixed grid
    try:
        subplot_ref = fig._grid_ref[1][0]  # (row=2, col=1) 0-indexed
        _xid = subplot_ref[0][0].trace_kwargs.get("xaxis", _xid)
        _yid = subplot_ref[0][0].trace_kwargs.get("yaxis", _yid)
    except Exception:
        pass
    add_photometric_bands(fig, xaxis_id=_xid, yaxis_id=_yid)

    wl_range = [lam[-1], lam[0]]
    fig.update_xaxes(
        title="",
        showticklabels=False,
        title_standoff=10,
        row=2,
        col=1,
        range=wl_range,
    )
    fig.update_yaxes(
        domain=[0.52, 0.80],
        title=ylabel,
        range=flux_range,  # None → Plotly auto-scales
        row=2,
        col=1,
    )

    return True


def make_uvplot(
    fig,
    u_coords,
    v_coords,
    array_type: str,
    baseline_names: list[str],
    baseline_colors: list[str],
    bl_lengths: dict[str, float],
):
    """
    Add a small UV-plane subplot with one color per baseline.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to add the UV-plane into.
    u_coords, v_coords : list[np.ndarray]
        Lists of u and v coordinates (one array per baseline).
    baseline_names : list[str]
        Names of the baselines (for legend if needed).
    baseline_colors : list[str]
        Hex colors for each baseline (same as visibility plots).
    domain_x, domain_y : list[float]
        Position and size of the UV inset (like add_axes in MPL).
    """

    # --- Plot each baseline's UV points
    seen = set()
    for i_u, i_v, name, color in zip(
        u_coords, v_coords, baseline_names, baseline_colors, strict=True
    ):
        showleg = name not in seen
        if showleg:
            seen.add(name)

        fig.add_trace(
            go.Scatter(
                x=[i_u],
                y=[i_v],
                mode="markers",
                marker=dict(size=6, color=color),
                name=f"{name} ({bl_lengths[name]:.1f} m)",
                showlegend=showleg,
                legendgroup="uv",
                legendgrouptitle_text="Baselines",
            ),
            row=2,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=[-i_u],
                y=[-i_v],
                name=name,
                mode="markers",
                marker=dict(size=6, color=color),
                showlegend=False,
            ),
            row=2,
            col=3,
        )

    if array_type.upper() == "UT":
        uv_max = 150
    else:  # AT
        uv_max = 210

    row, col = 2, 3
    fig.update_xaxes(
        title="U (m)",
        range=[-uv_max, uv_max],
        constrain="domain",
        showgrid=False,
        zeroline=True,
        row=row,
        col=col,
        domain=[0.75, 1.00],
    )

    fig.update_yaxes(
        title="V (m)",
        range=[-uv_max, uv_max],
        constrain="domain",
        showgrid=False,
        zeroline=True,
        row=row,
        col=col,
        domain=[0.52, 0.8],
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig


def plot_obs_groups(
    fig,
    data,
    baseline_names: list[str],
    baseline_colors: list[str],
    show_errors: bool = False,
    obs_name: str = "V2",
    obs_range: Sequence[int | float | None] | None = None,
    col: int = 1,
):
    """
    Plot visibility for 6 baselines, rows 3–8.
    Handles either 1 group (6 arrays) or 4 groups (24 arrays).

    Parameters
    ----------
    fig : go.Figure
        Target Plotly figure.
    data : dict
        Dictionnary of all loaded data.
    baseline_names : list[str]
        Names of the 6 baselines.
    baseline_colors : list[str]
        Hex colors for each baseline.
    obs_name : str
        Observable type: "V2" (squared visibility), "V" (correlated flux), or "dphi" (diff phase).
    obs_range : list[float] | None
        Y-axis range. If None, auto-scales for correlated flux or uses [0, 1.1] for V².
    """
    wavelength = np.ascontiguousarray(data["WLEN"], dtype=np.float64) * 1e6

    if obs_name == "dphi":
        v2_series = np.ascontiguousarray(data["VIS"]["DPHI"], dtype=np.float64)
        v2_err = np.ascontiguousarray(data["VIS"]["DPHIERR"], dtype=np.float64)
        v2_flags = data["VIS"]["FLAG"]
        title = "diff. phase [°]"
        if obs_range is None:
            obs_range = [-190, 190]
    elif obs_name == "V":
        v2_series = np.ascontiguousarray(data["VIS"]["VISAMP"], dtype=np.float64)
        v2_err = np.ascontiguousarray(data["VIS"]["VISAMPERR"], dtype=np.float64)
        v2_flags = data["VIS"]["FLAG"]
        title = "Correlated flux (Jy)"
        # Auto-scale for correlated flux (can be negative, like in legacy)
        if obs_range is None or obs_range == [0, None]:
            # Compute min/max allowing negative values
            min_val = float(np.nanmin(v2_series))
            max_val = float(np.nanmax(v2_series))
            # Add 10% margin like matplotlib autoscale
            margin = (max_val - min_val) * 0.1
            obs_range = [min_val - margin, max_val + margin]
    else:
        v2_series = np.ascontiguousarray(data["VIS2"]["VIS2"], dtype=np.float64)
        v2_err = np.ascontiguousarray(data["VIS2"]["VIS2ERR"], dtype=np.float64)
        v2_flags = data["VIS2"]["FLAG"]
        title = "V²"
        if obs_range is None:
            # V² bounds: clamp to [-0.2, 1.2] like legacy
            max_v2 = float(np.nanmax(v2_series))
            max_v2 = max_v2 if max_v2 < 1.2 else 1.2
            min_v2 = float(np.nanmin(v2_series))
            min_v2 = min_v2 if min_v2 > -0.2 else -0.2
            obs_range = [min_v2, max_v2]

    # List of arrays containing V² data.
    # - If len(v2_series) == 6  → 1 exposure group
    # - If len(v2_series) == 24 → 4 exposure groups

    n_baselines = len(baseline_names)
    n_series = len(v2_series)
    n_groups = n_series // n_baselines

    alphas = np.linspace(1.0, 0.1, n_groups)
    # Use only as many alpha values as groups
    alpha_levels = alphas[:n_groups]

    for b_idx, (bname, bcolor) in enumerate(
        zip(baseline_names, baseline_colors, strict=True)
    ):
        row = 3 + b_idx  # rows 3→8

        # Get data slices: if n_groups=1 → one array, if n_groups=4 → 4 arrays spaced by 6
        group_data = v2_series[b_idx::n_baselines]
        group_flags = v2_flags[b_idx::n_baselines]
        group_err = v2_err[b_idx::n_baselines]
        for g_idx, (data, flag, err) in enumerate(
            zip(group_data, group_flags, group_err, strict=True)
        ):
            valid = ~flag
            err_y = None

            if show_errors and err is not None:
                err_y = dict(
                    type="data",
                    array=err[valid],
                    visible=True,
                    color="lightgrey",
                    thickness=1,
                    width=2,
                )

            fig.add_trace(
                go.Scatter(
                    x=wavelength[valid],
                    y=data[valid],
                    mode="lines",
                    line=dict(width=1.8, color=bcolor),
                    opacity=alpha_levels[g_idx],
                    name=bname,
                    legendgroup=obs_name,
                    error_y=err_y,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        # Axes formatting
        fig.update_yaxes(
            range=obs_range, title=title if b_idx == 2 else "", row=row, col=col
        )
        wl_range = [wavelength.min(), wavelength.max()]
        fig.update_xaxes(
            title="Wavelength (µm)" if row == 8 else "",
            range=wl_range,
            showticklabels=(row == 8),
            row=row,
            col=col,
        )

    return fig


def plot_closure_groups(
    fig,
    data,
    triplet_names: list[str],
    triplet_colors: list[str],
    show_errors: bool = False,
    obs_name: str = "CPHASE",
    obs_range: list[float] | None = None,
    col: int = 3,
):
    """
    Plot closure phase for 4 triplets, rows 4–7.
    Handles either 1 group (4 arrays) or 4 groups (16 arrays).

    Parameters
    ----------
    fig : go.Figure
        Target Plotly figure.
    data : dict
        Dictionnary of all loaded data.
    triplet_names : list[str]
        Names of the 4 triplets.
    triplet_colors : list[str]
        Hex colors for each triplet.
    show_errors : bool, optional
        Whether to display error bars.
    obs_name : str, optional
        Observable name ("CPHASE" or "CAMP").
    obs_range : list[float], optional
        y-axis range, e.g. [-180, 180].
    col : int, optional
        Column index in the subplot grid.
    """

    if obs_range is None:
        obs_range = [-180, 180]

    wavelength = np.ascontiguousarray(data["WLEN"], dtype=np.float64) * 1e6

    # Select observable type
    if obs_name.upper() in ("CPHASE", "CP"):
        obs_series = np.ascontiguousarray(data["T3"]["CLOS"], dtype=np.float64)
        obs_err = np.ascontiguousarray(data["T3"]["CLOSERR"], dtype=np.float64)
        obs_flags = data["T3"]["FLAG"]
        title = "Closure phase [°]"
    else:
        raise ValueError(f"Unsupported observable: {obs_name}")

    # Structure: 4 triplets per group → 4 or 16 series total
    n_triplets = len(triplet_names)
    n_series = len(obs_series)
    n_groups = n_series // n_triplets

    alphas = np.linspace(1.0, 0.1, n_groups)
    alpha_levels = alphas[:n_groups]

    for t_idx, (tname, tcolor) in enumerate(
        zip(triplet_names, triplet_colors, strict=True)
    ):
        row = 4 + t_idx

        # Select group slices
        group_data = obs_series[t_idx::n_triplets]
        group_flags = obs_flags[t_idx::n_triplets]
        group_err = obs_err[t_idx::n_triplets]

        seen = set()
        for g_idx, (data_arr, flag_arr, err_arr) in enumerate(
            zip(group_data, group_flags, group_err, strict=True)
        ):
            valid = ~flag_arr
            err_y = None

            showleg = tname not in seen
            if showleg:
                seen.add(tname)

            if show_errors and err_arr is not None:
                err_y = dict(
                    type="data",
                    array=err_arr[valid],
                    visible=True,
                    color="lightgrey",
                    thickness=1,
                    width=2,
                )

            fig.add_trace(
                go.Scatter(
                    x=wavelength[valid],
                    y=data_arr[valid],
                    mode="lines",
                    line=dict(width=1.8, color=tcolor),
                    opacity=alpha_levels[g_idx],
                    name=tname,
                    legendgroup=obs_name,
                    legendgrouptitle_text="Closure phases",
                    error_y=err_y,
                    showlegend=showleg,
                ),
                row=row,
                col=col,
            )

        # Axes formatting
        fig.update_yaxes(
            range=obs_range, title=title if t_idx == 1 else "", row=row, col=col
        )
        wl_range = [wavelength[-1], wavelength[0]]
        fig.update_xaxes(
            title="Wavelength (µm)" if row == 7 else "",
            range=wl_range,
            showticklabels=(row == 7),
            row=row,
            col=col,
        )

    return fig


def _build_canvas(data: dict) -> go.Figure:
    """
    Create the 8x3 subplot canvas and add the shared title annotation from *data*.
    No data traces, no baseline / colour computation (all done per-BCD in
    :func:`_add_bcd_all_traces`).
    """
    meta = create_meta(data)

    rel_scale_vis = 0.08
    fig = make_subplots(
        rows=8,
        cols=3,
        specs=[
            [{"type": "table"}, {"type": "xy"}, {"type": "table"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
        ],
        vertical_spacing=0.01,
        horizontal_spacing=0.08,
        column_widths=[0.33, 0.33, 0.34],
        row_heights=[
            0.2,
            0.25,
            rel_scale_vis,
            rel_scale_vis,
            rel_scale_vis,
            rel_scale_vis,
            rel_scale_vis,
            rel_scale_vis,
        ],
    )

    # Title annotation is always at annotations[0] – updated by buttons in multi-BCD mode
    make_title(fig, meta)

    # Fixed axis domain settings (independent of data content)
    fig.update_yaxes(domain=[0.82, 1], row=1, col=3)
    fig.update_yaxes(domain=[0.82, 1], row=2, col=1)

    return fig


def _compute_band_ranges(band_data_dict: dict[str, dict]) -> dict:
    """
    Scan every BCD in *band_data_dict* and return consistent axis ranges for
    the whole band:

    - ``wl_range``  : ``[wl_min, wl_max]`` in µm (X axis)
    - ``vis_range`` : Y range for the V²/corr-flux rows
    - ``flux_range``: Y range for the spectrum panel

    Any value may be ``None`` when the corresponding data block is absent.
    """
    wl_min: float | None = None
    wl_max: float | None = None
    vis_min: float | None = None
    vis_max: float | None = None
    flux_min: float | None = None
    flux_max: float | None = None
    is_corrflux = False

    for bcd_data in band_data_dict.values():
        # --- Wavelength ---
        try:
            lam = np.ascontiguousarray(bcd_data["WLEN"], dtype=np.float64) * 1e6
            w0, w1 = float(lam.min()), float(lam.max())
            wl_min = w0 if wl_min is None else min(wl_min, w0)
            wl_max = w1 if wl_max is None else max(wl_max, w1)
        except Exception:
            pass

        # --- Visibility/corrflux ---
        vis2_block = bcd_data.get("VIS2")
        vis_block = bcd_data.get("VIS")
        flux_block = bcd_data.get("FLUX")
        has_flux = flux_block is not None and flux_block.get("FLUX") is not None

        if not has_flux and vis_block is not None:
            # Correlated flux (N band, no photometry)
            is_corrflux = True
            try:
                arr = np.ascontiguousarray(vis_block["VISAMP"], dtype=np.float64)
                v0, v1 = float(np.nanmin(arr)), float(np.nanmax(arr))
                vis_min = v0 if vis_min is None else min(vis_min, v0)
                vis_max = v1 if vis_max is None else max(vis_max, v1)
            except Exception:
                pass
        elif vis2_block is not None:
            # Squared visibility: clamp to [-0.2, 1.2]
            try:
                arr = np.ascontiguousarray(vis2_block["VIS2"], dtype=np.float64)
                v0 = max(-0.2, float(np.nanmin(arr)))
                v1 = min(1.2, float(np.nanmax(arr)))
                vis_min = v0 if vis_min is None else min(vis_min, v0)
                vis_max = v1 if vis_max is None else max(vis_max, v1)
            except Exception:
                pass

        # --- Spectrum ---
        if has_flux and isinstance(flux_block, dict):
            try:
                arr = np.ascontiguousarray(flux_block["FLUX"], dtype=np.float64)
                f0, f1 = float(np.nanmin(arr)), float(np.nanmax(arr))
                flux_min = f0 if flux_min is None else min(flux_min, f0)
                flux_max = f1 if flux_max is None else max(flux_max, f1)
            except Exception:
                pass

    wl_range: list[float] | None = None
    if wl_min is not None and wl_max is not None:
        wl_range = [wl_min, wl_max]

    vis_range: list[float] | None = None
    if vis_min is not None and vis_max is not None:
        if is_corrflux:
            margin = max((vis_max - vis_min) * 0.1, 1.0)
            vis_range = [vis_min - margin, vis_max + margin]
        else:
            vis_range = [vis_min, vis_max]

    flux_range: list[float] | None = None
    if flux_min is not None and flux_max is not None:
        margin = max((flux_max - flux_min) * 0.1, 1.0)
        flux_range = [flux_min - margin, flux_max + margin]

    return {"wl_range": wl_range, "vis_range": vis_range, "flux_range": flux_range}


def _add_bcd_all_traces(
    fig: go.Figure,
    data: dict,
    mix_color: bool = False,
    with_annotations: bool = True,
    vis_range_override: list[float] | None = None,
    flux_range_override: list[float] | None = None,
) -> tuple[int, int]:
    """
    Add **all** per-BCD traces to *fig*: tables, spectrum, VLTI layout, UV plot and
    interferometric data (V²/corrflux, diff phase, closure phase).

    Baseline names, CP names, observable type and colours are computed directly from
    *data* so that each BCD uses its own geometry.

    Parameters
    ----------
    mix_color : bool
        If ``True``, closure-phase colours are derived by mixing the baseline colours.
    with_annotations : bool
        When ``True`` (default), annotation labels (table headers, VLTI title) are
        added.  Set to ``False`` for secondary BCDs to avoid duplicate annotations.

    Returns
    -------
    tuple[int, int]
        Half-open trace index range ``[n_start, n_end)`` for all traces added.
    """
    n_start = len(fig.data)

    # --- Per-BCD geometry / observable type ---------------------------------
    baseline_names = build_blname_list(data)
    cp_names = build_cpname_list(data)

    has_flux = False
    try:
        flux_data = data.get("FLUX")
        if flux_data is not None and flux_data.get("FLUX") is not None:
            has_flux = True
    except (KeyError, TypeError):
        pass

    obs_type = "V2" if has_flux else "V"
    vis_range: list = [0.0, 1.1] if has_flux else [0.0, None]
    # Allow caller to override with a pre-computed band-wide range
    if vis_range_override is not None:
        vis_range = vis_range_override

    if not has_flux and with_annotations:
        fig.update_yaxes(showticklabels=False, row=2, col=1)
        fig.update_xaxes(showticklabels=False, row=2, col=1)

    if mix_color:
        cp_colors = mix_colors_for_closure(baseline_colors, baseline_names, cp_names)
    else:
        cp_colors = ["#FAA050", "#E0C9A6", "#BE7C7E", "#26AEB8"]

    meta = create_meta(data)

    # --- Tables ---
    make_table(
        fig,
        data=meta,
        title="Meta Information",
        color="#CFE4F8",
        row=1,
        col=1,
        x_annot=0.08,
        y_annot=1.04,
        keys=["band", "disp", "bcd", "dit", "chopping"],
        add_annotation=with_annotations,
    )

    quality: dict = {}
    quality["seeing"] = round(data["SEEING"], 2) * u.arcsec
    quality["tau0"] = round(data["TAU0"] * 1e3, 2) * u.ms
    quality["wind_speed"] = data["HDR"].get("ESO ISS AMBI WINDSP", np.nan) * u.m / u.s
    quality["humidity"] = data["HDR"].get("ESO ISS AMBI RHUM", np.nan) * u.percent
    airmass_start = data["HDR"].get("ESO ISS AIRM START", np.nan)
    airmass_end = data["HDR"].get("ESO ISS AIRM END", np.nan)
    quality["airmass"] = round((airmass_start + airmass_end) / 2.0, 2)
    fill_colors = [
        ["rgba(255,255,255,0.8)"],
        [_qc_color(v, k) for k, v in quality.items()],
    ]
    make_table(
        fig,
        data=quality,
        title="Observing conditions",
        color="#DDEAEA",
        fill_colors=fill_colors,
        row=1,
        col=3,
        x_annot=0.93,
        y_annot=1.04,
        add_annotation=with_annotations,
    )

    # --- Spectrum ---
    plot_spectrum(fig, data, flux_range=flux_range_override)

    # --- VLTI layout + UV plot ---
    ref_station = {
        data["STA_INDEX"][i]: data["STA_NAME"][i] for i in range(len(data["STA_INDEX"]))
    }
    station_flux: list = []
    try:
        flux_block = data["FLUX"]
    except (KeyError, TypeError):
        flux_block = None
    if flux_block is not None:
        try:
            sta_index = flux_block["STA_INDEX"]
        except (KeyError, TypeError):
            sta_index = None
        if sta_index is not None:
            station_flux = [ref_station[sta] for sta in sta_index]

    if data.get("VIS2") is not None:
        ucoord = data["VIS2"]["U"]
        vcoord = data["VIS2"]["V"]
    else:
        ucoord = data["VIS"]["U"]
        vcoord = data["VIS"]["V"]

    ncols = np.size(ucoord) // len(baseline_colors)
    array_type = "UT"
    if "AT" in data["TEL_NAME"][0]:
        array_type = "AT"

    bl_lengths = make_vltiplot_mini(
        fig,
        2,
        2,
        title="VLTI Layout",
        color="#E3F2FD",
        tels=station_flux,
        baseline_names=baseline_names,
        baseline_colors=baseline_colors,
        add_annotation=with_annotations,
    )

    make_uvplot(
        fig,
        u_coords=ucoord,
        v_coords=vcoord,
        array_type=array_type,
        baseline_names=baseline_names * ncols,
        baseline_colors=baseline_colors * ncols,
        bl_lengths=bl_lengths,
    )

    # --- Interferometric data ---
    plot_obs_groups(
        fig,
        data,
        baseline_names=baseline_names,
        baseline_colors=baseline_colors,
        show_errors=False,
        obs_name=obs_type,
        obs_range=vis_range,
    )
    plot_obs_groups(
        fig,
        data,
        baseline_names=baseline_names,
        baseline_colors=baseline_colors,
        show_errors=True,
        obs_name="dphi",
        obs_range=[-190, 190],
        col=2,
    )
    plot_closure_groups(
        fig,
        data,
        triplet_names=cp_names,
        triplet_colors=cp_colors,
        obs_range=[-190, 190],
    )

    n_end = len(fig.data)
    return n_start, n_end


def _finalize_figure(fig: go.Figure) -> None:
    """Apply final layout settings and x-axis wavelength linking."""
    fig.update_layout(
        template="plotly_white",
        width=1100,
        height=820,
        margin=dict(t=40, b=20, l=60, r=40),
    )

    for row in range(3, 9):
        fig.update_xaxes(matches="x2", row=row, col=1)
        fig.update_xaxes(matches="x2", row=row, col=2)

    for row in range(4, 8):
        fig.update_xaxes(matches="x2", row=row, col=3)


def make_static_matisse_plot(data, mix_color: bool = False):
    """Build a static single-BCD MATISSE figure."""
    fig = _build_canvas(data)
    _add_bcd_all_traces(fig, data, mix_color=mix_color, with_annotations=True)
    _finalize_figure(fig)
    return fig


def _detect_band_from_header(hdr: Any) -> str:
    """
    Determine spectral band from a FITS primary header.

    Returns
    -------
    str
        ``"N"`` (AQUARIUS detector), ``"LM"`` (HAWAII-2RG detector), or ``""`` unknown.
    """
    det = hdr.get("HIERARCH ESO DET CHIP NAME", "")
    if det == "AQUARIUS":
        return "N"
    if det == "HAWAII-2RG":
        return "LM"
    return ""


def _detect_chop_from_header(hdr: Any) -> str:
    """
    Determine chopping mode from a FITS primary header.

    Returns
    -------
    str
        ``"NOCHOP"`` when ``HIERARCH ESO ISS CHOP ST == "F"``, else ``"CHOP"``.
    """
    chop = hdr.get("HIERARCH ESO ISS CHOP ST", "F")
    return "NOCHOP" if chop == "F" else "CHOP"


_BCD_FILENAME_RE = re.compile(
    r"_(IN|OUT)_(IN|OUT)",
    re.IGNORECASE,
)


def _extract_bcd_from_filename(stem: str) -> str | None:
    """Return the **original** BCD key embedded in a filename, if present.

    Filenames produced by `remove_bcd` keep the original BCD mode in their
    stem (e.g. ``fake_calib_IR-LM_LOW_IN_IN_noChop_noBCD``).  When the file
    has the ``_noBCD`` suffix the FITS header is reset to OUT/OUT, so we
    fall back to the filename to recover the true BCD state.

    Returns ``None`` when no BCD pattern is found.
    """
    matches = _BCD_FILENAME_RE.findall(stem)
    if not matches:
        return None
    # Take the *last* match – earlier parts of the name may contain
    # unrelated IN/OUT tokens (e.g. band identifiers).
    bcd1, bcd2 = matches[-1]
    return f"{bcd1.upper()}_{bcd2.upper()}"


def _bcd_button_sort_key(bcd_key: str) -> tuple[int, str]:
    """Sort BCD keys as OUT/OUT, IN/IN, IN/OUT, OUT/IN, then others."""
    upper = bcd_key.upper()
    if upper.startswith("OUT_OUT"):
        return (0, upper)
    if upper.startswith("IN_IN"):
        return (1, upper)
    if upper.startswith("IN_OUT"):
        return (2, upper)
    if upper.startswith("OUT_IN"):
        return (3, upper)
    return (4, upper)


def find_siblings_all_bands(file_path: Path | str) -> dict[str, dict[str, Path]]:
    """
    Scan the parent directory of *file_path* for FITS files that share the same
    ``HIERARCH ESO TPL START`` keyword, grouped by spectral band.

    Returns
    -------
    dict[str, dict[str, Path]]
        ``{band: {bcd_key: path}}``, e.g.
        ``{"LM": {"IN_IN": ..., "OUT_OUT": ...}, "N": {"IN_IN": ...}}``.
    """
    from astropy.io import fits as astrofits

    file_path = Path(file_path).resolve()
    parent = file_path.parent

    try:
        primary_hdr = astrofits.getheader(file_path, 0)
        tpl_start = primary_hdr.get("HIERARCH ESO TPL START")
    except Exception:
        tpl_start = None

    candidates: dict[str, list[tuple[str, str, Path]]] = {}
    for fits_file in sorted(parent.glob("*.fits")):
        if fits_file.name.startswith("._"):
            continue
        try:
            hdr = astrofits.getheader(fits_file, 0)
            file_tpl = hdr.get("HIERARCH ESO TPL START")
        except Exception:
            continue

        if tpl_start is not None and file_tpl != tpl_start:
            continue

        band = _detect_band_from_header(hdr)
        if not band:
            continue

        bcd1 = hdr.get("HIERARCH ESO INS BCD1 NAME", "")
        bcd2 = hdr.get("HIERARCH ESO INS BCD2 NAME", "")

        # For _noBCD files the header is always reset to OUT/OUT
        # regardless of the original BCD state.  Prefer the filename.
        fname_bcd = _extract_bcd_from_filename(fits_file.stem)
        is_nobcd = "_noBCD" in fits_file.stem

        if is_nobcd and fname_bcd:
            base_bcd_key = fname_bcd
        elif bcd1 and bcd2:
            base_bcd_key = f"{bcd1}_{bcd2}"
        else:
            # No BCD metadata (e.g. calibrated files) → use filename stem so
            # every file gets a unique entry instead of all collapsing to IN_IN.
            base_bcd_key = fits_file.stem

        chop_tag = _detect_chop_from_header(hdr)
        candidates.setdefault(band, []).append((base_bcd_key, chop_tag, fits_file))

    result: dict[str, dict[str, Path]] = {}
    for band, entries in candidates.items():
        result[band] = {}
        base_counts: dict[str, int] = {}
        for base_key, _, _ in entries:
            base_counts[base_key] = base_counts.get(base_key, 0) + 1

        for base_key, chop_tag, fits_file in entries:
            if base_counts.get(base_key, 0) > 1:
                # Same BCD appears multiple times in this band:
                # expose chopping explicitly in selector labels.
                bcd_key = f"{base_key}_{chop_tag}"
            else:
                bcd_key = base_key

            # Last-resort disambiguation if duplicates still remain.
            key = bcd_key
            n = 1
            while key in result[band]:
                n += 1
                key = f"{bcd_key}_{n}"
            result[band][key] = fits_file

    return result


def make_multi_bcd_plot(file_path: Path | str, mix_color: bool = False) -> go.Figure:
    """
    Build a MATISSE figure with Plotly ``updatemenus`` controls to switch between
    BCD positions (IN_IN, OUT_OUT, …) and, when present, spectral bands (LM / N).

    All files sharing the same ``ESO TPL START`` as *file_path* are discovered.
    Files are grouped by band (LM / N) and BCD position.  A single static HTML
    figure is produced where buttons or a dropdown switch the active dataset.

    Falls back to :func:`make_static_matisse_plot` when only one file is found.

    Parameters
    ----------
    file_path : Path | str
        One of the OIFITS files; determines TPL START and the default combo.
    mix_color : bool, optional
        Derive closure-phase colours from baseline colours when ``True``.
    """
    from astropy.io import fits as astrofits

    from matisse.core.utils.oifits_reader import open_oifits

    file_path = Path(file_path).resolve()

    # --- Discover all (band × BCD) siblings ---------------------------------
    all_siblings = find_siblings_all_bands(file_path)  # {band: {bcd: path}}

    total_combos = sum(len(d) for d in all_siblings.values())
    if total_combos <= 1:
        data = open_oifits(str(file_path))
        return make_static_matisse_plot(data, mix_color)

    # --- Identify primary (band, BCD) ---------------------------------------
    try:
        primary_hdr = astrofits.getheader(file_path, 0)
        primary_band = _detect_band_from_header(primary_hdr)
        bcd1 = primary_hdr.get("HIERARCH ESO INS BCD1 NAME", "IN")
        bcd2 = primary_hdr.get("HIERARCH ESO INS BCD2 NAME", "IN")
        primary_bcd = f"{bcd1}_{bcd2}"
        primary_chop = _detect_chop_from_header(primary_hdr)
    except Exception:
        primary_band = next(iter(all_siblings))
        primary_bcd = next(iter(all_siblings[primary_band]))
        primary_chop = ""

    if primary_band not in all_siblings:
        primary_band = next(iter(all_siblings))
    if primary_bcd not in all_siblings[primary_band]:
        chop_key = f"{primary_bcd}_{primary_chop}" if primary_chop else ""
        if chop_key and chop_key in all_siblings[primary_band]:
            primary_bcd = chop_key
        else:
            primary_bcd = next(iter(all_siblings[primary_band]))

    # --- Load all datasets --------------------------------------------------
    all_data: dict[str, dict[str, dict]] = {}
    for band, bcd_paths in all_siblings.items():
        all_data[band] = {}
        for bcd_key, bcd_path in bcd_paths.items():
            try:
                all_data[band][bcd_key] = open_oifits(str(bcd_path))
            except Exception as exc:
                logger.warning("Could not load %s (%s) – skipped", bcd_path.name, exc)

    all_data = {b: d for b, d in all_data.items() if d}
    if not all_data:
        data = open_oifits(str(file_path))
        return make_static_matisse_plot(data, mix_color)

    if primary_band not in all_data:
        primary_band = next(iter(all_data))
    if primary_bcd not in all_data[primary_band]:
        primary_bcd = next(iter(all_data[primary_band]))

    primary_data = all_data[primary_band][primary_bcd]
    multi_band = len(all_data) > 1

    # --- Pre-compute per-band axis ranges ----------------------------------
    # One consistent wl_range / vis_range / flux_range shared across all BCDs
    # of each band, so switching BCD within a band never rescales the axes.
    band_ranges: dict[str, dict] = {
        band: _compute_band_ranges(bcd_dict) for band, bcd_dict in all_data.items()
    }

    # --- Build the shared canvas -------------------------------------------
    fig = _build_canvas(primary_data)

    # --- Add all traces for every (band, bcd) combo ------------------------
    # Ordering: secondary-band combos first, then primary band (non-primary
    # BCDs), primary BCD absolutely last.  The last _add_bcd_all_traces call
    # sets the axis ranges in the layout — we want the primary combo's ranges
    # to be the initial state shown to the user.
    all_combos_flat: list[tuple[str, str]] = []
    for band, bcd_dict in all_data.items():
        if band == primary_band:
            continue
        for bcd_key in bcd_dict:
            all_combos_flat.append((band, bcd_key))
    for bcd_key in all_data.get(primary_band, {}):
        if bcd_key != primary_bcd:
            all_combos_flat.append((primary_band, bcd_key))
    all_combos_flat.append((primary_band, primary_bcd))  # primary last
    combos_ordered = all_combos_flat

    combo_ranges: dict[tuple[str, str], tuple[int, int]] = {}
    for combo in combos_ordered:
        band, bcd_key = combo
        bcd_data = all_data[band][bcd_key]
        br = band_ranges[band]
        is_primary = combo == (primary_band, primary_bcd)
        n_start, n_end = _add_bcd_all_traces(
            fig,
            bcd_data,
            mix_color=mix_color,
            with_annotations=is_primary,
            vis_range_override=br.get("vis_range"),
            flux_range_override=br.get("flux_range"),
        )
        combo_ranges[combo] = (n_start, n_end)

    # Hide non-primary traces
    for combo, (n_start, n_end) in combo_ranges.items():
        if combo != (primary_band, primary_bcd):
            for i in range(n_start, n_end):
                fig.data[i].visible = False

    # --- Pre-compute title HTML for every combo ----------------------------
    combo_titles = {
        (band, bcd_key): _make_title_text(create_meta(bcd_data))
        for band, bcd_dict in all_data.items()
        for bcd_key, bcd_data in bcd_dict.items()
    }

    if multi_band:
        # ----------------------------------------------------------------
        # Two independent HTML <select> controls injected via post_script.
        # This is the only reliable way to cross band × BCD in a static
        # Plotly HTML file (two native Plotly updatemenus cannot share
        # state; a combined dropdown collapses orthogonal axes into one).
        # ----------------------------------------------------------------
        import json

        vis_lookup: dict[str, list[bool]] = {}
        titles_lookup: dict[str, str] = {}
        for active_combo in combos_ordered:
            vis: list[bool] = []
            for combo, (n_start, n_end) in combo_ranges.items():
                vis.extend([combo == active_combo] * (n_end - n_start))
            key = f"{active_combo[0]}::{active_combo[1]}"
            vis_lookup[key] = vis
            titles_lookup[key] = combo_titles[active_combo]

        bcds_per_band: dict[str, list[str]] = {}
        for band, bcd_key in combos_ordered:
            bcds_per_band.setdefault(band, [])
            if bcd_key not in bcds_per_band[band]:
                bcds_per_band[band].append(bcd_key)
        for band in bcds_per_band:
            bcds_per_band[band] = sorted(bcds_per_band[band], key=_bcd_button_sort_key)

        all_bands = list(dict.fromkeys(b for b, _ in combos_ordered))

        # Per-band relayout args: x-axis wavelength range + spectrum & vis y-ranges.
        # Axis IDs hardcoded from the 8x3 subplot grid (2 tables in row 1):
        #   xaxis2  = spectrum x  (all signal x-axes linked via matches="x2")
        #   yaxis2  = spectrum y
        #   yaxis5/8/11/14/17/20 = vis rows 3-8, col 1
        _VIS_Y_AXES = [
            "yaxis5",
            "yaxis8",
            "yaxis11",
            "yaxis14",
            "yaxis17",
            "yaxis20",
        ]
        layout_per_band: dict[str, dict] = {}
        for band, br in band_ranges.items():
            updates: dict = {}
            _band_has_flux = any(
                (d.get("FLUX") or {}).get("FLUX") is not None
                for d in all_data.get(band, {}).values()
            )
            if _band_has_flux:
                # Normal band with flux: show real wavelength / flux ranges
                if br.get("wl_range"):
                    updates["xaxis2.range"] = br["wl_range"]
                if br.get("flux_range"):
                    updates["yaxis2.range"] = br["flux_range"]
            else:
                # No flux: keep x synced with wl_range; fix y to [0,1] so the
                # NO FLUX DATA text (placed at mid_wl, y=0.5) stays centred.
                if br.get("wl_range"):
                    updates["xaxis2.range"] = br["wl_range"]
                updates["yaxis2.range"] = [0, 1]
            if br.get("vis_range"):
                for ax in _VIS_Y_AXES:
                    updates[f"{ax}.range"] = br["vis_range"]
            # ylabel for the middle visibility row (b_idx=2 → row 5 col 1 → yaxis11)
            # switches between V² (has flux) and Correlated flux (no flux)
            updates["yaxis11.title.text"] = (
                "V²" if _band_has_flux else "Correlated flux (Jy)"
            )
            # spectrum tick labels: hide when no flux (only the NO FLUX DATA trace shown)
            updates["xaxis2.showticklabels"] = _band_has_flux
            updates["yaxis2.showticklabels"] = _band_has_flux
            layout_per_band[band] = updates

        post_script = (
            """
(function() {
  var gd = document.querySelector('.plotly-graph-div');
  var visLookup     = """
            + json.dumps(vis_lookup)
            + """;
  var titlesLookup  = """
            + json.dumps(titles_lookup)
            + """;
  var bcdsPerBand   = """
            + json.dumps(bcds_per_band)
            + """;
  var allBands      = """
            + json.dumps(all_bands)
            + """;
  var layoutPerBand = """
            + json.dumps(layout_per_band)
            + """;
  var initBand = """
            + json.dumps(primary_band)
            + """;
  var initBcd  = """
            + json.dumps(primary_bcd)
            + """;

  // --- Build selector bar ------------------------------------------------
  var bar = document.createElement('div');
  bar.style.cssText = [
    'display:flex', 'align-items:center', 'gap:8px',
    'padding:5px 10px', 'font-family:Arial,sans-serif',
    'font-size:13px', 'background:#f5f5f5',
    'border-bottom:1px solid #d0d8e4'
  ].join(';');

  function makeLabel(text) {
    var s = document.createElement('span');
    s.textContent = text;
    s.style.fontWeight = 'bold';
    s.style.color = '#444';
    return s;
  }
  function makeSel(style_extra) {
    var s = document.createElement('select');
    s.style.cssText = (
      'padding:3px 8px;border-radius:4px;border:1px solid #9ecae1;' +
      'font-size:13px;cursor:pointer;' + (style_extra || '')
    );
    return s;
  }

  var bandSel = makeSel('min-width:70px;');
  allBands.forEach(function(b) {
    var o = new Option('Band : ' + b, b);
    if (b === initBand) o.selected = true;
    bandSel.add(o);
  });

  var bcdSel = makeSel('min-width:120px;');
  function populateBcds(band, selectBcd) {
    while (bcdSel.options.length) bcdSel.remove(0);
    (bcdsPerBand[band] || []).forEach(function(bcd) {
      var o = new Option('BCD : ' + bcd.replace(/_/g, '/'), bcd);
      bcdSel.add(o);
    });
    bcdSel.value =
      ((bcdsPerBand[band] || []).indexOf(selectBcd) !== -1)
        ? selectBcd
        : (bcdsPerBand[band] || [initBcd])[0];
  }
  populateBcds(initBand, initBcd);

  bar.appendChild(makeLabel(''));
  bar.appendChild(bandSel);
  bar.appendChild(bcdSel);
  gd.parentNode.insertBefore(bar, gd);

  // --- Update plot -------------------------------------------------------
  // On band or BCD change: update trace visibility, axis ranges (wl + y)
  // and the title annotation in a single pair of restyle/relayout calls.
  function updatePlot() {
    var band = bandSel.value;
    var key  = band + '::' + bcdSel.value;
    if (!visLookup[key]) return;
    Plotly.restyle(gd, {visible: visLookup[key]});
    var relayout = Object.assign({}, layoutPerBand[band] || {});
    relayout['annotations[0].text'] = titlesLookup[key];
    Plotly.relayout(gd, relayout);
  }

  bandSel.addEventListener('change', function() {
    // Keep the current BCD selection if it exists in the new band.
    populateBcds(bandSel.value, bcdSel.value);
    updatePlot();
  });
  bcdSel.addEventListener('change', updatePlot);
})();
"""
        )
        fig._matisse_post_script = post_script
        _finalize_figure(fig)
        return fig

    # ---- Single-band: same HTML <select> bar as multi-band, BCD only -----
    import json

    vis_lookup_sb: dict[str, list[bool]] = {}
    titles_lookup_sb: dict[str, str] = {}
    for active_combo in combos_ordered:
        vis_sb: list[bool] = []
        for combo, (n_start, n_end) in combo_ranges.items():
            vis_sb.extend([combo == active_combo] * (n_end - n_start))
        vis_lookup_sb[active_combo[1]] = vis_sb
        titles_lookup_sb[active_combo[1]] = combo_titles[active_combo]

    bcd_keys_sb = sorted(
        list(dict.fromkeys(bcd_key for _, bcd_key in combos_ordered)),
        key=_bcd_button_sort_key,
    )

    post_script_sb = (
        """
(function() {
  var gd = document.querySelector('.plotly-graph-div');
  var visLookup    = """
        + json.dumps(vis_lookup_sb)
        + """;
  var titlesLookup = """
        + json.dumps(titles_lookup_sb)
        + """;
  var allBcds      = """
        + json.dumps(bcd_keys_sb)
        + """;
  var initBcd      = """
        + json.dumps(primary_bcd)
        + """;

  var bar = document.createElement('div');
  bar.style.cssText = [
    'display:flex', 'align-items:center', 'gap:8px',
    'padding:5px 10px', 'font-family:Arial,sans-serif',
    'font-size:13px', 'background:#f5f5f5',
    'border-bottom:1px solid #d0d8e4'
  ].join(';');

  var bcdSel = document.createElement('select');
  bcdSel.style.cssText = (
    'padding:3px 8px;border-radius:4px;border:1px solid #9ecae1;' +
    'font-size:13px;cursor:pointer;min-width:120px;'
  );
  allBcds.forEach(function(bcd) {
    var o = new Option('BCD : ' + bcd.replace(/_/g, '/'), bcd);
    if (bcd === initBcd) o.selected = true;
    bcdSel.add(o);
  });
  bar.appendChild(bcdSel);
  gd.parentNode.insertBefore(bar, gd);

  function updatePlot() {
    var key = bcdSel.value;
    if (!visLookup[key]) return;
    Plotly.restyle(gd, {visible: visLookup[key]});
    Plotly.relayout(gd, {'annotations[0].text': titlesLookup[key]});
  }

  bcdSel.addEventListener('change', updatePlot);
})();
"""
    )
    fig._matisse_post_script = post_script_sb
    _finalize_figure(fig)
    return fig


def show_plot(
    fig,
    filename: str = "matisse_view.html",
    auto_open: bool = True,
    post_script: str = "",
) -> go.Figure:
    """
    Display the MATISSE plot in a persistent HTML file.
    Reusing the same file avoids opening a new browser tab each time.

    Parameters
    ----------
    fig : go.Figure
    filename : str
        Output HTML filename.
    auto_open : bool
        Open the file in the default browser after writing.
    post_script : str
        Optional JavaScript snippet injected at the end of the HTML ``<body>``.
        Used to add interactive controls (e.g. band × BCD selectors) that go
        beyond what static Plotly ``updatemenus`` support.
    """
    pio.write_html(
        fig, file=filename, auto_open=auto_open, post_script=post_script or None
    )
    print(f"✅ Figure saved to {filename}. Refresh the browser tab to see updates.")
    return fig
