from typing import Any

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from astropy import units as u
from plotly.subplots import make_subplots

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

telescope_colors = ["#A2C8E8", "#F8C8DC", "#FBE8A6", "#C7E5B4"]
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
        - `VIS2["STA_INDEX"]` : 2D array of shape (n_baselines, 2) giving
          the station index pairs for each baseline.

    Returns
    -------
    np.ndarray
        Array of baseline names (dtype='<U8' typically), e.g.:
        ["A0-G1", "K0-J1", "G1-I1", ...].
    """
    sta_index = data["STA_INDEX"]
    n_telescopes = len(sta_index)
    n_baseline = n_telescopes * (n_telescopes - 1) // 2

    # Mapping from station index to station name
    ref_station = {sta_index[i]: data["STA_NAME"][i] for i in range(n_telescopes)}

    # Get all pairs of station indices for each baseline
    all_sta_index = data["VIS2"]["STA_INDEX"][:n_baseline]

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


def add_photometric_bands(fig, ymin, ymax, row, col):
    """
    Add a vertical shaded band (like axvspan) as a trace, not a layout shape.
    """

    bands = {
        "L-band": (3.2, 4.1, "#DAE5EB"),
        "M-band": (4.5, 5.0, "#DAE5EB"),
        "N-band": (8.0, 13.0, "#DAE5EB"),
    }

    for name, (x0, x1, color) in bands.items():
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[ymin, ymin, ymax, ymax, ymin],
                fill="toself",
                fillcolor=color,
                mode="none",
                line=dict(width=0),
                hoverinfo="skip",
                name=name or "",
                showlegend=False,
                opacity=0.3,
            ),
            row=row,
            col=col,
        )

    return fig


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
                    # textposition="middle left",
                    showlegend=False,
                    name=f"{tels[i]}",
                    marker=dict(size=11, color=telescope_colors[i]),
                    # textfont=dict(size=12, color="black"),
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
        domain=[0.64, 0.92],  # hauteur et position haute
        visible=False,
        row=row,
        col=col,
    )
    # --- Add title band above
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


def make_title(fig, meta, color="navy"):
    target = meta.get("target", "Unknown")
    config = meta.get("config", "A0-B2-D0-C1")
    date = meta.get("date", "2025-05-20T00:35:04")
    fig.add_annotation(
        text=f"<b><span style='color:{color};font-size:20px;'>{target}</span></b><br>"
        f"<span style='font-size:14px;'>Configuration: <b>{config}</b></span><br>"
        f"<span style='font-size:12px;color:#555;'>Date: {date}</span>",
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


def plot_spectrum(fig, data):
    def _annotate_missing_flux() -> bool:
        fig.add_annotation(
            text="<b>NO FLUX DATA</b>",
            xref="paper",
            yref="paper",
            x=0.05,
            y=0.74,
            showarrow=False,
            font=dict(size=14, color="#555"),
            align="center",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(120,120,120,0.4)",
            borderwidth=1,
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

    ymin = float(np.nanmin(all_flux_values))
    ymax = float(np.nanmax(all_flux_values))

    add_photometric_bands(
        fig,
        ymin,
        ymax,
        row=2,
        col=1,
    )

    for i in range(len(table_flux)):
        flux = np.ascontiguousarray(table_flux[i], dtype=np.float64)
        name_label = f"{ref_telescope[sta_index[i]]}-{ref_station[sta_index[i]]}"
        fig.add_trace(
            go.Scatter(
                x=lam,
                y=flux,
                mode="lines",
                name=name_label,
                line=dict(color=telescope_colors[i]),
                legendgroup="spectre",
                legendgrouptitle_text="Telescopes",
                showlegend=True,
            ),
            row=2,
            col=1,
        )

    wl_range = [lam[-1], lam[0]]
    fig.update_xaxes(
        title="Wavelength (µm)",
        title_standoff=10,
        row=2,
        col=1,
        range=wl_range,
    )
    fig.update_yaxes(
        domain=[0.58, 0.80],
        title="Flux (arbitrary units)",
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
        uv_max = 150

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
        domain=[0.6, 0.82],
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
    obs_range: list[float] | None = None,
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
    """
    if obs_range is None:
        obs_range = [0, 1.1]

    wavelength = np.ascontiguousarray(data["WLEN"], dtype=np.float64) * 1e6

    if obs_name == "dphi":
        v2_series = np.ascontiguousarray(data["VIS"]["DPHI"], dtype=np.float64)
        v2_err = np.ascontiguousarray(data["VIS"]["DPHIERR"], dtype=np.float64)
        v2_flags = data["VIS"]["FLAG"]
        title = "diff. phase [°]"
    elif obs_name == "V":
        v2_series = np.ascontiguousarray(data["VIS"]["VISAMP"], dtype=np.float64)
        v2_err = np.ascontiguousarray(data["VIS"]["VISAMPERR"], dtype=np.float64)
        v2_flags = data["VIS"]["FLAG"]
        title = "V"
    else:
        v2_series = np.ascontiguousarray(data["VIS2"]["VIS2"], dtype=np.float64)
        v2_err = np.ascontiguousarray(data["VIS2"]["VIS2ERR"], dtype=np.float64)
        v2_flags = data["VIS2"]["FLAG"]
        title = "V²"

    # List of arrays containing V² data.
    # - If len(v2_series) == 6  → 1 exposure group
    #  - If len(v2_series) == 24 → 4 exposure groups

    n_baselines = len(baseline_names)
    n_series = len(v2_series)
    n_groups = n_series // n_baselines
    assert n_groups in (1, 4), f"Unexpected group count: {n_groups}"

    alphas = [1.0, 0.7, 0.4, 0.1]
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
        wl_range = [wavelength[-1], wavelength[0]]
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
    assert n_groups in (1, 4), f"Unexpected group count: {n_groups}"

    alphas = [1.0, 0.7, 0.4, 0.1]
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


def make_static_matisse_plot(data, mix_color: bool = False):
    meta = create_meta(data)

    rel_scale_vis = 0.08
    # --- Création du canevas 8x3 ---
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
            0.13,
            0.18,
            rel_scale_vis,
            rel_scale_vis,
            rel_scale_vis,
            rel_scale_vis,
            rel_scale_vis,
            rel_scale_vis,
        ],
        # shared_xaxes="columns",
    )

    baseline_names = build_blname_list(data)
    cp_names = build_cpname_list(data)

    # Make title with informations
    make_title(fig, meta)

    # Block 1 - Metadata
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
    )

    # Block 2 - Quality check
    # -----------------------
    quality = {}
    quality["seeing"] = data["SEEING"] * u.arcsec
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
    )

    ref_station = {
        data["STA_INDEX"][i]: data["STA_NAME"][i] for i in range(len(data["STA_INDEX"]))
    }
    station_flux = []
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

    fig.update_yaxes(domain=[0.82, 1.00], row=1, col=1)
    fig.update_yaxes(domain=[0.82, 1.00], row=1, col=3)

    # Block 3 - Spectrum
    has_flux = plot_spectrum(fig, data)
    fig.update_yaxes(domain=[0.68, 0.84], row=2, col=1)
    if not has_flux:
        fig.update_yaxes(showticklabels=False, row=2, col=1)
        fig.update_xaxes(showticklabels=False, row=2, col=1)

    ucoord = data["VIS2"]["U"]
    vcoord = data["VIS2"]["V"]
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

    # VISIBILITY PLOT
    plot_obs_groups(
        fig,
        data,
        baseline_names=baseline_names,
        baseline_colors=baseline_colors,
        show_errors=False,
    )

    # DIFF PHASE PLOT
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

    if mix_color:
        cp_colors = mix_colors_for_closure(baseline_colors, baseline_names, cp_names)
    else:
        cp_colors = ["#FAA050", "#E0C9A6", "#BE7C7E", "#26AEB8"]

    plot_closure_groups(
        fig,
        data,
        triplet_names=cp_names,
        triplet_colors=cp_colors,
        obs_range=[-190, 190],
    )

    fig.update_layout(
        template="plotly_white",
        width=1200,
        height=950,
        margin=dict(t=40, b=20, l=60, r=40),
    )

    # fig.update_xaxes(matches="x2", row=2, col=1)
    fig.update_xaxes(matches="x2", row=3, col=1)
    fig.update_xaxes(matches="x2", row=4, col=1)
    fig.update_xaxes(matches="x2", row=5, col=1)
    fig.update_xaxes(matches="x2", row=6, col=1)
    fig.update_xaxes(matches="x2", row=7, col=1)
    fig.update_xaxes(matches="x2", row=8, col=1)

    fig.update_xaxes(matches="x2", row=3, col=2)
    fig.update_xaxes(matches="x2", row=4, col=2)
    fig.update_xaxes(matches="x2", row=5, col=2)
    fig.update_xaxes(matches="x2", row=6, col=2)
    fig.update_xaxes(matches="x2", row=7, col=2)
    fig.update_xaxes(matches="x2", row=8, col=2)

    fig.update_xaxes(matches="x2", row=4, col=3)
    fig.update_xaxes(matches="x2", row=5, col=3)
    fig.update_xaxes(matches="x2", row=6, col=3)
    fig.update_xaxes(matches="x2", row=7, col=3)

    return fig


def show_plot(fig, filename: str = "matisse_view.html", auto_open: bool = True):
    """
    Display the MATISSE plot in a persistent HTML file.
    Reusing the same file avoids opening a new browser tab each time.
    """
    # Save figure in HTML file to be static on web browser
    pio.write_html(fig, file=filename, auto_open=auto_open)
    print(f"✅ Figure saved to {filename}. Refresh the browser tab to see updates.")
    return fig
