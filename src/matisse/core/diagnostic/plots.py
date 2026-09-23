"""
Plotly figures for MATISSE night diagnostics.

Modern replacement of legacy ``mat_showTransFunc.py``: V² (or V) of
calibrators and science targets together with the transfer function per
baseline (left column) and closure phases per triangle (right column),
all as a function of time. Hovering a point shows target, BCD, seeing,
tau0 and the file name, so that a suspicious calibrator can be spotted
and excluded before running ``matisse calibrate``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#: Marker symbol per BCD configuration.
BCD_SYMBOLS = {
    "OUT-OUT": "circle",
    "IN-IN": "square",
    "IN-OUT": "diamond",
    "OUT-IN": "triangle-up",
}

COLOR_TF = "#1f77b4"
COLOR_CAL = "#b0b0b0"
COLOR_SCI = "#d62728"
COLOR_CP_CAL = "#2ca02c"

#: Legend group -> colour (data source).
SOURCE_COLORS = {
    "cal": COLOR_CAL,
    "sci": COLOR_SCI,
    "tf": COLOR_TF,
    "cpcal": COLOR_CP_CAL,
}

HOVER = (
    "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
    "%{x|%H:%M:%S} UT<br>"
    "value = %{y:.3f} ± %{error_y.array:.3f}<br>"
    'BCD %{customdata[2]} %{customdata[6]} | seeing %{customdata[3]:.2f}" | τ0 %{customdata[4]:.1f} ms<br>'
    "<i>%{customdata[5]}</i><extra></extra>"
)


def _customdata(df: pd.DataFrame) -> np.ndarray:
    cols = ["target", "category", "bcd", "seeing", "tau0", "file", "chop"]
    return df[cols].to_numpy()


def _to_vis(df: pd.DataFrame) -> pd.DataFrame:
    """Convert V² (and error) into V."""
    df = df.copy()
    v = np.sqrt(np.abs(df["value"]))
    df["err"] = 0.5 * df["err"] / np.where(v > 0, v, np.nan)
    df["value"] = v
    return df


def _add_points(
    fig: go.Figure,
    df: pd.DataFrame,
    row: int,
    col: int,
    source: str,
    color: str,
    opacity: float = 1.0,
    size: int = 7,
) -> None:
    """One trace per (BCD, chop): marker shape = BCD, open marker = Chop.

    Data traces are hidden from the legend; they belong to the legend group
    ``source`` so that clicking the corresponding legend entry toggles them.
    """
    for (bcd, chop), g in df.groupby(["bcd", "chop"], sort=False):
        symbol = BCD_SYMBOLS.get(bcd, "x")
        if chop == "Chop":
            symbol += "-open"
        fig.add_trace(
            go.Scatter(
                x=g["time"],
                y=g["value"],
                error_y={"type": "data", "array": g["err"], "thickness": 1},
                mode="markers",
                marker={
                    "color": color,
                    "symbol": symbol,
                    "size": size,
                    "line": {
                        "width": 1 if chop == "Chop" else 0.5,
                        "color": color if chop == "Chop" else "black",
                    },
                },
                opacity=opacity,
                name=f"{source} {bcd} {chop}",
                legend=SOURCE_LEGEND.get(source, "legend"),
                legendgroup=source,
                showlegend=False,
                customdata=_customdata(g),
                hovertemplate=HOVER,
            ),
            row=row,
            col=col,
        )


#: Legend holding each source entry: sources are split on two rows
#: (``legend`` / ``legend2``); BCD and chopping symbols go in ``legend3``.
SOURCE_LEGEND = {"cal": "legend", "sci": "legend", "tf": "legend2", "cpcal": "legend2"}


def _add_legend(
    fig: go.Figure, sources: dict[str, str], bcds: list[str], has_chop: bool
) -> None:
    """Legend-only traces: colour = data source, marker = BCD (+ Chop).

    Each entry is its own legend group so that entries are laid out on one
    row in a horizontal legend; source groups match the data traces.
    """

    def _entry(name: str, group: str, legend: str, color: str, symbol: str) -> None:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "color": color,
                    "symbol": symbol,
                    "size": 10,
                    "line": {"width": 1, "color": "black"},
                },
                name=name,
                legend=legend,
                legendgroup=group,
                showlegend=True,
                hoverinfo="skip",
            )
        )

    for key, label in sources.items():
        _entry(label, key, SOURCE_LEGEND[key], SOURCE_COLORS[key], "square")
    for bcd in bcds:
        _entry(bcd, f"_bcd{bcd}", "legend3", "white", BCD_SYMBOLS[bcd])
    if has_chop:
        _entry("Chop (open)", "_chop", "legend3", "gray", "circle-open")


def _legend_layout(height: int, margin_t: int, margin_b: int) -> tuple[dict, list]:
    """Place the three legends below the plots (paper coordinates).

    Legend titles are drawn as annotations: plotly legend titles shift the
    first row, which would misalign the two source rows.
    """
    px = 1.0 / (height - margin_t - margin_b)  # 1 pixel in paper units
    y0 = -85 * px  # below the x-axis title
    x_src, x_bcd = 0.2, 0.55
    common = {
        "orientation": "h",
        "yanchor": "top",
        "xanchor": "left",
        "groupclick": "togglegroup",
        "tracegroupgap": 10,
    }
    layout = {
        "legend": {**common, "x": x_src, "y": y0},
        "legend2": {**common, "x": x_src, "y": y0 - 24 * px},
        "legend3": {**common, "x": x_bcd, "y": y0},
    }
    titles = [
        {
            "text": text,
            "x": x,
            "y": y0,
            "xref": "paper",
            "yref": "paper",
            "xanchor": "right",
            "yanchor": "top",
            "yshift": -3,
            "showarrow": False,
        }
        for text, x in (
            ("<b>Colour = source</b>", x_src),
            ("<b>Symbol = BCD</b>", x_bcd),
        )
    ]
    return layout, titles


def _add_target_labels(fig: go.Figure, df: pd.DataFrame) -> None:
    """Dotted vertical line at the start of each template (TPL START)."""
    if df.empty:
        return
    for _, r in df.sort_values("mjd").groupby("tpl", sort=False).head(1).iterrows():
        color = COLOR_SCI if r["category"] == "SCI" else "gray"
        fig.add_vline(
            x=r["time"],
            line={"color": color, "width": 0.6, "dash": "dot"},
            row="all",
            col=1,
        )
        fig.add_annotation(
            x=r["time"],
            y=1.0,
            xref="x",
            yref="y domain",
            text=r["target"],
            textangle=-90,
            showarrow=False,
            xanchor="right",
            yanchor="top",
            xshift=-1,
            font={"size": 10, "color": color},
        )


def _axis_name(axis: str, row: int, col: int) -> str:
    """Layout key of a 2-column make_subplots axis (``yaxis``, ``yaxis3``...)."""
    idx = (row - 1) * 2 + col
    return axis if idx == 1 else f"{axis}{idx}"


def _add_night(
    fig: go.Figure,
    df_tf: pd.DataFrame,
    df_vis2: pd.DataFrame,
    df_t3: pd.DataFrame | None,
    n_rows: int,
    qty: str,
    tf_name: str,
) -> dict:
    """Add the traces/shapes of one night; return its night-specific layout."""
    layout: dict = {}
    baselines = sorted(set(df_tf["baseline"]) | set(df_vis2["baseline"]))
    cal = df_vis2[df_vis2["category"] == "CAL"]
    sci = df_vis2[df_vis2["category"] == "SCI"]

    for i, bl in enumerate(baselines):
        row = i + 1
        _add_points(
            fig,
            cal[cal["baseline"] == bl],
            row,
            1,
            "cal",
            COLOR_CAL,
            opacity=0.6,
            size=6,
        )
        _add_points(fig, sci[sci["baseline"] == bl], row, 1, "sci", COLOR_SCI)
        tf_bl = df_tf[df_tf["baseline"] == bl].sort_values("mjd")
        # TF interpolated linearly between calibrators, per BCD position
        # and chopping mode (mat_cal_oifits calibrates them separately).
        for _, g in tf_bl.groupby(["bcd", "chop"], sort=False):
            fig.add_trace(
                go.Scatter(
                    x=g["time"],
                    y=g["value"],
                    mode="lines",
                    line={"color": COLOR_TF, "width": 0.8, "dash": "dot"},
                    opacity=0.6,
                    hoverinfo="skip",
                    legend=SOURCE_LEGEND["tf"],
                    legendgroup="tf",
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
        _add_points(fig, tf_bl, row, 1, "tf", COLOR_TF)

    # Normalised quantities -> fixed range. Correlated-flux TF (e.g. N band
    # in coherent mode) are in ADU/Jy: keep autorange.
    vmax = np.nanmax(
        pd.concat([df_tf["value"], df_vis2["value"]]).to_numpy(), initial=0
    )
    for row in range(1, n_rows + 1):
        y = _axis_name("yaxis", row, 1)
        label = baselines[row - 1] if row <= len(baselines) else ""
        layout[f"{y}.title.text"] = label
        if vmax < 1.5:
            layout[f"{y}.range"] = [-0.05, 1.15]
            layout[f"{y}.autorange"] = False
        else:
            layout[f"{y}.autorange"] = True

    triangles = sorted(df_t3["baseline"].unique()) if df_t3 is not None else []
    if df_t3 is not None:
        cp_cal = df_t3[df_t3["category"] == "CAL"]
        cp_sci = df_t3[df_t3["category"] == "SCI"]
        for i, tri in enumerate(triangles):
            row = i + 1
            fig.add_hline(y=0, line={"color": "gray", "width": 0.8}, row=row, col=2)
            _add_points(
                fig,
                cp_cal[cp_cal["baseline"] == tri],
                row,
                2,
                "cpcal",
                COLOR_CP_CAL,
            )
            _add_points(
                fig,
                cp_sci[cp_sci["baseline"] == tri],
                row,
                2,
                "sci",
                COLOR_SCI,
            )
    for row in range(1, n_rows + 1):
        y = _axis_name("yaxis", row, 2)
        layout[f"{y}.title.text"] = (
            f"{triangles[row - 1]} [°]" if row <= len(triangles) else ""
        )

    _add_target_labels(fig, df_vis2)

    times = pd.concat([df_vis2["time"], df_tf["time"]])
    pad = pd.Timedelta(minutes=15)
    layout["xaxis.range"] = [times.min() - pad, times.max() + pad]
    layout["xaxis.autorange"] = False
    return layout


def make_transfer_function_plot(
    df_tf: pd.DataFrame,
    df_vis2: pd.DataFrame,
    df_t3: pd.DataFrame | None = None,
    band: str = "LM",
    wl_range: tuple[float, float] | None = None,
    show_vis: bool = False,
    title: str | None = None,
) -> go.Figure:
    """Build the transfer-function diagnostic figure.

    Data are split by observing night (``night`` column, see
    :func:`matisse.core.diagnostic.night.night_label`). With several nights,
    a drop-down menu selects the night to display; only the first one is
    visible at start. Rows follow the baselines of each night, so nights
    with different array configurations can be mixed.

    Parameters
    ----------
    df_tf, df_vis2, df_t3 : pandas.DataFrame
        Outputs of :func:`matisse.core.diagnostic.night.extract_timeseries`
        for the ``TF2``, ``VIS2`` and ``T3`` tables.
    band : str
        Band label for the title.
    wl_range : (float, float), optional
        Wavelength window (µm), shown in the title.
    show_vis : bool
        Plot V and TF instead of V² and TF².
    title : str, optional
        Custom title (same for all nights).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if show_vis:
        df_tf, df_vis2 = _to_vis(df_tf), _to_vis(df_vis2)
    qty = "V" if show_vis else "V²"
    tf_name = "TF" if show_vis else "TF²"

    nights = sorted(set(df_vis2["night"]) | set(df_tf["night"]))
    if not nights:
        nights = [""]

    def _sel(df: pd.DataFrame, n: str) -> pd.DataFrame:
        return df[df["night"] == n]

    def _sel_t3(n: str) -> pd.DataFrame | None:
        return None if df_t3 is None else _sel(df_t3, n)

    n_rows = 1
    for n in nights:
        n_bl = len(set(_sel(df_tf, n)["baseline"]) | set(_sel(df_vis2, n)["baseline"]))
        n_tri = (
            df_t3.loc[df_t3["night"] == n, "baseline"].nunique()
            if df_t3 is not None
            else 0
        )
        n_rows = max(n_rows, n_bl, n_tri)

    fig = make_subplots(
        rows=n_rows,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.02,
        horizontal_spacing=0.07,
        column_titles=[f"{qty} & {tf_name} vs time", "Closure phase vs time"],
    )
    height = max(170 * n_rows, 600)
    margin = {"t": 110, "b": 170}
    legend_layout, legend_titles = _legend_layout(height, margin["t"], margin["b"])
    base_annotations = [
        a.to_plotly_json() for a in fig.layout.annotations
    ] + legend_titles
    wl_txt = f" — λ ∈ [{wl_range[0]:.2f}, {wl_range[1]:.2f}] µm" if wl_range else ""

    states = []
    for n in nights:
        i_trace = len(fig.data)
        i_shape, i_ann = len(fig.layout.shapes), len(fig.layout.annotations)
        layout = _add_night(
            fig, _sel(df_tf, n), _sel(df_vis2, n), _sel_t3(n), n_rows, qty, tf_name
        )
        v = _sel(df_vis2, n)
        n_cal = v.loc[v["category"] == "CAL", "tpl"].nunique()
        n_sci = v.loc[v["category"] == "SCI", "tpl"].nunique()
        states.append(
            {
                "night": n,
                "label": f"Night {n} ({n_cal} CAL / {n_sci} SCI)",
                "traces": range(i_trace, len(fig.data)),
                "shapes": [x.to_plotly_json() for x in fig.layout.shapes[i_shape:]],
                "annotations": [
                    x.to_plotly_json() for x in fig.layout.annotations[i_ann:]
                ],
                "layout": layout,
                "title": title
                or f"MATISSE transfer function — night {n} — band {band}{wl_txt}",
            }
        )

    def _update(st: dict) -> dict:
        return {
            "shapes": st["shapes"],
            "annotations": base_annotations + st["annotations"],
            "title.text": st["title"],
            **st["layout"],
        }

    # Legend-only traces, always visible
    i_legend = len(fig.data)
    all_df = pd.concat([d for d in (df_tf, df_vis2, df_t3) if d is not None])
    sources = {}
    if (df_vis2["category"] == "CAL").any():
        sources["cal"] = f"{qty} calibrator"
    if all_df["category"].eq("SCI").any():
        sources["sci"] = f"{qty} / CP science"
    if not df_tf.empty:
        sources["tf"] = f"Transfer function {tf_name}"
    if df_t3 is not None and (df_t3["category"] == "CAL").any():
        sources["cpcal"] = "CP calibrator"
    bcds = [b for b in BCD_SYMBOLS if b in set(all_df["bcd"])]
    _add_legend(fig, sources, bcds, all_df["chop"].eq("Chop").any())

    def _mask(st: dict) -> list[bool]:
        return [i in st["traces"] or i >= i_legend for i in range(len(fig.data))]

    # Initial state = first night
    for tr, vis in zip(fig.data, _mask(states[0]), strict=True):
        tr.visible = vis
    first = _update(states[0])
    # Assign arrays directly: update_layout() would merge them element-wise.
    fig.layout.shapes = first.pop("shapes")
    fig.layout.annotations = first.pop("annotations")
    fig.update_layout(first)

    if len(states) > 1:
        buttons = [
            {
                "label": st["label"],
                "method": "update",
                "args": [
                    {"visible": _mask(st)},
                    _update(st),
                ],
            }
            for st in states
        ]
        fig.update_layout(
            updatemenus=[
                {
                    "buttons": buttons,
                    "direction": "down",
                    "showactive": True,
                    "x": 0.0,
                    "xanchor": "left",
                    "y": 1.07,
                    "yanchor": "bottom",
                }
            ]
        )

    fig.update_xaxes(matches="x")  # zoom in time on both columns together
    fig.update_xaxes(title_text="Time [UT]", row=n_rows, col=1)
    fig.update_xaxes(title_text="Time [UT]", row=n_rows, col=2)
    fig.update_layout(
        title={"x": 0.5},
        template="plotly_white",
        height=height,
        width=1500,
        hovermode="closest",
        margin=margin,
        **legend_layout,
    )
    return fig
