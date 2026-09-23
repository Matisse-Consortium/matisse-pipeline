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
    name: str,
    color: str,
    show_legend: bool,
    opacity: float = 1.0,
    size: int = 7,
) -> None:
    """One trace per (BCD, chop): marker shape = BCD, open marker = Chop."""
    for (bcd, chop), g in df.groupby(["bcd", "chop"], sort=False):
        symbol = BCD_SYMBOLS.get(bcd, "x")
        if chop == "Chop":
            symbol += "-open"
        label = f"{name} {bcd}" + (" chop" if chop == "Chop" else "")
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
                name=label,
                legendgroup=label,
                showlegend=show_legend,
                customdata=_customdata(g),
                hovertemplate=HOVER,
            ),
            row=row,
            col=col,
        )


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
        row, first = i + 1, i == 0
        _add_points(
            fig,
            cal[cal["baseline"] == bl],
            row,
            1,
            f"{qty} cal",
            COLOR_CAL,
            first,
            opacity=0.6,
            size=6,
        )
        _add_points(
            fig, sci[sci["baseline"] == bl], row, 1, f"{qty} sci", COLOR_SCI, first
        )
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
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
        _add_points(fig, tf_bl, row, 1, tf_name, COLOR_TF, first)

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
            row, first = i + 1, i == 0
            fig.add_hline(y=0, line={"color": "gray", "width": 0.8}, row=row, col=2)
            _add_points(
                fig,
                cp_cal[cp_cal["baseline"] == tri],
                row,
                2,
                "CP cal",
                COLOR_CP_CAL,
                first,
            )
            _add_points(
                fig,
                cp_sci[cp_sci["baseline"] == tri],
                row,
                2,
                "CP sci",
                COLOR_SCI,
                False,
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
    base_annotations = [a.to_plotly_json() for a in fig.layout.annotations]
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

    # Initial state = first night
    for i, tr in enumerate(fig.data):
        tr.visible = i in states[0]["traces"]
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
                    {"visible": [i in st["traces"] for i in range(len(fig.data))]},
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
        height=max(170 * n_rows, 600),
        width=1500,
        hovermode="closest",
        legend={"orientation": "h", "y": -0.06, "x": 0.5, "xanchor": "center"},
        margin={"t": 110, "b": 90},
    )
    return fig
