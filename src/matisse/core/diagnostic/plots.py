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
    'BCD %{customdata[2]} | seeing %{customdata[3]:.2f}" | τ0 %{customdata[4]:.1f} ms<br>'
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
            height=None,
            font={"size": 10, "color": color},
        )


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
        Custom title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if show_vis:
        df_tf, df_vis2 = _to_vis(df_tf), _to_vis(df_vis2)
    qty = "V" if show_vis else "V²"
    tf_name = "TF" if show_vis else "TF²"

    baselines = sorted(set(df_tf["baseline"]) | set(df_vis2["baseline"]))
    triangles = sorted(df_t3["baseline"].unique()) if df_t3 is not None else []
    n_rows = max(len(baselines), len(triangles), 1)

    fig = make_subplots(
        rows=n_rows,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.02,
        horizontal_spacing=0.07,
        column_titles=[f"{qty} & {tf_name} vs time", "Closure phase vs time"],
    )

    cal = df_vis2[df_vis2["category"] == "CAL"]
    sci = df_vis2[df_vis2["category"] == "SCI"]

    for i, bl in enumerate(baselines):
        row = i + 1
        first = i == 0
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
        fig.update_yaxes(title_text=bl, row=row, col=1)

    # Normalised quantities -> fixed range. Correlated-flux TF (e.g. N band
    # in coherent mode) are in ADU/Jy: keep autorange.
    vmax = np.nanmax(
        pd.concat([df_tf["value"], df_vis2["value"]]).to_numpy(), initial=0
    )
    if vmax < 1.5:
        fig.update_yaxes(range=[-0.05, 1.15], col=1)

    if df_t3 is not None:
        cp_cal = df_t3[df_t3["category"] == "CAL"]
        cp_sci = df_t3[df_t3["category"] == "SCI"]
        for i, tri in enumerate(triangles):
            row = i + 1
            first = i == 0
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
            fig.update_yaxes(title_text=f"{tri} [°]", row=row, col=2)

    _add_target_labels(fig, df_vis2)

    fig.update_xaxes(matches="x")  # zoom in time on both columns together
    fig.update_xaxes(title_text="Time [UT]", row=n_rows, col=1)
    fig.update_xaxes(title_text="Time [UT]", row=n_rows, col=2)

    if title is None:
        date = ""
        if not df_vis2.empty:
            date = df_vis2["time"].min().strftime("%Y-%m-%d")
        wl_txt = f" — λ ∈ [{wl_range[0]:.2f}, {wl_range[1]:.2f}] µm" if wl_range else ""
        title = f"MATISSE transfer function — {date} — band {band}{wl_txt}"

    fig.update_layout(
        title={"text": title, "x": 0.5},
        template="plotly_white",
        height=max(170 * n_rows, 600),
        width=1500,
        hovermode="closest",
        legend={"orientation": "h", "y": -0.06, "x": 0.5, "xanchor": "center"},
        margin={"t": 90, "b": 90},
    )
    return fig
