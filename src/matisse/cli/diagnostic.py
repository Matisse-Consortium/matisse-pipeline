"""
MATISSE night diagnostic CLI command.

Gathers visualisation tools that help deciding how to reduce/calibrate a
night (which calibrators to keep, which time span, which band...).
Each diagnostic is enabled with its own flag, e.g.::

    matisse diagnostic --tf                       # ./reduced_OIFITS
    matisse diagnostic path/reduced_OIFITS --tf -b LM --chop nochop --vis
"""

from __future__ import annotations

import logging
from pathlib import Path

import plotly.io as pio
import typer
from rich.table import Table

from matisse.core.diagnostic import (
    DEFAULT_WL_RANGE,
    extract_timeseries,
    load_night,
    make_transfer_function_plot,
    tf_statistics,
)
from matisse.core.utils.log_utils import console, log, section, set_verbosity
from matisse.viewer import viewer_plotly

VALID_BANDS = ("LM", "N")


def _print_tf_table(stats, band: str) -> None:
    table = Table(title=f"Transfer function stability — {band}", show_lines=False)
    table.add_column("Baseline", style="cyan")
    table.add_column("N pts", justify="right")
    table.add_column("N cal", justify="right")
    table.add_column("median", justify="right")
    table.add_column("scatter [%]", justify="right")
    table.add_column("min / max", justify="right")
    for bl, r in stats.iterrows():
        scatter = r["scatter_pct"]
        style = "green" if scatter < 5 else "yellow" if scatter < 10 else "red"
        table.add_row(
            str(bl),
            f"{int(r['n'])}",
            f"{int(r['n_cal'])}",
            f"{r['median']:.3f}",
            f"[{style}]{scatter:.1f}[/]",
            f"{r['min']:.3f} / {r['max']:.3f}",
        )
    console.print(table)


def _output(fig, save: Path | None, band: str, open_browser: bool) -> None:
    if save is None:
        viewer_plotly.show_plot(
            fig, filename=f"matisse_tf_{band}.html", auto_open=open_browser
        )
        return
    # One file per band when several bands are plotted.
    path = save.with_name(f"{save.stem}_{band}{save.suffix}")
    ext = path.suffix.lower()
    if ext == ".html":
        pio.write_html(fig, path, auto_open=open_browser)
    elif ext in {".png", ".pdf"}:
        # kaleido/choreographer are very verbose at INFO level.
        for name in ("kaleido", "choreographer", "logistro"):
            logging.getLogger(name).setLevel(logging.WARNING)
        pio.write_image(fig, path)
    else:
        console.print(f"[red]Unsupported format {ext}. Use .html, .png or .pdf.[/]")
        raise typer.Exit(code=1)
    log.info(f"💾 Figure saved as {path}")


def run_tf(
    night,
    band: str,
    wl_range: tuple[float, float] | None,
    show_vis: bool,
    save: Path | None,
    open_browser: bool,
    chop: str = "all",
) -> bool:
    """Transfer-function diagnostic for one band. Return False if no data."""
    wl = wl_range or DEFAULT_WL_RANGE[band]
    df_tf = extract_timeseries(night, "TF2", band, wl, chop)
    df_vis2 = extract_timeseries(night, "VIS2", band, wl, chop)
    df_t3 = extract_timeseries(night, "T3", band, wl, chop)

    if df_vis2.empty:
        log.warning(f"No {band} data found.")
        return False
    if df_tf.empty:
        log.warning(
            f"No TF2 table found in {band} calibrators (CALIB_RAW_INT). "
            "Only raw V² will be shown."
        )
    else:
        _print_tf_table(tf_statistics(df_tf), band)

    n_cal = df_vis2.loc[df_vis2["category"] == "CAL", "tpl"].nunique()
    n_sci = df_vis2.loc[df_vis2["category"] == "SCI", "tpl"].nunique()
    console.print(
        f"[cyan]{band}[/]: {n_cal} CAL / {n_sci} SCI templates "
        f"({df_vis2['file'].nunique()} files), "
        f"λ ∈ [{wl[0]:.2f}, {wl[1]:.2f}] µm"
    )

    fig = make_transfer_function_plot(
        df_tf, df_vis2, df_t3, band=band, wl_range=wl, show_vis=show_vis
    )
    _output(fig, save, band, open_browser)
    return True


def diagnostic(
    datadir: Path = typer.Argument(
        Path("reduced_OIFITS"),
        help="reduced_OIFITS directory (output of 'matisse format').",
    ),
    tf: bool = typer.Option(
        False,
        "--tf",
        help="Transfer function (TF²) + raw V² and closure phases vs time.",
    ),
    bands: list[str] = typer.Option(
        list(VALID_BANDS),
        "--band",
        "-b",
        help="Band(s) to display (LM and/or N). Repeatable.",
    ),
    wl_range: tuple[float, float] | None = typer.Option(
        None,
        "--wl-range",
        help="Wavelength window in µm used to average channels "
        "(default: LM 3.0-4.0, N 8.5-10.5).",
    ),
    show_vis: bool = typer.Option(
        False,
        "--vis",
        help="Show V and TF instead of V² and TF².",
    ),
    chop: str = typer.Option(
        "all",
        "--chop",
        help="Chopping mode to display: all, chop or nochop (Chop = open markers).",
        case_sensitive=False,
    ),
    save: Path | None = typer.Option(
        None,
        "--save",
        "-s",
        help="Save figure (.html, .png or .pdf); band is appended to the name.",
    ),
    open_browser: bool = typer.Option(
        True,
        "--open/--no-open",
        help="Open the HTML figure in the browser.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose mode."),
):
    """
    Night diagnostics to help decide how to reduce and calibrate MATISSE data.

    Select one or several diagnostics with the dedicated flags (currently: --tf).
    """
    set_verbosity(log, verbose)

    if not tf:
        console.print("[yellow]No diagnostic selected. Use at least one of: --tf[/]")
        raise typer.Exit(code=1)

    invalid = set(bands) - set(VALID_BANDS)
    if invalid:
        console.print(f"[red]Invalid band(s): {invalid}. Choose from LM, N.[/]")
        raise typer.Exit(code=1)

    chop = chop.lower()
    if chop not in ("all", "chop", "nochop"):
        console.print(
            f"[red]Invalid --chop {chop!r}. Choose from all, chop, nochop.[/]"
        )
        raise typer.Exit(code=1)

    if not datadir.is_dir():
        log.error(f"❌ Directory {datadir} not found.")
        raise typer.Exit(code=1)

    section("MATISSE night diagnostic")
    night = load_night(datadir)
    if not night:
        log.error(f"No reduced OIFITS (*_RAW_INT) found in {datadir}.")
        raise typer.Exit(code=1)
    log.info(f"{len(night)} OIFITS files loaded from {datadir.resolve()}")

    if tf:
        section("Transfer function")
        found = [
            run_tf(night, band, wl_range, show_vis, save, open_browser, chop)
            for band in bands
        ]
        if not any(found):
            raise typer.Exit(code=1)
