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

import pandas as pd
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


def _print_tf_table(stats, title: str) -> None:
    table = Table(title=title, show_lines=False)
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


def _print_nights(df_vis2: pd.DataFrame, band: str) -> None:
    """Summary of the observing nights found in the directory."""
    table = Table(title=f"Observing nights — {band}")
    table.add_column("Night", style="cyan")
    table.add_column("Config")
    table.add_column("CAL", justify="right")
    table.add_column("SCI", justify="right")
    table.add_column("Targets")
    for night, g in df_vis2.groupby("night"):
        cal = g[g["category"] == "CAL"]
        sci = g[g["category"] == "SCI"]
        table.add_row(
            str(night),
            ", ".join(sorted(g["config"].unique())),
            str(cal["tpl"].nunique()),
            str(sci["tpl"].nunique()),
            ", ".join(sorted(sci["target"].unique()))
            + (" | " if len(sci) and len(cal) else "")
            + ", ".join(f"[dim]{t}[/]" for t in sorted(cal["target"].unique())),
        )
    console.print(table)


def _write(fig, path: Path, open_browser: bool) -> None:
    ext = path.suffix.lower()
    if ext == ".html":
        pio.write_html(fig, path, auto_open=open_browser)
    else:
        # kaleido/choreographer are very verbose at INFO level.
        for name in ("kaleido", "choreographer", "logistro"):
            logging.getLogger(name).setLevel(logging.WARNING)
        pio.write_image(fig, path)
    log.info(f"💾 Figure saved as {path}")


def run_tf(
    oidata,
    band: str,
    wl_range: tuple[float, float] | None,
    show_vis: bool,
    save: Path | None,
    open_browser: bool,
    chop: str = "all",
    nights: list[str] | None = None,
    magic: bool = False,
) -> bool:
    """Transfer-function diagnostic for one band. Return False if no data."""
    wl = wl_range or DEFAULT_WL_RANGE[band]
    dfs = {
        key: extract_timeseries(oidata, key, band, wl, chop, magic)
        for key in ("TF2", "VIS2", "T3")
    }
    if nights:
        dfs = {k: df[df["night"].isin(nights)] for k, df in dfs.items()}
    df_tf, df_vis2, df_t3 = dfs["TF2"], dfs["VIS2"], dfs["T3"]

    if df_vis2.empty:
        log.warning(f"No {band} data found.")
        return False

    _print_nights(df_vis2, band)
    if df_tf.empty:
        log.warning(
            f"No TF2 table found in {band} calibrators (CALIB_RAW_INT). "
            "Only raw V² will be shown."
        )
    for n, g in df_tf.groupby("night"):
        _print_tf_table(
            tf_statistics(g), f"Transfer function stability — {band} — night {n}"
        )
    if magic and band != "LM":
        log.warning("BCD magic numbers are only defined for LM: not applied in N.")
    mn = magic and band == "LM"
    console.print(
        f"[cyan]{band}[/]: λ ∈ [{wl[0]:.2f}, {wl[1]:.2f}] µm, chop={chop}"
        + (", [magenta]BCD magic numbers applied (display only)[/]" if mn else "")
    )

    def _fig(tf, v2, t3):
        return make_transfer_function_plot(
            tf,
            v2,
            t3,
            band=band + (" + BCD magic numbers" if mn else ""),
            wl_range=wl,
            show_vis=show_vis,
        )

    if save is not None and save.suffix.lower() not in {".html", ".png", ".pdf"}:
        console.print(
            f"[red]Unsupported format {save.suffix}. Use .html, .png or .pdf.[/]"
        )
        raise typer.Exit(code=1)

    if save is None or save.suffix.lower() == ".html":
        # Interactive: one file, night selected with a drop-down menu.
        fig = _fig(df_tf, df_vis2, df_t3)
        if save is None:
            viewer_plotly.show_plot(
                fig, filename=f"matisse_tf_{band}.html", auto_open=open_browser
            )
        else:
            _write(fig, save.with_name(f"{save.stem}_{band}.html"), open_browser)
        return True

    # Static images: one file per night.
    for n in sorted(df_vis2["night"].unique()):
        fig = _fig(
            df_tf[df_tf["night"] == n],
            df_vis2[df_vis2["night"] == n],
            df_t3[df_t3["night"] == n],
        )
        _write(fig, save.with_name(f"{save.stem}_{band}_{n}{save.suffix}"), False)
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
    magic: bool = typer.Option(
        False,
        "--magic",
        help="Apply the packaged BCD magic numbers to V² and TF² (LM only). "
        "Display only: OIFITS files are not modified.",
    ),
    nights: list[str] | None = typer.Option(
        None,
        "--night",
        "-n",
        help="Observing night(s) to show (evening date YYYY-MM-DD). Repeatable. "
        "Default: all nights, selectable with a drop-down menu.",
    ),
    save: Path | None = typer.Option(
        None,
        "--save",
        "-s",
        help="Save figure (.html, .png or .pdf). Band (and night for images) "
        "are appended to the name.",
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
    oidata = load_night(datadir)
    if not oidata:
        log.error(f"No reduced OIFITS (*_RAW_INT) found in {datadir}.")
        raise typer.Exit(code=1)
    log.info(f"{len(oidata)} OIFITS files loaded from {datadir.resolve()}")

    if tf:
        section("Transfer function")
        found = [
            run_tf(
                oidata,
                band,
                wl_range,
                show_vis,
                save,
                open_browser,
                chop,
                nights,
                magic,
            )
            for band in bands
        ]
        if not any(found):
            raise typer.Exit(code=1)
