"""
MATISSE automatic calibration CLI command (Typer-based)
"""

from pathlib import Path

import typer

from matisse.core.auto_calib import run_calibration
from matisse.core.utils.log_utils import (
    console,
    log,
    section,
    set_verbosity,
)


def calibrate(
    datadir: Path = typer.Option(
        Path.cwd(),
        "--data-dir",
        "-d",
        help="Directory containing reduced MATISSE OIFITS files (default: current).",
    ),
    resultdir: Path | None = typer.Option(
        None,
        "--result-dir",
        "-r",
        help="Directory to store calibrated OIFITS (default: calibrated).",
    ),
    timespan: float = typer.Option(
        1,
        "--timespan",
        "-t",
        help="Time window in hours for calibrator association.",
    ),
    bands: list[str] = typer.Option(
        ["LM", "N"],
        "--bands",
        "-b",
        help="Spectral bands to process (N and/or LM).",
    ),
    cumul_block: bool = typer.Option(
        False,
        "--cumul-block/--no-cumul-block",
        help="Enable cumulBlock parameter in mat_cal_oifits (rarely used, expert only).",
    ),
    force_sci: list[str] | None = typer.Option(
        None,
        "--force-sci",
        help="Target name (HIERARCH ESO OBS TARG NAME) to treat as SCI even if classified as CALIB_RAW_INT. Repeatable. Useful when the dataset contains only calibrators.",
    ),
    custom_recipes_dir: Path | None = typer.Option(
        None,
        "--recipe-dir",
        help="Custom directory for MATISSE recipes (default: user esorex repository).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose mode.",
    ),
):
    """
    Run automatic calibration (transfer function).

    This command generates SOF files associating science targets
    with their calibrators (close in time), then runs the mat_cal_oifits recipe from
    the MATISSE DRS to produce calibrated OIFITS files.
    """
    # --- 1. Verbosity and header ---
    section("MATISSE Calibration Pipeline")
    set_verbosity(log, verbose)

    # --- 2. Handle defaults ---
    if resultdir is None:
        resultdir = Path("calibrated")
        log.debug("Result directory not provided. Using calibrated/.")

    # --- 3. Show configuration ---
    section("Configuration")
    console.print(f"[cyan]Raw data directory:[/] {datadir.resolve()}")
    console.print(f"[cyan]Result directory:[/] {resultdir.resolve()}")
    console.print(f"[magenta]Timespan:[/] {timespan} hours")
    console.print(f"[green]Bands:[/] {', '.join(bands)}")
    if force_sci:
        console.print(f"[yellow]Forced SCI targets:[/] {', '.join(force_sci)}")
    console.print(f"[dim]Verbose:[/] {'ON' if verbose else 'OFF'}")

    # --- 4. Validate bands ---
    valid_bands = {"N", "LM"}
    invalid = set(bands) - valid_bands
    if invalid:
        console.print(f"[red]Invalid bands: {invalid}. Choose from N, LM.[/]")
        raise typer.Exit(code=1)

    # --- 5. Run calibration ---
    try:
        run_calibration(
            input_dir=datadir,
            output_dir=resultdir,
            bands=bands,
            timespan=timespan,
            cumul_block=cumul_block,
            custom_recipes_dir=custom_recipes_dir,
            force_sci_names=force_sci,
        )

        log.info(
            f"[green][SUCCESS] Calibrated files saved to[/] [magenta]{Path(*resultdir.resolve().parts[-2:])}/[/]"
        )
        console.rule("[bold green]Calibration completed successfully[/]")

    except Exception as err:
        console.rule("[bold red]Calibration failed[/]")
        log.exception("MATISSE calibration execution failed.")
        typer.echo(f"[ERROR] Calibration failed: {err}")
        raise typer.Exit(code=1) from err
