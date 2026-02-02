"""
MATISSE automatic calibration CLI command (Typer-based)
"""

from pathlib import Path

import typer

from matisse_pipeline.core.auto_calib import run_calibration
from matisse_pipeline.core.utils.log_utils import (
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
        help="Directory containing raw MATISSE FITS files (default: current).",
    ),
    resultdir: Path | None = typer.Option(
        None,
        "--result-dir",
        "-r",
        help="Directory to store calibrated OIFITS (default: <datadir>_CALIBRATED).",
    ),
    timespan: float = typer.Option(
        0.04,
        "--timespan",
        "-t",
        help="Time window in days for calibrator association.",
    ),
    bands: list[str] = typer.Option(
        ["LM", "N"],
        "--bands",
        "-b",
        help="Spectral bands to process (N and/or LM).",
    ),
    cumul_block: bool = typer.Option(
        True,
        "--cumul-block/--no-cumul-block",
        help="Enable cumulBlock parameter in mat_cal_oifits.",
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
        resultdir = datadir.parent / f"{datadir.name}_CALIBRATED"
        log.info("Result directory not provided. Using <datadir>_CALIBRATED.")

    # --- 3. Show configuration ---
    section("Configuration")
    console.print(f"[cyan]Raw data directory:[/] {datadir.resolve()}")
    console.print(f"[cyan]Result directory:[/] {resultdir.resolve()}")
    console.print(f"[magenta]Timespan:[/] {timespan} days")
    console.print(f"[green]Bands:[/] {', '.join(bands)}")
    console.print(f"[yellow]Cumul block:[/] {cumul_block}")
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
        )

        log.info(f"[green][SUCCESS] Calibrated files saved to {resultdir.resolve()}")
        console.rule("[bold green]Calibration completed successfully[/]")

    except Exception as err:
        console.rule("[bold red]Calibration failed[/]")
        log.exception("MATISSE calibration execution failed.")
        typer.echo(f"[ERROR] Calibration failed: {err}")
        raise typer.Exit(code=1) from err
