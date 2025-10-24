"""
MATISSE automatic data reduction CLI command (Typer-based)
Refactored from mat_autoPipeline.py — B008-safe version
"""

from enum import Enum
from pathlib import Path

import typer

from matisse_pipeline.core.auto_pipeline import (
    run_pipeline,
)
from matisse_pipeline.core.utils.log_utils import (
    console,
    log,
    section,
    set_verbosity,
)

app = typer.Typer(help="Reduce MATISSE raw data automatically.")


class Resolution(str, Enum):
    LOW = "LOW"
    MED = "MED"
    HIGH = "HIGH"

    @classmethod
    def _missing_(cls, value: object):  # To be insensitive to case
        if isinstance(value, str):
            value_upper = value.upper()
            for member in cls:
                if member.value == value_upper:
                    return member
        raise ValueError(
            f"Invalid resolution '{value}'. Choose one of: LOW, MED, HIGH."
        )


def reduce(
    datadir: Path = typer.Option(
        Path.cwd(),
        "--datadir",
        "-d",
        help="Directory containing raw MATISSE FITS files (default: current).",
    ),
    calibdir: Path | None = typer.Option(
        None,
        "--calibdir",
        "-c",
        help="Calibration directory (default: same as raw).",
    ),
    resultdir: Path | None = typer.Option(
        None,
        "--resultdir",
        "-r",
        help="Directory to store reduction results (default: current).",
    ),
    nbcore: int = typer.Option(1, "--nbcore", "-n", help="Number of CPU cores to use."),
    tplid: str = typer.Option("", "--tplid", help="Template ID to select."),
    tplstart: str = typer.Option("", "--tplstart", help="Template start to select."),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing results."
    ),
    skip_l: bool = typer.Option(False, "--skipL", help="Skip L band data."),
    skip_n: bool = typer.Option(False, "--skipN", help="Skip N band data."),
    resol: Resolution = typer.Option(
        Resolution.LOW,
        "--resol",
        help="Spectral resolution (LOW, MED, HIGH). Case-insensitive.",
    ),
    spectral_binning: str = typer.Option(
        "", "--spectral-binning", help="Spectral binning to improve SNR."
    ),
    max_iter: int = typer.Option(
        1, "--max-iter", help="Maximum number of reduction iterations."
    ),
    param_n: str = typer.Option(
        "/useOpdMod=TRUE", "--paramN", help="Recipe parameters for N band."
    ),
    param_l: str = typer.Option(
        "/tartyp=57/useOpdMod=FALSE",
        "--paramL",
        help="Recipe parameters for L/M band.",
    ),
    check_blocks: bool = typer.Option(
        False,
        "--check",
        help="Check FITS files and the different pipeline blocks to be executed.",
    ),
    check_calib: bool = typer.Option(
        False,
        "--check_cal",
        help="Check if calibration files already processed.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose mode"),
):
    """
    Run the MATISSE automatic pipeline.
    """
    # --- 1. Verbosity and header ---
    section("MATISSE Reduction Pipeline")
    set_verbosity(log, verbose)

    # --- 2. Handle defaults manually ---
    if calibdir is None:
        calibdir = datadir
        log.info("Calibration directory not provided. Using datadir as fallback.")
    if resultdir is None:
        resultdir = Path.cwd()
        log.info("Result directory not provided. Using current directory.")

    # --- 3. Show configuration summary ---
    section("Configuration")
    console.print(f"[cyan]Raw data directory:[/] {datadir.resolve()}")
    console.print(f"[cyan]Calibration directory:[/] {calibdir.resolve()}")
    console.print(f"[cyan]Result directory:[/] {resultdir.resolve()}")
    console.print(f"[magenta]CPU cores:[/] {nbcore}")
    console.print(f"[green]Resolution:[/] {resol.value}")
    console.print(f"[yellow]Max iterations:[/] {max_iter}")
    console.print(f"[dim]Verbose:[/] {'ON' if not verbose else 'OFF'}")

    # --- 4. Resolve paths for core function ---
    dir_raw = str(datadir.resolve()) + "/"
    dir_calib = str(calibdir.resolve())
    dir_result = str(resultdir.resolve())

    # --- 5. Run pipeline and handle errors ---
    try:
        run_pipeline(
            dirRaw=dir_raw,
            dirResult=dir_result,
            dirCalib=dir_calib,
            nbCore=nbcore,
            resol=resol,
            paramL=param_l,
            paramN=param_n,
            overwrite=int(overwrite),
            maxIter=max_iter,
            skipL=skip_l,
            skipN=skip_n,
            tplstartsel=tplstart,
            tplidsel=tplid,
            spectralBinning=spectral_binning,
            check_blocks=check_blocks,
            check_calib=check_calib,
        )
        log.info(f"[green][SUCCESS] Results saved to {dir_result}")
        console.rule("[bold green]Reduction completed successfully[/]")
    except Exception as err:
        console.rule("[bold red]Reduction failed[/]")
        log.exception("MATISSE pipeline execution failed.")
        typer.echo(f"[ERROR] Reduction failed: {err}")
        raise typer.Exit(code=1) from err


# -------------------------
# Main entrypoint
# -------------------------
def main():
    """CLI entrypoint for MATISSE pipeline reduction."""
    app()


if __name__ == "__main__":
    main()
