"""
MATISSE automatic data reduction CLI command (Typer-based)
Refactored from mat_autoPipeline.py — B008-safe version
"""

from enum import Enum
from pathlib import Path

import typer

from matisse.cli.doctor import (
    _get_env_recipe_dirs,
    find_matisse_recipe_dir,
)
from matisse.core.auto_pipeline import (
    run_pipeline,
)
from matisse.core.utils.log_utils import (
    console,
    log,
    section,
    set_verbosity,
)


class Resolution(str, Enum):
    LOW = "LOW"
    MED = "MED"
    HIGH = "HIGH"


def reduce(
    datadir: Path = typer.Option(
        Path.cwd(),
        "--data-dir",
        "-d",
        help="Directory containing raw MATISSE FITS files (default: current).",
    ),
    calibdir: Path | None = typer.Option(
        None,
        "--calib-dir",
        "-c",
        help="Calibration directory (default: same as raw).",
    ),
    resultdir: Path | None = typer.Option(
        None,
        "--result-dir",
        "-r",
        help="Directory to store reduction results (default: current).",
    ),
    max_iter: int = typer.Option(
        1, "--max-iter", help="Maximum number of reduction iterations."
    ),
    nbcore: int = typer.Option(1, "--nbcore", "-n", help="Number of CPU cores to use."),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing results."
    ),
    tplid: str = typer.Option("", "--tplid", help="Template ID to select."),
    tplstart: str = typer.Option("", "--tplstart", help="Template start to select."),
    skip_l: bool = typer.Option(False, "--skipL", help="Skip L band data."),
    skip_n: bool = typer.Option(False, "--skipN", help="Skip N band data."),
    resol: Resolution = typer.Option(
        Resolution.LOW,
        "--resol",
        help="Spectral resolution.",
    ),
    spectral_binning: str = typer.Option(
        "", "--spectral-binning", help="Spectral binning to improve SNR."
    ),
    custom_recipes_dir: Path | None = typer.Option(
        None,
        "--recipe-dir",
        help="Custom directory for MATISSE recipes (default: user esorex repository).",
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
        "--check-blocks",
        help="Check FITS files and the different pipeline blocks to be executed.",
    ),
    check_calib: bool = typer.Option(
        False,
        "--check-cal",
        help="Check if calibration files already processed.",
    ),
    detailed_block: int | None = typer.Option(
        None,
        "--block-cal",
        help="Show calibration filenames attached to the given reduction block number.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose mode"),
):
    """
    Run automatic reduction to produce uncalibrated oifits files and processed calibration maps if needed beforehand.

    It processes raw MATISSE data files and produces
    uncalibrated results. This pipeline is designed to run over multiple iterations (--max-iter)
    in order to first reduce calibration files (flat, kappa matrix, shift map, etc.), and
    then science files using the calibration products obtained in previous iterations.
    We encourage users to run the pipeline using multiple cores (--nbcore) to speed up
    processing time. The final results are stored in FITS files in the --result-dir
    directory and can later be formatted into OIFITS files using the *format* command.
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
    _show_recipe_info(custom_recipes_dir)
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
            detailed_block=detailed_block,
            custom_recipes_dir=custom_recipes_dir,
        )
        if not check_blocks and not check_calib:
            log.info(f"[green][SUCCESS] Results saved to {dir_result}")
            console.rule("[bold green]Reduction completed successfully[/]")
        else:
            console.rule("[bold green]Check mode: no files will be processed[/]")
    except Exception as err:
        console.rule("[bold red]Reduction failed[/]")
        log.exception("MATISSE pipeline execution failed.")
        typer.echo(f"[ERROR] Reduction failed: {err}")
        raise typer.Exit(code=1) from err


def _show_recipe_info(custom_recipes_dir: Path | None = None) -> None:
    """Display MATISSE recipe directory and available recipes."""
    # Check for ESOREX_PLUGIN_DIR environment variable first
    if custom_recipes_dir is not None:
        console.print(
            f"[cyan]Override pipeline recipe path (from --recipe-dir):[/] {custom_recipes_dir}"
        )
        return None

    env_dirs = _get_env_recipe_dirs()
    if env_dirs:
        console.print(
            f"[cyan]Pipeline recipe path (from ESOREX_PLUGIN_DIR):[/] {':'.join(str(d) for d in env_dirs)}"
        )
    else:
        # Find MATISSE recipe directory
        probe = find_matisse_recipe_dir(extra_candidates=[], verbose=False)

        if probe:
            console.print(f"[cyan]Pipeline recipes directory:[/] {probe.recipe_dir}")
            console.print(
                f"[cyan]Available MATISSE recipes:[/] {len(probe.matisse_recipes)} found"
            )
        else:
            console.print(
                "[bold red]⚠️ Error: No MATISSE recipes found.[/] "
                "Set ESOREX_PLUGIN_DIR or use --recipe-dir in doctor command."
            )
            return None
