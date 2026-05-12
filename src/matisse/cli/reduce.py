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
from matisse.core.tidyup import tidyup_path
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
    ALL = "ALL"


class FilterMode(str, Enum):
    VF = "vf"
    PF = "pf"
    JP = "jp"


PANEL_IO = "Input / Output"
PANEL_CORE = "Core Reduction"
PANEL_SELECTION = "Selection"
PANEL_RECIPE_ADVANCED = "Recipe Advanced (rarely changed)"
PANEL_CHECKS = "Checks / Diagnostics"
PANEL_LOGGING = "Logging / Reporting"


def _validate_filter_mode(value: str) -> str:
    """Validate and normalize a comma-separated list of filter modes."""
    normalized = value.strip().lower()
    if normalized == "none":
        return "none"

    allowed_values = {mode.value for mode in FilterMode}
    parsed_values = [part.strip().lower() for part in value.split(",") if part.strip()]

    if not parsed_values:
        raise typer.BadParameter(
            "Expected a comma-separated list using vf, pf, jp, or 'none'."
        )

    invalid_values = [part for part in parsed_values if part not in allowed_values]
    if invalid_values:
        allowed = ", ".join(mode.value for mode in FilterMode)
        invalid = ", ".join(invalid_values)
        raise typer.BadParameter(
            f"Invalid filter mode(s): {invalid}. Allowed values: {allowed}, or 'none'."
        )

    # Deduplicate while preserving user order.
    unique_values = list(dict.fromkeys(parsed_values))
    return ",".join(unique_values)


def _validate_filter_baseline(value: int | None) -> int | None:
    """Validate that filter_baseline is between 1 and 6 (MATISSE baselines)."""
    if value is None:
        return None
    if value not in range(1, 7):
        raise typer.BadParameter("Must be an integer between 1 and 6.")
    return value


def reduce(
    datadir: Path = typer.Option(
        Path.cwd(),
        "--data-dir",
        "-d",
        help="Directory containing raw MATISSE FITS files.",
        rich_help_panel=PANEL_IO,
    ),
    calibdir: Path | None = typer.Option(
        None,
        "--calib-dir",
        "-c",
        help="Calibration directory (default: same as raw).",
        rich_help_panel=PANEL_IO,
    ),
    resultdir: Path | None = typer.Option(
        None,
        "--result-dir",
        "-r",
        help="Directory to store reduction results (default: current).",
        rich_help_panel=PANEL_IO,
    ),
    nbcore: int = typer.Option(
        1,
        "--nbcore",
        "-n",
        help="Number of CPU cores to use.",
        rich_help_panel=PANEL_IO,
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing results.",
        rich_help_panel=PANEL_IO,
    ),
    resol: Resolution = typer.Option(
        Resolution.ALL,
        "--resol",
        help="Spectral resolution.",
        show_default="ALL",
        rich_help_panel=PANEL_CORE,
    ),
    custom_recipes_dir: Path | None = typer.Option(
        None,
        "--recipe-dir",
        help="Custom directory for MATISSE recipes (default: user esorex repository).",
        rich_help_panel=PANEL_RECIPE_ADVANCED,
    ),
    spectral_average: int = typer.Option(
        -1,
        "--spectral-average",
        help="Number of spectral channels to average to improve SNR (if -1 (default) apply: 7 for N band, 5 for L/M band).",
        rich_help_panel=PANEL_RECIPE_ADVANCED,
    ),
    vfactor_mode: bool = typer.Option(
        True,
        "--vfactor/--no-vfactor",
        help="Enable vfactor correction for VIS2 (L/M band only, recipe 2.0.1+).",
        rich_help_panel=PANEL_CORE,
    ),
    pfactor_mode: bool = typer.Option(
        True,
        "--pfactor/--no-pfactor",
        help="Enable pfactor correction for VIS2 (L/M band only, recipe 2.0.1+).",
        rich_help_panel=PANEL_CORE,
    ),
    filter_mode: str = typer.Option(
        "vf,pf,jp",
        "--filter-mode",
        callback=lambda value: _validate_filter_mode(value),
        help="Filter modes: vf (vfactor), pf (pfactor), jp (fringe jump), or none (recipe 2.0.1+ only).",
        rich_help_panel=PANEL_CORE,
    ),
    filter_baseline: int | None = typer.Option(
        None,
        "--filter-baseline",
        callback=lambda value: _validate_filter_baseline(value),
        help="Filter data by baseline index (1–6, recipe 2.0.1+ only)."
        + " Note: Indices follow the MATISSE data order. To filter multiple baselines, run the reduction for each index individually. Future updates may allow passing multiple indices and combining the resulting OIFITS files. "
        + "[dim][Default: no filtering.][/dim]",
        rich_help_panel=PANEL_CORE,
    ),
    skip_l: bool = typer.Option(
        False,
        "--skipL",
        help="Skip L band data.",
        rich_help_panel=PANEL_CORE,
    ),
    skip_n: bool = typer.Option(
        False,
        "--skipN",
        help="Skip N band data.",
        rich_help_panel=PANEL_CORE,
    ),
    tplid: str = typer.Option(
        "",
        "--tplid",
        help="Template ID to select.",
        rich_help_panel=PANEL_RECIPE_ADVANCED,
    ),
    tplstart: str = typer.Option(
        "",
        "--tplstart",
        help="Template start to select.",
        rich_help_panel=PANEL_RECIPE_ADVANCED,
    ),
    param_n: str = typer.Option(
        "/useOpdMod=TRUE",
        "--paramN",
        help="Recipe parameters for N band.",
        rich_help_panel=PANEL_RECIPE_ADVANCED,
    ),
    param_l: str = typer.Option(
        "/tartyp=57/useOpdMod=FALSE",
        "--paramL",
        help="Recipe parameters for L/M band.",
        rich_help_panel=PANEL_RECIPE_ADVANCED,
    ),
    check_blocks: bool = typer.Option(
        False,
        "--check-blocks",
        help="Check FITS files and the different pipeline blocks to be executed.",
        rich_help_panel=PANEL_CHECKS,
    ),
    check_calib: bool = typer.Option(
        False,
        "--check-cal",
        help="Check if calibration files already processed.",
        rich_help_panel=PANEL_CHECKS,
    ),
    check_files: bool = typer.Option(
        False,
        "--check-files",
        help="Check raw FITS files and their headers without running the pipeline.",
        rich_help_panel=PANEL_CHECKS,
    ),
    detailed_block: int | None = typer.Option(
        None,
        "--block-cal",
        help="Show calibration filenames attached to the given reduction block number.",
        rich_help_panel=PANEL_CHECKS,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose mode",
        rich_help_panel=PANEL_LOGGING,
    ),
    save_report: bool = typer.Option(
        False,
        "--save-report",
        help="Save pipeline summary tables as an SVG report in the result directory.",
        rich_help_panel=PANEL_LOGGING,
    ),
):
    """
    Run automatic reduction to produce uncalibrated oifits files and processed calibration maps if needed beforehand.

    It processes raw MATISSE data files and produces
    uncalibrated results. This pipeline is designed to run over multiple iterations
    in order to first reduce calibration files (flat, kappa matrix, shift map, etc.), and
    then science files using the calibration products obtained in previous iterations.
    All reduction steps are handled automatically by the pipeline, which determines which
    files need to be processed and in which order. The final products are stored in *reduced/*
    subdirectories within the specified result directory (current by default).

    We encourage users to run the pipeline using multiple cores (--nbcore) to speed up
    processing time. Intermediate products are stored in --result-dir/reduced and final
    OIFITS files are stored in --result-dir/reduced_OIFITS. If the pipeline is re-run with the same result directory,
    it will skip already processed files unless --overwrite is specified.
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
    console.print(
        f"[yellow]VFACTOR correction:[/] {'ON' if vfactor_mode else 'OFF'}, "
        + f"[yellow]PFACTOR correction:[/] {'ON' if pfactor_mode else 'OFF'}, "
        + f"[yellow]FILTER mode:[/] {filter_mode}, "
        + f"[yellow]FILTER baseline:[/] {filter_baseline if filter_baseline is not None else 'none'}"
    )
    console.print(f"[dim]Verbose:[/] {'ON' if not verbose else 'OFF'}")

    # --- 4. Resolve paths for core function ---
    dir_raw = str(datadir.resolve()) + "/"
    dir_calib = str(calibdir.resolve())
    dir_result = str(resultdir.resolve())

    # --- 5. Run pipeline and handle errors ---
    try:
        pipeline_summary = run_pipeline(
            dirRaw=dir_raw,
            dirResult=dir_result,
            dirCalib=dir_calib,
            nbCore=nbcore,
            resol="" if resol == Resolution.ALL else resol.value,
            paramL=param_l,
            paramN=param_n,
            overwrite=int(overwrite),
            skipL=skip_l,
            skipN=skip_n,
            tplstartsel=tplstart,
            tplidsel=tplid,
            spectralAverage=spectral_average,
            check_blocks=check_blocks,
            check_calib=check_calib,
            detailed_block=detailed_block,
            custom_recipes_dir=custom_recipes_dir,
            save_report_svg=save_report,
            inventory=check_files,
            vfactor_mode=vfactor_mode,
            pfactor_mode=pfactor_mode,
            filter_mode=filter_mode,
            filter_baseline=filter_baseline,
        )

        if not check_blocks and not check_calib and not check_files:
            tidyup_path(
                Path(dir_result) / "reduced",
                expected_oifits=pipeline_summary.get("expected_oifits"),
            )
            log.info(
                "[green][SUCCESS] Results saved to reduced/ (intermediate) and reduced_OIFITS/ (oifits)"
            )
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
