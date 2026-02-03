"""Compute BCD magic numbers from calibrator observations."""

import logging
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import typer
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from matisse.cli.reduce import Resolution
from matisse.core.bcd import BCDConfig, compute_bcd_corrections
from matisse.core.bcd.visualization import plot_poly_corrections_results
from matisse.core.utils.log_utils import console, log


class BCDMode(str, Enum):
    IN_IN = "IN_IN"
    OUT_IN = "OUT_IN"
    IN_OUT = "IN_OUT"
    ALL = "ALL"


class SpectralBand(str, Enum):
    LM = "LM"
    N = "N"


def compute_magic_numbers(
    input_dirs: list[Path] | None = typer.Argument(
        None,
        help="One or more directories containing OIFITS files (e.g., /data/2019*/*_OIFITS). Required unless using --results-dir to plot existing results.",
        exists=True,
    ),
    bcd_mode: BCDMode = typer.Option(
        BCDMode.IN_IN,
        "--bcd-mode",
        "-b",
        help="BCD configuration to compute.",
    ),
    band: SpectralBand = typer.Option(
        SpectralBand.LM,
        "--band",
        help="Spectral band.",
    ),
    resolution: Resolution = typer.Option(
        Resolution.LOW,
        "--resol",
        help="Spectral resolution.",
    ),
    extension: str = typer.Option(
        "OI_VIS2",
        "--extension",
        "-e",
        help="OIFITS extension type (OI_VIS or OI_VIS2).",
    ),
    prefix: str = typer.Option(
        "MN2025",
        "--prefix",
        "-p",
        help="Prefix of the npy files to store magic numbers.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for correction files (default: <prefix>_results in current directory).",
    ),
    wavelength_range: list[float] = typer.Option(
        [3.3, 3.8],
        "--wavelength-range",
        help="Wavelength range in microns (for averaging computing).",
    ),
    poly_order: int = typer.Option(
        1,
        "--poly-order",
        help="Polynome order to be fitted.",
    ),
    tau0_min: float | None = typer.Option(
        None,
        "--tau0-min",
        help="Minimum coherence time in ms (reject files below this threshold).",
    ),
    chopping: bool = typer.Option(
        False,
        "--chopping/--no-chopping",
        help="Use chopping files.",
    ),
    correlated_flux: bool = typer.Option(
        False,
        "--correlated-flux/--no-correlated-flux",
        help="Filter for correlated flux.",
    ),
    results_dir: Path | None = typer.Option(
        None,
        "--results-dir",
        help="Existing results directory (with CSV files) to plot magic numbers without recomputing.",
    ),
    plot: bool = typer.Option(
        True,
        "--plot/--no-plot",
        help="Generate diagnostic plots.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging.",
    ),
) -> None:
    """
    Compute BCD magic numbers from MATISSE calibrator observations.

    This command processes pairs of OUT_OUT and BCD configuration files
    to compute instrumental corrections (magic numbers) that can be applied
    to science observations. The BCD (Beam Commuting Device) allows us to swap
    the VLTI beams, interchange the peak fringe position, and thus achieve
    better closure-phase correction and improved SNR.
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Set default output_dir based on prefix if not provided
    if output_dir is None:
        output_dir = Path.cwd() / f"{prefix.lower()}_results"
        output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which modes to process
    modes_to_process = (
        [BCDMode.IN_IN, BCDMode.OUT_IN, BCDMode.IN_OUT]
        if bcd_mode == BCDMode.ALL
        else [bcd_mode]
    )

    # Display header
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]BCD Magic Numbers Computation[/bold cyan]\n"
            f"Mode: [yellow]{bcd_mode.value}[/yellow] | "
            f"Band: [yellow]{band.value}[/yellow] | "
            f"Resolution: [yellow]{resolution.value}[/yellow]",
            border_style="cyan",
        )
    )

    log.info(
        f"Processing {len(input_dirs or [])} input directories with {len(modes_to_process)} mode(s)"
    )

    # Create configuration
    try:
        # Implicit plotting mode: if no input_dirs but a results_dir is provided
        if not input_dirs and results_dir is not None:
            log.info(
                "No input dirs provided; plotting existing BCD corrections from %s (mode=%s)",
                results_dir,
                bcd_mode.value,
            )
            _plot_existing_or_exit(results_dir, bcd_mode.value, plot)
            return

        if not input_dirs:
            console.print(
                "[bold red]✗[/bold red] Missing input directories (required unless using --results-dir)",
                style="red",
            )
            raise typer.Exit(code=1)

        # Process each mode
        all_results = {}
        for current_mode in modes_to_process:
            log.info(f"Processing mode: {current_mode.value}")

            config = BCDConfig(
                bcd_mode=current_mode.value,
                prefix=prefix.upper(),
                band=band.value,
                resolution=resolution.value,
                extension=extension.upper(),
                output_dir=output_dir,
                wavelength_low=wavelength_range[0] * 1e-6,  # Convert to meters
                wavelength_high=wavelength_range[1] * 1e-6,
                correlated_flux=correlated_flux,
                poly_order=poly_order,
                tau0_min=tau0_min,
            )

            # Compute corrections with progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Computing {current_mode.value} corrections...",
                    total=None,
                )

                results = compute_bcd_corrections(
                    folders=[str(d) for d in (input_dirs or [])],
                    config=config,
                    chopping=chopping,
                    show_plots=plot,
                    progress=progress,
                    task_id=task,
                )

                progress.update(task, completed=True, total=1)

            all_results[current_mode.value] = (results, config)

        # Display results for all modes
        for mode_name, (results, config) in all_results.items():
            console.print()
            console.print(f"[bold cyan]Results for mode: {mode_name}[/bold cyan]")
            _display_results(results, output_dir, config)

        # Success message
        console.print()
        console.print(
            f"[bold green]✓[/bold green] Successfully processed "
            f"{len(all_results)} mode(s)"
        )

        if plot:
            console.print("[dim]Diagnostic plots displayed.[/dim]")
            plt.show()

    except FileNotFoundError as e:
        log.error(f"File not found: {e}")
        raise typer.Exit(code=1) from e
    except ValueError as e:
        console.print(f"[bold red]✗[/bold red] Configuration error: {e}", style="red")
        log.error(f"Data error: {e}")
        raise typer.Exit(code=1) from e


def _plot_existing_or_exit(target_dir: Path, bcd_mode: str, show: bool) -> None:
    """Helper to plot existing CSVs with consistent error handling."""
    try:
        fig = plot_poly_corrections_results(output_dir=target_dir, bcd_mode=bcd_mode)
        if show:
            plt.show()
        else:
            plt.close(fig)
    except FileNotFoundError as exc:
        console.print(
            f"[bold red]✗[/bold red] Missing CSV files in {target_dir}: {exc}",
            style="red",
        )
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover - defensive
        console.print(
            f"[bold red]✗[/bold red] Failed to plot corrections: {exc}", style="red"
        )
        log.exception("Failed to plot corrections")
        raise typer.Exit(code=1) from exc


def _display_results(
    results: dict,
    output_dir: Path,
    config: BCDConfig,
) -> None:
    """Display computation results in a formatted table."""

    # Averaged magic numbers
    if results.get("corrections") is not None:
        aver = results["corrections"]["mean_over_files"]
        std = results["corrections"]["std_over_files"]
        aver_str = [f"{aver[i]:.2f}±{std[i]:.2f}" for i in range(len(aver))]
        baseline_pairs = results["corrections"]["baseline_pairs"]
        baseline_names = results["corrections"]["baseline_names"]

    aff_bl1 = baseline_pairs[0][0]
    aff_bl2 = baseline_pairs[1][0]

    rev_aff_bl1 = baseline_pairs[0][1]
    rev_aff_bl2 = baseline_pairs[1][1]
    # Files table
    table = Table(
        show_header=True,
        header_style="bold white",
        border_style="cyan",
    )
    table.add_column("Filename", style="white")
    for i in range(6):
        style = "white"
        if i in (aff_bl1, aff_bl2):
            style = "green"
        elif i in (rev_aff_bl1, rev_aff_bl2):
            style = "yellow"
        table.add_column(f"Aver. MN BL {i + 1}", style=style)

    # CSV output file
    if results.get("csv_file") is not None:
        table.add_row(results["csv_file"].name, *aver_str)

    # Already added CSV above

    console.print(table)

    # Summary info
    console.print(f"[bold]Output directory:[/bold] {output_dir}")
    console.print(
        "Affected baselines: "
        + f"{aff_bl1 + 1} [green]({baseline_names[aff_bl1]})[/green]"
        + " [white]<->[/white] "
        + f"{rev_aff_bl1 + 1} [yellow]({baseline_names[rev_aff_bl1]})[/yellow]"
        + " ; "
        + f"{aff_bl2 + 1} [green]({baseline_names[aff_bl2]})[/green]"
        + " [white]<->[/white] "
        + f"{rev_aff_bl2 + 1} [yellow]({baseline_names[rev_aff_bl2]})[/yellow]"
    )
    console.print(
        f"[bold]BCD mode:[/bold] {config.bcd_mode} | "
        f"[bold]Band:[/bold] {config.band} | "
        f"[bold]Resolution:[/bold] {config.resolution}"
    )
