"""Compute BCD magic numbers from calibrator observations."""

import logging
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

from matisse_pipeline.core.bcd import BCDConfig, compute_bcd_corrections
from matisse_pipeline.core.utils.log_utils import console, log


def compute_magic_numbers(
    input_dirs: list[Path] = typer.Argument(
        ...,
        help="One or more directories containing OIFITS files (e.g., /data/2019*/*_OIFITS).",
        exists=True,
    ),
    bcd_mode: str = typer.Option(
        "IN_IN",
        "--bcd-mode",
        "-b",
        help="BCD configuration to compute (IN_IN, OUT_IN, IN_OUT).",
    ),
    band: str = typer.Option(
        "LM",
        "--band",
        help="Spectral band (LM or N).",
    ),
    resolution: str = typer.Option(
        "LOW",
        "--resol",
        help="Spectral resolution (LOW or HIGH).",
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
    output_dir: Path = typer.Option(
        Path.cwd(),
        "--output-dir",
        "-o",
        help="Output directory for correction files (default: current).",
    ),
    wavelength_low: float = typer.Option(
        3.3,
        "--wavelength-low",
        help="Lower wavelength bound in microns.",
    ),
    wavelength_high: float = typer.Option(
        3.8,
        "--wavelength-high",
        help="Upper wavelength bound in microns.",
    ),
    poly_order: int = typer.Option(
        2,
        "--poly-order",
        help="Polynome order to be fitted.",
    ),
    chopping: bool = typer.Option(
        True,
        "--chopping/--no-chopping",
        help="Use chopping files.",
    ),
    correlated_flux: bool = typer.Option(
        False,
        "--correlated-flux/--no-correlated-flux",
        help="Filter for correlated flux.",
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

    # Display header
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]BCD Magic Numbers Computation[/bold cyan]\n"
            f"Mode: [yellow]{bcd_mode}[/yellow] | "
            f"Band: [yellow]{band}[/yellow] | "
            f"Resolution: [yellow]{resolution}[/yellow]",
            border_style="cyan",
        )
    )

    log.info(f"Processing {len(input_dirs)} input directories")

    # Create configuration
    try:
        config = BCDConfig(
            bcd_mode=bcd_mode.upper(),
            prefix=prefix.upper(),
            band=band.upper(),
            resolution=resolution.upper(),
            extension=extension.upper(),
            output_dir=output_dir,
            wavelength_low=wavelength_low * 1e-6,  # Convert to meters
            wavelength_high=wavelength_high * 1e-6,
            correlated_flux=correlated_flux,
            poly_order=poly_order,
        )
    except ValueError as e:
        console.print(f"[bold red]✗[/bold red] Configuration error: {e}", style="red")
        raise typer.Exit(code=1) from e

    # Compute corrections with progress tracking
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Computing {bcd_mode} corrections...",
                total=None,
            )

            results = compute_bcd_corrections(
                folders=[str(d) for d in input_dirs],
                config=config,
                chopping=chopping,
                show_plots=plot,
                progress=progress,
                task_id=task,
            )

            progress.update(task, completed=True, total=1)

        # Display results table
        _display_results(results, output_dir, config)

        # Success message
        console.print()
        console.print(
            f"[bold green]✓[/bold green] Successfully processed "
            f"{results['n_files']} file pairs"
        )

        if plot:
            console.print("[dim]Diagnostic plots displayed.[/dim]")
            plt.show()

    except FileNotFoundError as e:
        log.error(f"File not found: {e}")
        raise typer.Exit(code=1) from e
    except ValueError as e:
        console.print(f"[bold red]✗[/bold red] Invalid  {e}", style="red")
        log.error(f"Data error: {e}")
        raise typer.Exit(code=1) from e


def _display_results(
    results: dict,
    output_dir: Path,
    config: BCDConfig,
) -> None:
    """Display computation results in a formatted table."""
    console.print()

    # Files table
    table = Table(
        title="Generated Files",
        show_header=True,
        header_style="bold magenta",
        border_style="cyan",
    )
    table.add_column("Type", style="cyan", width=15)
    table.add_column("Filename", style="green")
    table.add_column("Aver. magic numbers")

    # Wavelength file
    table.add_row("Wavelengths", results["wavelength_file"].name, "-")

    aver_over_file = results["corrections"]["mean"].mean(axis=0)
    # Correction files
    for i, f in enumerate(results["correction_files"]):
        table.add_row(f"Baseline {i}", f.name, str(round(aver_over_file[i], 2)))

    console.print(table)

    # Summary info
    console.print()
    console.print(f"[bold]Output directory:[/bold] {output_dir}")
    console.print(
        f"[bold]BCD mode:[/bold] {config.bcd_mode} | "
        f"[bold]Band:[/bold] {config.band} | "
        f"[bold]Resolution:[/bold] {config.resolution}"
    )
