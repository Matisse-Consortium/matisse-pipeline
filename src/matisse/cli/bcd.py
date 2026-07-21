"""BCD (Beam Commuting Device) sub-commands for magic numbers computation and correction."""

import logging
import shutil
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import typer
from astropy.io import fits
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
from matisse.core.bcd.correction import apply_bcd_corrections
from matisse.core.bcd.merge import find_sci_filename, merge_by_tpl_start, remove_bcd
from matisse.core.bcd.visualization import (
    compare_bcd_corrections,
    plot_poly_corrections_results,
)
from matisse.core.utils.log_utils import console, log


class BCDMode(str, Enum):
    IN_IN = "IN_IN"
    OUT_IN = "OUT_IN"
    IN_OUT = "IN_OUT"
    ALL = "ALL"


class SpectralBand(str, Enum):
    LM = "LM"
    N = "N"


# Create main BCD command group
app = typer.Typer(help="BCD (Beam Commuting Device) correction tools")


@app.command(name="compute")
def compute(
    input_dirs: list[Path] | None = typer.Argument(
        None,
        help="One or more directories containing OIFITS files (e.g. reduced_OIFITS). Required unless using --results-dir to plot existing results.",
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
    output_dir: Path = typer.Option(
        Path("bcd_calibration_results"),
        "--output-dir",
        "-o",
        help="Output directory for correction files.",
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
        "--chopping",
        help="Use chopped files.",
    ),
    correlated_flux: bool = typer.Option(
        False,
        "--correlated-flux",
        help="Filter for correlated flux.",
    ),
    results_dir: Path | None = typer.Option(
        None,
        "--results-dir",
        help="Existing results directory (with CSV files) to plot magic numbers without recomputing.",
    ),
    plot: bool = typer.Option(
        False,
        "--plot",
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
    # Setup logging level based on verbose flag
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger("matisse").setLevel(log_level)

    output_dir = Path(output_dir)
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
                transient=True,
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


@app.command(name="apply")
def apply(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing OIFITS files (e.g., *_OIFITS/).",
        exists=True,
    ),
    corrections_dir: Path | None = typer.Argument(
        None,
        help="Directory containing BCD correction files (default: bundled master calibration).",
    ),
    chopping: bool = typer.Option(
        False,
        "--chopping",
        help="Use chopped files.",
    ),
    merge: bool = typer.Option(
        False,
        "--merge",
        "-m",
        help="Merge corrected BCD modes into a single OIFITS file (with reordering by BCD mode).",
    ),
    plot: bool = typer.Option(
        False,
        "--plot",
        "-p",
        help="Generate diagnostic plots for the applied corrections.",
    ),
    split_chopping: bool = typer.Option(
        False,
        "--split-chopping",
        help="When merging, keep chopped and unchopped files separate instead of merging them together.",
    ),
    sub_band: list[str] | None = typer.Option(
        None,
        "--sub-band",
        help=(
            "Sub-band for quality metrics. Use either a predefined band "
            "(--sub-band L|M|N) or an explicit range in microns "
            "(--sub-band 3.2,4.0). "
            "Default: full correction range."
        ),
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed metrics tables for each file (useful for debugging individual corrections).",
    ),
) -> None:
    """
    Apply stored BCD magic numbers to the observations (only VIS2 are affected).

    By default, shows a summary table of all processed files. Use --verbose to see
    detailed metrics for each file individually.
    """
    parsed_sub_band: str | list[float] | None = None
    if sub_band:
        if len(sub_band) == 1:
            token = sub_band[0].strip()
            if "," in token:
                parts = [part.strip() for part in token.split(",")]
                if len(parts) != 2:
                    raise typer.BadParameter(
                        "Range format must contain exactly two values: low,high"
                    )
                try:
                    parsed_sub_band = [float(parts[0]), float(parts[1])]
                except ValueError as exc:
                    raise typer.BadParameter(
                        "Range values must be numeric microns, e.g. 3.2,4.0"
                    ) from exc
            else:
                parsed_sub_band = token.upper()
        elif len(sub_band) == 2:
            try:
                parsed_sub_band = [float(sub_band[0]), float(sub_band[1])]
            except ValueError as exc:
                raise typer.BadParameter(
                    "When providing two --sub-band values, both must be numeric"
                ) from exc
        else:
            raise typer.BadParameter(
                "Use --sub-band L|M|N or exactly two values for a range"
            )

    apply_bcd_corrections(
        input_dir,
        corrections_dir,
        chopping=chopping,
        merge=merge,
        verbose=verbose,
        plot=plot,
        split_chopping=split_chopping,
        sub_band=parsed_sub_band,
    )
    raise typer.Exit(code=0)


@app.command(name="remove")
def remove(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing OIFITS files (e.g., *_OIFITS/).",
        exists=True,
    ),
    chopping: bool = typer.Option(
        False,
        "--chopping",
        help="Use chopped files.",
    ),
    cal: bool = typer.Option(
        False,
        "--cal",
        help="Process CAL files (by default only SCI files are processed).",
    ),
    band: SpectralBand = typer.Option(SpectralBand.LM, "--band", help="Spectral band."),
) -> None:
    """
    Remove BCD effects in OIFITS files from the specified directory.

    All files will be renamed with the suffix _noBCD and the BCD ordering will follow
    the OUT_OUT convention. This is required to prepare files for next steps of the
    pipeline (e.g., calibration with genoca).
    """
    list_scivis = find_sci_filename(
        input_dir, chopping=chopping, band=band.value, include_cal=cal
    )
    log.info(f"List of science files to process: {list_scivis}")

    for file in list_scivis:
        hdu = fits.open(file)
        remove_bcd(hdu, save=True)
        hdu.close()

    list_bcd_removed = sorted(
        f for f in Path(input_dir).glob("*_noBCD.fits") if not f.name.startswith("._")
    )
    nobcd_dir = input_dir.parent / f"{input_dir.name}_noBCD"
    nobcd_dir.mkdir(parents=True, exist_ok=True)
    for file in list_bcd_removed:
        target_path = nobcd_dir / file.name
        file.rename(target_path)
        log.debug(f"Moved {file.name} to {target_path}")

    for file in list_scivis:
        if "OUT_OUT" in file.name:
            target_path = nobcd_dir / file.name
            shutil.copy2(file, target_path)
            log.debug(f"Copied {file.name} to {target_path}")


@app.command(name="compare")
def compare(
    data_dir: Path = typer.Argument(
        ...,
        help="Directory containing OIFITS files (e.g., *_OIFITS/).",
        exists=True,
    ),
    corrections_dir: Path | None = typer.Option(
        None,
        "--corrections-dir",
        help="Directory with correction CSVs (shades calibration windows on plots).",
        exists=True,
    ),
    plot: bool = typer.Option(
        False,
        "--plot",
        "-p",
        help="Display interactive comparison figures (disabled by default; PDFs are always saved).",
    ),
) -> None:
    """
    Compare BCD corrections across all modes for each TPL start.

    Auto-detects corrected files in data_dir, groups them by TPL start,
    and plots V² for all 4 BCD positions. If a merged file is found,
    it is overlaid in black.

    Example:
        matisse bcd compare /data/2026-01-15_OIFITS_bcd_corr/
    """
    try:
        compare_bcd_corrections(
            data_dir=data_dir,
            corrections_dir=corrections_dir,
            show_plot=plot,
        )
    except FileNotFoundError as e:
        console.print(f"[bold red]✗[/bold red] {e}", style="red")
        raise typer.Exit(code=1) from e
    except ValueError as e:
        console.print(f"[bold red]✗[/bold red] {e}", style="red")
        raise typer.Exit(code=1) from e


@app.command(name="merge")
def merge(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing OIFITS files (e.g., *_OIFITS/).",
        exists=True,
    ),
    combine_chopping: bool = typer.Option(
        True,
        "--combine-chopping",
        help="When merging, combine chopped and unchopped files into one final OIFITS file.",
    ),
) -> None:
    """
    Merge BCD modes into a single OIFITS file with OUT_OUT ordering.

    This is useful for preparing files for next steps of the pipeline (e.g., calibration with genoca).
    """
    merge_by_tpl_start(
        str(input_dir),
        save=True,
        output_dir=input_dir,
        separate_chopping=combine_chopping,
    )
    return None


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
        if len(std[std == 0]) == len(std):
            aver_str = [f"{aver[i]:.2f}" for i in range(len(aver))]
        else:
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
