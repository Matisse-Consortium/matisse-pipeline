from pathlib import Path

import typer

from matisse_pipeline.core.tidyup import tidyup_path
from matisse_pipeline.core.utils.log_utils import log, set_verbosity


def format_results(
    directory: Path = typer.Argument(
        ..., help="Directory containing reduced fits files."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging."
    ),
):
    """
    Format reduced data into OIFITS files using FITS metadata.

    This command recursively scans the Iter*/ directories for science products (SCI, CAL)
    and moves them into corresponding Iter*_OIFITS/ folders.
    Each file is then renamed based on its observation start time, spectral resolution,
    BCD configuration, and other relevant FITS metadata.
    """
    set_verbosity(log, verbose)

    if not directory.exists():
        log.error(f"❌ Directory {directory} not found.")
        raise typer.Exit(1)

    log.info(f"Starting MATISSE OIFITS tidyup in {directory}...")
    tidyup_path(directory)
    log.info("✅ Tidyup complete.")
