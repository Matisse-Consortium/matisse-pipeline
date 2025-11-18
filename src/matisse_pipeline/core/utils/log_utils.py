"""
Centralized Rich-based logging and console utilities for the MATISSE pipeline.
"""

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

# --- Detect test mode (pytest or typer CliRunner) ---
IS_TEST = "pytest" in sys.modules or "click.testing" in sys.modules

# --- Create a stable console ---
# In test mode, use a dummy stream to avoid ValueError on closed stderr
if IS_TEST:
    from io import StringIO

    _fake_stream = StringIO()
    console = Console(file=_fake_stream, force_terminal=False)
else:
    console = Console()

# --- Configure logging safely ---
if not logging.getLogger().hasHandlers():
    if IS_TEST:
        # Use a simple StreamHandler for pytest / typer tests
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            datefmt="%H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
    else:
        # Normal RichHandler for CLI
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            datefmt="%H:%M:%S",
            handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
        )

# --- Global project logger ---
log = logging.getLogger("matisse")


# Optional helper to switch verbosity dynamically
def set_verbosity(log, verbose: bool) -> None:
    """Adjust global log level based on verbosity flag."""
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    log.info(f"Log level set to {'DEBUG' if verbose else 'INFO'}.")


def section(title: str) -> None:
    console.print()
    console.rule(f"[bold cyan]{title}[/]")


def iteration_banner(iter_number: int):
    title = f"🔄 ITERATION {iter_number}"
    console.print()
    console.print(
        Panel.fit(
            f"[bold white]{title}[/]",
            border_style="bright_green",
            padding=(0, 5),
        ),
        justify="center",
    )


def get_detector_name(elt):
    hdr = elt["input"][0][2] if elt["input"] else {}
    return hdr.get("HIERARCH ESO DET CHIP NAME", "N/A") if hdr else "N/A"


def get_target_name(elt):
    hdr = elt["input"][0][2] if elt["input"] else {}
    return hdr.get("ESO OBS TARG NAME", "CAL FILE") if hdr else "N/A"


def show_calibration_status(listRedBlocks, console):
    """
    Display a minimal table: one line per calibration block,
    grouped by detector (AQUARIUS first, then HAWAII-2RG).
    """
    # detectors = ["AQUARIUS", "HAWAII-2RG"]
    bands = {"AQUARIUS": "N", "HAWAII-2RG": "L"}
    expected_tags = [
        "BADPIX",
        "NONLINEARITY",
        "OBS_FLATFIELD",
        "SHIFT_MAP",
        "KAPPA_MATRIX",
        "JSDC_CAT",
    ]

    table = Table(
        show_header=True,
        header_style="bold green",
        title_style="bold",
        expand=False,
        title="Calibration Summary",
    )

    table.add_column("Detector", justify="center", style="bold")
    table.add_column("Band", justify="center", style="magenta")
    table.add_column("Action", style="green")
    table.add_column("Block #", justify="center", style="dim")

    # One column per expected tags
    for tag in expected_tags:
        table.add_column(tag, justify="center")

    detector_map = {}
    for block in listRedBlocks:
        detector = get_detector_name(block)
        tags_present = {tag for _, tag in block["calib"]}
        detector_map.setdefault(detector, set()).update(tags_present)

    enriched_blocks = []
    nblock = 1
    for block in listRedBlocks:
        if not block:
            continue
        detector = get_detector_name(block)
        action = block.get("action", "")
        enriched_blocks.append((detector, block, action, nblock))
        nblock += 1

    # --- Sort by detectors
    # enriched_blocks.sort(key=lambda x: (x[2], x[0]))
    for detector, block, action, nblock in enriched_blocks:
        tags_present = {tag for _, tag in block["calib"]}

        row = [
            detector,
            bands.get(detector, "?"),
            action,
            str(nblock),
        ]

        for tag in expected_tags:
            if tag == "KAPPA_MATRIX" and detector == "AQUARIUS":
                row.append("–")
            else:
                row.append("✅" if tag in tags_present else "❌")
        table.add_row(*row)

    console.print()
    console.print(table, justify="center")


def show_blocs_status(listCmdEsorex, iterNumber, maxIter, listRedBlocks, check_blocks):
    """Print table containing the different block informations."""

    if listCmdEsorex == [] or iterNumber == maxIter:
        table = Table(
            title="\n- MATISSE final reduction summary -",
            show_header=True,
            header_style="bold magenta",
            title_style="bold cyan",
            expand=True,
        )

        table.add_column("TPL Start", style="cyan", no_wrap=True)
        table.add_column("Target", style="yellow")
        table.add_column("Tag", style="white")
        table.add_column("Detector", style="magenta")
        table.add_column("Action", style="green")
        table.add_column("Status", justify="center", style="bold")
        table.add_column("Message", style="dim")

        n_ok = n_skip = n_fail = 0

        listRedBlocks = sorted(
            listRedBlocks,
            key=lambda e: (
                e.get("action", ""),
                get_detector_name(e),
                get_target_name(e),
            ),
        )

        for elt in listRedBlocks:
            tplstart = elt.get("tplstart", "N/A")
            tag = elt["input"][0][1]
            detector = get_detector_name(elt)
            action = elt.get("action", "N/A")
            status = elt.get("status", 0)
            iteration = elt.get("iter", "?")
            target = get_target_name(elt)

            # Determine message and style
            if status == 1:
                msg = f"Completed (iter {iteration})"
                table.add_row(
                    tplstart, target, tag, detector, action, "✅ [green]OK[/]", msg
                )
                n_ok += 1
            elif status == -2:
                msg = "Ignored (check mode)"
                table.add_row(
                    tplstart, target, tag, detector, action, "[cyan]SKIP[/]", msg
                )
                n_skip += 1
            else:
                if elt["action"] == "NO-ACTION":
                    msg = "Data not taken into account by the Pipeline"
                    table.add_row(
                        tplstart, target, tag, detector, action, "❌ [red]FAIL[/]", msg
                    )
                    n_fail += 1
                else:
                    if check_blocks:
                        msg = "Reduction Block not processed - Check mode"
                    else:
                        msg = "Reduction Block not processed - Missing calibration"
                    table.add_row(
                        tplstart,
                        target,
                        tag,
                        detector,
                        action,
                        "⚠ [yellow]SKIP[/]",
                        msg,
                    )
                    n_skip += 1

        console.print(table)

        # --- Global pipeline statistics ---
        console.print(
            Panel.fit(
                f"[green]Successful:[/] {n_ok}  |  [yellow]Skipped:[/] {n_skip}  |  [red]Failed:[/] {n_fail}  |  [cyan]Total:[/] {len(listRedBlocks)}",
                title="[bold]Global Pipeline Statistics[/]",
                border_style="cyan",
            ),
            justify="center",
        )

        console.rule(style="dim")

        # Break logic (to be called inside a loop)
        return True  # signal to break the loop
    return False
