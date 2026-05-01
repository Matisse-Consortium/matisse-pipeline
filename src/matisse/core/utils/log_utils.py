"""
Centralized Rich-based logging and console utilities for the MATISSE pipeline.
"""

import logging
import sys
from pathlib import Path

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

# --- Report console: fixed width, records output for SVG export ---
REPORT_WIDTH = 140
_report_console = Console(
    record=True, width=REPORT_WIDTH, force_terminal=True, file=open("/dev/null", "w")
)

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


def show_calibration_status(listRedBlocks, console, detailed_block: int | None = None):
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

    table.add_column("TPL Start", justify="center", style="bold")
    table.add_column("Detector", justify="center", style="bold")
    table.add_column("Band", justify="center", style="magenta")
    table.add_column("Action", style="green")
    table.add_column("#", justify="center", style="dim")

    # One column per expected tags
    for tag in expected_tags:
        table.add_column(tag, justify="center")

    detector_map: dict[str, set[str]] = {}
    for block in listRedBlocks:
        detector = get_detector_name(block)
        tags_present = {tag for _, tag in block["calib"]}
        detector_map.setdefault(detector, set()).update(tags_present)

    enriched_blocks = []
    nblock = 1
    for block in listRedBlocks:
        if not block:
            continue
        tplstart = block["tplstart"]
        detector = get_detector_name(block)
        action = block.get("action", "")
        enriched_blocks.append((tplstart, detector, block, action, nblock))
        nblock += 1

    # --- Sort by detectors
    # enriched_blocks.sort(key=lambda x: (x[2], x[0]))
    for tplstart, detector, block, action, nblock in enriched_blocks:
        tags_present = {tag for _, tag in block["calib"]}

        row = [
            tplstart.split(".")[0],
            detector,
            bands.get(detector, "?"),
            action,
            str(nblock),
        ]

        not_required_cal_dict = {
            "ACTION_MAT_EST_FLAT": [
                "OBS_FLATFIELD",
                "SHIFT_MAP",
                "KAPPA_MATRIX",
                "JSDC_CAT",
            ],
            "ACTION_MAT_EST_SHIFT": [
                "SHIFT_MAP",
                "KAPPA_MATRIX",
                "JSDC_CAT",
            ],
            "ACTION_MAT_EST_KAPPA": [
                "KAPPA_MATRIX",
                "JSDC_CAT",
            ],
            "ACTION_MAT_RAW_ESTIMATES": [],
        }

        for tag in expected_tags:
            if tag == "KAPPA_MATRIX" and detector == "AQUARIUS":
                row.append("–")
            elif tag in not_required_cal_dict[action] and action:
                row.append("–")
            else:
                row.append("✅" if tag in tags_present else "❌")
        table.add_row(*row)

    console.print()
    console.print(table, justify="center")
    _report_console.print()
    _report_console.print(table, justify="center")

    if detailed_block is None:
        return

    if not enriched_blocks:
        console.print(
            "[yellow]No reduction blocks available for calibration details.[/]"
        )
        return

    if detailed_block < 1 or detailed_block > len(enriched_blocks):
        console.print(
            f"[yellow]Block #{detailed_block} is out of range (available: 1-{len(enriched_blocks)}).[/]"
        )
        return

    _, _, block, _, _ = enriched_blocks[detailed_block - 1]

    detail_table = Table(
        title=f"Calibrations for block #{detailed_block}",
        show_header=True,
        header_style="bold cyan",
        expand=False,
    )
    detail_table.add_column("Tag", style="magenta")
    detail_table.add_column("File", style="white")
    detail_table.add_column("Path", style="dim")

    if not block.get("calib"):
        detail_table.add_row("—", "No calibration files", "")
    else:
        for filepath, tag in block["calib"]:
            path = Path(filepath)
            detail_table.add_row(tag, path.name, str(path.parent))

    console.print(detail_table, justify="center")
    _report_console.print(detail_table, justify="center")


def show_blocs_status(listCmdEsorex, listRedBlocks, check_blocks):
    """Print table containing the different block informations."""

    if listCmdEsorex == []:
        table = Table(
            title="\n- MATISSE final reduction summary -",
            show_header=True,
            header_style="bold magenta",
            title_style="bold cyan",
            expand=True,
        )

        table.add_column("#", style="bold white", no_wrap=True, justify="right")
        table.add_column("TPL Start", style="cyan", no_wrap=True)
        table.add_column("Target", style="yellow")
        table.add_column("Tag", style="white")
        table.add_column("Band", style="magenta")
        table.add_column("Resol", style="white")
        table.add_column("Action", style="green")
        table.add_column("Status", justify="center", style="bold")
        table.add_column("Message", style="dim")

        n_ok = n_skip = n_fail = 0

        # Attach original block number before sorting
        indexed_blocks = [(i + 1, elt) for i, elt in enumerate(listRedBlocks)]
        for block_num, elt in indexed_blocks:
            tplstart = elt.get("tplstart", "N/A")
            tplstart = tplstart.split(".")[0]
            tag = elt["input"][0][1]
            detector = get_detector_name(elt)
            if detector == "AQUARIUS":
                band = "N"
            else:
                band = "LM"

            action = elt.get("action", "N/A")
            action = action.replace("ACTION_", "")
            status = elt.get("status", 0)
            hdr = elt["input"][0][2] if elt["input"] else {}
            if detector == "AQUARIUS":
                resol = hdr.get("HIERARCH ESO INS DIN NAME", "N/A") if hdr else "N/A"
            elif detector == "HAWAII-2RG":
                resol = hdr.get("HIERARCH ESO INS DIL NAME", "N/A") if hdr else "N/A"
            else:
                resol = "N/A"
            # iteration = elt.get("iter", "?")
            target = get_target_name(elt)

            # Determine message and style
            if status == 1:
                msg = "Completed"
                table.add_row(
                    str(block_num),
                    tplstart,
                    target,
                    tag,
                    band,
                    resol,
                    action,
                    "✅ [green]OK[/]",
                    msg,
                )
                n_ok += 1
            elif status == -2:
                msg = "To be processed (check mode)"
                table.add_row(
                    str(block_num),
                    tplstart,
                    target,
                    tag,
                    band,
                    resol,
                    action,
                    "[cyan]SKIP[/]",
                    msg,
                )
                n_skip += 1
            elif status == -1:
                error_msg = elt.get("error_msg", "") or "Esorex execution failed"
                table.add_row(
                    str(block_num),
                    tplstart,
                    target,
                    tag,
                    band,
                    resol,
                    action,
                    "❌ [red]FAIL[/]",
                    error_msg,
                )
                n_fail += 1
            else:
                if elt["action"] == "NO-ACTION":
                    msg = "Data not taken into account by the Pipeline"
                    table.add_row(
                        str(block_num),
                        tplstart,
                        target,
                        tag,
                        band,
                        resol,
                        action,
                        "❌ [red]FAIL[/]",
                        msg,
                    )
                    n_fail += 1
                else:
                    if check_blocks:
                        msg = "Missing calibration (Check mode)"
                    else:
                        msg = "Missing calibration"
                    table.add_row(
                        str(block_num),
                        tplstart,
                        target,
                        tag,
                        band,
                        resol,
                        action,
                        "⚠ [yellow]SKIP[/]",
                        msg,
                    )
                    n_skip += 1

        console.print(table)

        # --- Global pipeline statistics ---
        stats_panel = Panel.fit(
            f"[green]Successful:[/] {n_ok}  |  [yellow]Skipped:[/] {n_skip}  |  [red]Failed:[/] {n_fail}  |  [cyan]Total:[/] {len(listRedBlocks)}",
            title="[bold]Global Pipeline Statistics[/]",
            border_style="cyan",
        )
        console.print(stats_panel, justify="center")
        console.rule(style="dim")

        # Mirror to report console
        _report_console.print(table)
        _report_console.print(stats_panel, justify="center")
        _report_console.rule(style="dim")

        # Break logic (to be called inside a loop)
        return True  # signal to break the loop
    return False


def show_files_inventory(dirRaw):
    """Print a dfits-like inventory table of raw FITS files.

    Mirrors the output of:
      dfits *.fits | fitsort obs.targ.name tpl.start det.chip.name \\
          dpr.type ins.bcd1.name ins.bcd2.name iss.chop.st pro.catg
    """
    _HDR = {
        "TARGET": "HIERARCH ESO OBS TARG NAME",
        "TPL START": "HIERARCH ESO TPL START",
        # CHIP column replaced by BAND below
        "BAND": "HIERARCH ESO DET CHIP NAME",  # We'll map this to LM/N
        "DPR TYPE": "HIERARCH ESO DPR TYPE",
        "BCD1": "HIERARCH ESO INS BCD1 NAME",
        "BCD2": "HIERARCH ESO INS BCD2 NAME",
        "CHOP": "HIERARCH ESO ISS CHOP ST",
        "PRO CATG": "HIERARCH ESO PRO CATG",
    }

    from astropy.io import fits

    table = Table(
        title="\n- MATISSE files inventory -",
        show_header=True,
        header_style="bold magenta",
        title_style="bold cyan",
        expand=True,
    )

    table.add_column("#", style="bold white", no_wrap=True, justify="right")
    table.add_column("File", no_wrap=True)
    for col in _HDR:
        table.add_column(col, style="white")

    list_files = sorted(
        p for p in Path(dirRaw).glob("*.fits") if not p.name.startswith("._")
    )
    skipped_invalid = 0

    for i, filepath in enumerate(list_files):
        try:
            hdr = fits.getheader(filepath, 0)
        except Exception as err:
            skipped_invalid += 1
            log.warning(f"Skipping invalid FITS file in inventory: {filepath} ({err})")
            continue
        chip = hdr.get("HIERARCH ESO DET CHIP NAME", "")
        dpr_type = hdr.get(_HDR["DPR TYPE"], "")
        pro_catg = hdr.get(_HDR["PRO CATG"], "")
        # Map chip to band
        if "AQUARIUS" in chip:
            band = "N"
            if dpr_type in (
                "STD",
                "OBJECT",
                "STD,RMNREC",
                "OBJECT,RMNREC",
            ) or pro_catg in ("TARGET_RAW_INT", "CALIB_RAW_INT"):
                row_style = "yellow"
            else:
                row_style = "dim yellow"
        elif "HAWAII" in chip:
            band = "LM"
            if dpr_type in (
                "STD",
                "OBJECT",
                "STD,RMNREC",
                "OBJECT,RMNREC",
            ) or pro_catg in ("TARGET_RAW_INT", "CALIB_RAW_INT"):
                row_style = ""
            else:
                row_style = "dim"
        else:
            band = "–"
            row_style = "dim cyan"

        # Compose values, replacing CHIP with BAND
        values = []
        for key in _HDR:
            if key == "BAND":
                values.append(band)
            else:
                values.append(hdr.get(_HDR[key], "–"))
        table.add_row(str(i), Path(filepath).name, *values, style=row_style)

    console.print(table)
    _report_console.print(table)
    if skipped_invalid:
        console.print(
            f"[yellow]Skipped {skipped_invalid} invalid FITS file(s) in inventory.[/]"
        )
        _report_console.print(
            f"Skipped {skipped_invalid} invalid FITS file(s) in inventory."
        )
    return table


def save_report(output_dir: str | Path) -> Path | None:
    """Export the recorded report console output as an SVG file.

    The SVG uses a fixed width (REPORT_WIDTH columns) so the layout
    is consistent regardless of the user's terminal size.

    Returns the path to the saved file, or None if nothing was recorded.
    """
    svg_text = _report_console.export_svg(title="MATISSE Pipeline Report")
    if not svg_text.strip():
        return None

    output_path = Path(output_dir) / "matisse_report.svg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg_text, encoding="utf-8")
    log.info(f"Pipeline report saved to [magenta]{output_path}[/magenta]")
    return output_path


def parse_esorex_missing_files(log_path: str) -> list[str]:
    """
    Parse an esorex log file and extract lines mentioning missing files or inputs.

    Returns a deduplicated list of relevant excerpts for user reporting.
    """
    missing: list[str] = []
    seen: set[str] = set()
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if "missing" in line.lower():
                    stripped = line.strip()
                    if stripped and stripped not in seen:
                        missing.append(stripped)
                        seen.add(stripped)
    except (FileNotFoundError, OSError):
        pass
    return missing


def compact_missing_summary(lines: list[str]) -> str:
    """
    Build a compact 'Missing: dark file, badpix map' string from esorex log lines.

    Extracts the phrase immediately following the word "missing" (case-insensitive)
    from each line, e.g. "missing dark file" → "dark file".
    Falls back to a truncated first line if no such pattern is found.
    """
    import re

    types: list[str] = []
    seen: set[str] = set()
    for line in lines:
        # Capture 1-3 words after "missing", ignoring leading punctuation/spaces
        m = re.search(r"\bmissing\s+([^\n:,;]{1,40})", line, re.IGNORECASE)
        if m:
            phrase = m.group(1).strip().rstrip(".")
            if phrase and phrase.lower() not in seen:
                types.append(phrase)
                seen.add(phrase.lower())
    if types:
        summary = ", ".join(types[:6])
        return f"Missing: {summary}" + (" …" if len(types) > 6 else "")
    if lines:
        first = lines[0]
        return first[:70] + ("…" if len(first) > 70 else "")
    return "Esorex execution failed"
