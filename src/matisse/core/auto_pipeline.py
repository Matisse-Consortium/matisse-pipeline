"""
Core helpers for the MATISSE pipeline GUI series.

Created in 2016
Authors: pbe, fmillour, ame

Revised in 2025
Contributor: aso

This module exposes the core function of the automatic
MATISSE data reduction pipeline interface.
"""

# import argparse
from __future__ import annotations

import glob
import os
import shlex
import shutil
import subprocess
from multiprocessing import Pool
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
from astropy.io.fits import getheader
from astroquery.vizier import Vizier

# --- MATISSE logging and console (imported from core) ---
from rich.progress import Progress, SpinnerColumn

from matisse.core.lib_auto_pipeline import (
    add_mdfc_fluxes,
    matisse_action,
    matisse_calib,
    matisse_recipes,
    matisse_type,
)
from matisse.core.utils.common import remove_double_parameter
from matisse.core.utils.io_utils import resolve_raw_input
from matisse.core.utils.log_utils import (
    console,
    iteration_banner,
    log,
    save_report,
    section,
    show_blocs_status,
    show_calibration_status,
)


class RedBlock(TypedDict):
    action: str
    recipes: str
    param: str
    input: list[tuple[Path, str, dict[str, Any]]]
    calib: list[Any]
    status: int
    tplstart: str
    iter: int


def run_esorex(args):
    """
    Execute a single EsoRex command in a separate process.

    This function is designed to be called in parallel by a multiprocessing pool.
    It runs one EsoRex command, redirecting its standard output and error streams
    to dedicated `.log` and `.err` files located in the current working directory.

    Parameters
    ----------
    args : tuple
        A tuple containing:
        - cmd (str): The full EsoRex command to execute.
        - block_index (int): The index of the current processing block.
        - recipe (str): Short recipe name for display purposes.

    Returns
    -------
    tuple
        A tuple `(block_index, success, recipe, err_file)` where:
        - block_index (int): The index of the processed block.
        - success (bool): True if the command completed successfully, False otherwise.
        - recipe (str): Recipe name (for reporting).
        - err_file (str): Path to the .err log file (for failure diagnosis).

    Notes
    -----
    This function is typically executed inside a multiprocessing pool, for example:
    ```
    with Pool(processes=nb_core) as pool:
        for result in pool.imap_unordered(run_esorex, tasks):
            ...
    ```
    """
    cmd, block_index, recipe = args

    spl = cmd.split("%")
    cmd = spl[0].strip()

    # Extract output file names
    parts = cmd.split()
    out = parts[-1] + ".log"
    err = parts[-1] + ".err"

    # Extract working directory (after "=" in 2nd arg)
    if len(parts) > 1 and "=" in parts[1]:
        val = parts[1].split("=")
        workdir = val[1]
    else:
        workdir = "."

    # Run the command via subprocess (no shell needed)
    cmd_args = shlex.split(cmd)
    with (
        open(Path(workdir) / out, "w", encoding="utf-8") as fout,
        open(Path(workdir) / err, "w", encoding="utf-8") as ferr,
    ):
        result = subprocess.run(
            cmd_args,
            cwd=workdir,
            stdout=fout,
            stderr=ferr,
            check=False,
        )
    ret = result.returncode

    err_path = str(Path(workdir) / err)
    return block_index, ret == 0, recipe, err_path


def run_pipeline(
    dirRaw: str,
    dirCalib: str = "",
    dirResult: str = "",
    nbCore: int = 1,
    resol: str = "",
    paramL: str = "",
    paramN: str = "",
    overwrite: int = 0,
    skipL: bool = False,
    skipN: bool = False,
    tplstartsel: str = "",
    tplidsel: str = "",
    spectralBinning: str = "",
    check_blocks: bool = False,
    check_calib: bool = False,
    detailed_block: int | None = None,
    custom_recipes_dir: Path | None = None,
    save_report_svg: bool = False,
):
    """Main function to run MATISSE automatic pipeline."""

    # --- Section header ---
    section("Starting raw data reduction")

    # --- Setup Vizier to collect MATISSE magnitudes ---
    vizier_cat_mdfc = Vizier(
        columns=["med-Lflux", "med-Mflux", "med-Nflux"], catalog="II/361"
    )

    # 1) Resolve raw input into a normalized list of FITS files

    try:
        list_raw, input_origin = resolve_raw_input(dirRaw)
    except FileNotFoundError as err:
        # console.print("[red]No raw data found in the specified directory.[/]")
        log.error(str(err))
        raise SystemExit(1) from err

    log.info(f"Discovered {len(list_raw)} FITS files from {input_origin}.")

    # 2) Resolve calibration directory
    if dirCalib:
        p = Path(dirCalib)
        if p.is_dir():
            listArchive = [str(f) for f in p.rglob("*.fits")] + [
                str(f) for f in p.rglob("*.fits.gz")
            ]
            log.info(f"Calibration directory explicitly provided: {p}")
        else:
            listArchive = []

    # --- Start file processing (progress shown below) ---
    # -> Sort listRaw using template ID and template start
    section("Processing FITS Headers")

    allhdr = []
    failed_files = []
    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("[cyan]Reading FITS headers...", total=len(list_raw))
        for filename in list_raw:
            try:
                hdr = getheader(filename, 0)
                allhdr.append(hdr)
            except Exception as err:
                failed_files.append(filename)
                log.warning(f"Failed to read header: {filename} ({err})")
            progress.advance(task)

    if failed_files:
        console.print(
            f"[yellow]⚠ Warning:[/] {len(failed_files)} files failed to read headers."
        )
        log.warning(f"Header read failures: {[p.name for p in failed_files[:5]]}...")
    log.info(f"Successfully read {len(allhdr)} FITS headers.")

    # Keep a reference to all headers before filtering for skip reporting
    allhdr_all = list(allhdr)

    listRawSorted = []
    allhdrSorted = []
    listRes = []
    listGRA4MAT = []
    listhdrGRA4MAT = []
    for hdr, filename in zip(allhdr, list_raw, strict=False):
        chip = ""
        if "RMNREC" in hdr["HIERARCH ESO DPR TYPE"]:
            if "GRAVITY" in hdr["HIERARCH ESO DEL FT SENSOR"]:
                listGRA4MAT.append(filename)
                listhdrGRA4MAT.append(hdr)

        if "HIERARCH ESO TPL START" in hdr and "HIERARCH ESO DET CHIP NAME" in hdr:
            tplid = hdr["HIERARCH ESO TPL ID"]
            tplstart = hdr["HIERARCH ESO TPL START"]
            chip = hdr["HIERARCH ESO DET CHIP NAME"]

        # Warning : to be checked with the if or not
        if not skipL and chip == "HAWAII-2RG":
            # Append low resolution stuff in the front of the list
            disperser = hdr["HIERARCH ESO INS DIL NAME"]

            if resol != "":
                if disperser != resol:
                    continue

            # Go through all 4 cases. First case: tplid and tplstart given by user
            if tplidsel != "" and tplstartsel != "":
                if tplid == tplidsel and tplstart == tplstartsel:
                    listRawSorted.append(filename)
                    allhdrSorted.append(hdr)
                    listRes.append(disperser)
            # Second case: tpl ID given but not tpl start
            if tplidsel != "" and tplstartsel == "":
                if tplid == tplidsel:
                    listRawSorted.append(filename)
                    allhdrSorted.append(hdr)
                    listRes.append(disperser)
            # Third case: tpl start given but not tpl ID
            if tplidsel == "" and tplstartsel != "":
                if tplstart == tplstartsel:
                    listRawSorted.append(filename)
                    allhdrSorted.append(hdr)
                    listRes.append(disperser)
            # Fourth case: nothing given by user
            if tplidsel == "" and tplstartsel == "":
                listRawSorted.append(filename)
                allhdrSorted.append(hdr)
                listRes.append(disperser)

        if not skipN and chip == "AQUARIUS":
            # Append low resolution stuff in the front of the list
            disperser = hdr["HIERARCH ESO INS DIN NAME"]

            if resol != "":
                if disperser != resol:
                    continue

            # Go through all 4 cases. First case: tplid and tplstart given by user
            if tplidsel != "" and tplstartsel != "":
                if tplid == tplidsel and tplstart == tplstartsel:
                    listRawSorted.append(filename)
                    allhdrSorted.append(hdr)
                    listRes.append(disperser)
            # Second case: tpl ID given but not tpl start
            if tplidsel != "" and tplstartsel == "":
                if tplid == tplidsel:
                    listRawSorted.append(filename)
                    allhdrSorted.append(hdr)
                    listRes.append(disperser)
            # Third case: tpl start given but not tpl ID
            if tplidsel == "" and tplstartsel != "":
                if tplstart == tplstartsel:
                    listRawSorted.append(filename)
                    allhdrSorted.append(hdr)
                    listRes.append(disperser)
            # Fourth case: nothing given by user
            if tplidsel == "" and tplstartsel == "":
                listRawSorted.append(filename)
                allhdrSorted.append(hdr)
                listRes.append(disperser)

    # Replace original list with the sorted one
    listRaw = listRawSorted
    allhdr = allhdrSorted

    # Report skipped bands
    if skipL or skipN:
        n_lm = sum(
            1 for h in allhdr_all if h.get("HIERARCH ESO DET CHIP NAME") == "HAWAII-2RG"
        )
        n_n = sum(
            1 for h in allhdr_all if h.get("HIERARCH ESO DET CHIP NAME") == "AQUARIUS"
        )
        if skipL and n_lm > 0:
            log.warning(f"Skipped {n_lm} LM-band files (--skipL).")
        if skipN and n_n > 0:
            log.warning(f"Skipped {n_n} N-band files (--skipN).")

    # =====================================================
    # === STAGE 3: REDUCTION BLOCK CREATION ===
    # =====================================================
    section("Reduction Block Discovery")

    # Determination of the number of Reduction Blocks
    keyTplStart: list[str] = []
    listIterNumber: list[int] = []
    list_target: list[str] = []  # NEW: add target lists
    log.info("Determining the number of reduction blocks...")

    with Progress(console=console, transient=True) as progress:
        task = progress.add_task(
            "[cyan]Creating reduction blocks...", total=len(allhdr)
        )
        for hdr, filename, _res in zip(allhdr, listRaw, listRes, strict=False):
            try:
                tplstart = hdr["HIERARCH ESO TPL START"]
                chipname = hdr["HIERARCH ESO DET CHIP NAME"]
                targetname = hdr.get("ESO OBS TARG NAME", "CAL FILE")
            except KeyError:
                log.warning(f"{filename} is not a valid MATISSE FITS file.")
                progress.advance(task)
                continue
            # Reduction blocks are defined by template start and detector name
            keyTplStart.append(f"{tplstart}.{chipname}")
            list_target.append(targetname)
            progress.advance(task)

    # Log total number of unique templates before sorting
    unique_keys = sorted(set(keyTplStart))
    log.info(
        f"Discovered {len(unique_keys)} unique (TPL_START, DETECTOR) combinations."
    )

    # Put LOW first, then MED then HIGH (preserve instrument resolution order)
    keyTplStart2: list[str] = []
    for _, ikey2 in enumerate(unique_keys):
        idx = np.where([ikey == ikey2 for ikey in keyTplStart])[0][0]
        if listRes[idx] == "HIGH":
            listRes[idx] = "zHIGH"
        keyTplStart2.append(f"{listRes[idx]}%{ikey2}")

    # Final sorted block list (LOW → MED → HIGH)
    keyTplStart_sorted = sorted(set(keyTplStart2))
    keyTplStart = [it.split("%")[1] for it in keyTplStart_sorted]

    # Initialize iteration counters
    listIterNumber = [0 for _ in keyTplStart]
    # =====================================================
    # === STAGE 4: MAIN DATA REDUCTION ===
    # =====================================================
    section("Data reduction execution")

    MAX_SAFETY_ITER = 10  # Safety cap to prevent infinite loops

    iterNumber = 0
    while True:
        iterNumber += 1
        iteration_banner(iterNumber)
        # For iter > 1, demote repetitive messages to debug
        _iter_log = log.debug if iterNumber > 1 else log.info
        if iterNumber > 1:
            listIter: list[str] = []
            _iter_log("Listing files from previous iteration...")
            for iter in range(iterNumber - 1):
                _iter_log(f"Listing files from iteration {iter + 1}...")
                repIterPrev = os.path.join(dirResult, "reduced")
                listRepIter = [
                    os.path.join(repIterPrev, f)
                    for f in os.listdir(repIterPrev)
                    if os.path.isdir(os.path.join(repIterPrev, f))
                ]
                for elt in listRepIter:
                    listIter = listIter + [
                        os.path.join(elt, f)
                        for f in os.listdir(elt)
                        if os.path.isfile(os.path.join(elt, f))
                        and (f.endswith(".fits") or f.endswith(".fits.gz"))
                    ]

        _iter_log("Listing reduction blocks...")
        # Reduction Blocks List Initialization
        list_red_blocks: list[RedBlock] = [
            {
                "action": " ",
                "recipes": " ",
                "param": " ",
                "input": [],
                "calib": [],
                "status": 0,
                "tplstart": " ",
                "iter": iter_num + 1,
            }
            for _, iter_num in zip(keyTplStart, listIterNumber, strict=True)
        ]

        # Fill the list of raw data in the Reduction Blocks List
        log.debug("Listing files in the reduction blocks...")
        for hdr, filename in zip(allhdr, listRaw, strict=False):
            if "RMNREC" in hdr["HIERARCH ESO DPR TYPE"]:
                log.warning(f"{filename} is a RMNREC file!")
                continue
            else:
                try:
                    chipname = hdr["HIERARCH ESO DET CHIP NAME"]
                    stri = hdr["HIERARCH ESO TPL START"] + "." + chipname
                except Exception:
                    log.warning("{filename} is not a valid MATISSE fits file!")
                    continue
            tag = matisse_type(hdr)
            list_red_blocks[keyTplStart.index(stri)]["input"].append(
                (filename, tag, hdr)
            )

        # Fill the list of actions,recipes,params in the Reduction Blocks List
        log.debug("Listing actions in the reduction blocks...")
        for red_block in list_red_blocks:
            # hdr = red_block["input"][0][2]
            hdr = cast(dict[str, str], red_block["input"][0][2])
            chip = hdr["HIERARCH ESO DET CHIP NAME"]
            keyTplStartCurrent = hdr["HIERARCH ESO TPL START"] + "." + chip
            if chip == "AQUARIUS":
                resolution = hdr["HIERARCH ESO INS DIN NAME"]
            if chip == "HAWAII-2RG":
                resolution = hdr["HIERARCH ESO INS DIL NAME"]

            action = matisse_action(hdr, red_block["input"][0][1])

            tel = ""
            if "TELESCOP" in hdr:
                tel = hdr["TELESCOP"]

            recipes, param = matisse_recipes(
                action, hdr["HIERARCH ESO DET CHIP NAME"], tel, resolution
            )
            red_block["action"] = action
            red_block["recipes"] = recipes

            if action == "ACTION_MAT_RAW_ESTIMATES":
                if hdr["HIERARCH ESO DET CHIP NAME"] == "AQUARIUS":
                    if spectralBinning != "":
                        paramN += " --spectralBinning=" + spectralBinning
                    else:
                        paramN += " --spectralBinning=7"

                    if paramN == "":
                        red_block["param"] = param
                    else:
                        red_block["param"] = paramN + " " + param
                else:
                    if spectralBinning != "":
                        paramL += " --spectralBinning=" + spectralBinning
                    else:
                        paramL += " --spectralBinning=5"

                    if paramL == "":
                        red_block["param"] = param
                    else:
                        red_block["param"] = paramL + " " + param
            else:
                red_block["param"] = param
            red_block["tplstart"] = keyTplStartCurrent

        log.debug("Searching for GRA4MAT data...")
        count_grav = 0
        for red_block in list_red_blocks:
            hdr = red_block["input"][0][2]
            keyTplStartCurrent = hdr["HIERARCH ESO TPL START"]
            for fileGV, hdrGV in zip(listGRA4MAT, listhdrGRA4MAT, strict=True):
                if hdrGV["HIERARCH ESO TPL START"] == keyTplStartCurrent:
                    red_block["input"].append((fileGV, "RMNREC", hdrGV))
                    count_grav += 1

        _iter_log(f"{count_grav} files identified as GRA4MAT.")

        # Fill the list of calib in the Reduction Blocks List from dirCalib
        log.debug("Listing calibrations in the reduction blocks...")

        with Progress(console=console, transient=True) as progress:
            task = progress.add_task(
                "[cyan]Listing calibrations...",
                total=len(list_red_blocks),
            )
            for red_block in list_red_blocks:
                hdr = red_block["input"][0][2]
                calib, status = matisse_calib(
                    hdr,
                    red_block["action"],
                    listArchive,
                    red_block["calib"],
                    red_block["tplstart"].split(".")[0],
                )
                red_block["calib"] = calib
                red_block["status"] = status
                progress.advance(task)

        log.debug(
            "Listing calibrations from previous iteration in the reduction blocks..."
        )
        if iterNumber > 1:
            for red_block in list_red_blocks:
                hdr = red_block["input"][0][2]
                calib, status = matisse_calib(
                    hdr,
                    red_block["action"],
                    listIter,
                    red_block["calib"],
                    red_block["tplstart"].split(".")[0],
                )
                red_block["calib"] = calib
                red_block["status"] = status

        # Sort blocks: non-RAW_ESTIMATES first, then RAW_ESTIMATES (LM before N)
        def _block_sort_key(idx_block):
            idx, block = idx_block
            is_raw_est = 1 if block["action"] == "ACTION_MAT_RAW_ESTIMATES" else 0
            # AQUARIUS (N) after HAWAII-2RG (LM)
            detector = block["tplstart"].split(".")[-1]
            is_n_band = 1 if detector == "AQUARIUS" else 0
            return (is_raw_est, is_n_band, block["tplstart"])

        sorted_indices = sorted(
            range(len(list_red_blocks)),
            key=lambda i: _block_sort_key((i, list_red_blocks[i])),
        )
        list_red_blocks = [list_red_blocks[i] for i in sorted_indices]
        keyTplStart = [keyTplStart[i] for i in sorted_indices]
        listIterNumber = [listIterNumber[i] for i in sorted_indices]

        if detailed_block is not None:
            check_calib = True

        if check_calib:
            show_calibration_status(
                list_red_blocks, console, detailed_block=detailed_block
            )
            break

        # Create the SOF files
        _iter_log("Creating the sof files and directories...")

        repIter = os.path.join(dirResult, "reduced")
        if not check_blocks:
            if os.path.isdir(repIter):
                if overwrite == 1:
                    shutil.rmtree(repIter)
                    os.mkdir(repIter)
            else:
                os.mkdir(repIter)

        listCmdEsorex = []
        cptStatusOne = 0
        cptStatusZero = 0
        cptToProcess = 0
        cpt = 0

        skip_calib_iter = False
        for i_block, red_block in enumerate(list_red_blocks):
            rbname = red_block["recipes"] + "." + red_block["tplstart"]
            rbname_safe = rbname.replace(":", "_")
            # For iter > 1, only show blocks that will actually be processed
            _blog = _iter_log  # default: info for iter 1, debug for iter > 1
            if iterNumber == 1:
                log.info(f"[cyan]🧱 Block #{i_block + 1} : {rbname.split('.')[0]}")
            overwritei = overwrite
            if red_block["status"] == 1:
                cptStatusOne += 1

                # In check mode, mark block as ready but don't create
                # any files or directories (purely informational)
                if check_blocks:
                    _blog("Block ready to be processed (check mode).")
                    red_block["status"] = -2
                    cpt += 1
                    continue

                sofname = os.path.join(repIter, rbname_safe + ".sof")
                outputDir = os.path.join(repIter, rbname_safe + ".rb")
                print_sof_status = True
                if overwritei == 0:
                    files_found = (
                        glob.glob(os.path.join(outputDir, "*_RAW_INT_*.fits"))
                        or glob.glob(os.path.join(outputDir, "*_RAW_INT_*.fits.gz"))
                        or glob.glob(os.path.join(outputDir, "IM_BASIC.fits"))
                        or glob.glob(os.path.join(outputDir, "IM_BASIC.fits.gz"))
                        or glob.glob(os.path.join(outputDir, "OBS_FLATFIELD.fits"))
                        or glob.glob(os.path.join(outputDir, "OBS_FLATFIELD.fits.gz"))
                        or glob.glob(os.path.join(outputDir, "KAPPA_MATRIX.fits"))
                        or glob.glob(os.path.join(outputDir, "KAPPA_MATRIX.fits.gz"))
                        or glob.glob(os.path.join(outputDir, "SHIFT_MAP.fits"))
                        or glob.glob(os.path.join(outputDir, "SHIFT_MAP.fits.gz"))
                    )
                    if files_found:
                        existing_filename = Path(files_found[0]).name
                        _blog(
                            f"Block already processed, file {existing_filename} exists."
                        )
                        print_sof_status = False
                    else:
                        overwritei = 1
                        # Block will be processed — upgrade log level
                        if iterNumber > 1:
                            _blog = log.info
                            log.info(
                                f"[cyan]🧱 Block #{i_block + 1} : {rbname.split('.')[0]}"
                            )
                        _blog("🚀 Block will be processed.")

                resol = "no res"
                # Build SOF content in memory first
                sof_lines = []
                for frame, tag, hdr in red_block["input"]:
                    sof_lines.append(f"{str(frame)} {tag}\n")
                    resol = hdr["HIERARCH ESO INS DIL NAME"]
                for frame, tag in red_block["calib"]:
                    sof_lines.append(f"{str(frame)} {tag}\n")
                sof_content = "".join(sof_lines)

                # For iter > 1: compare new SOF with existing to detect
                # calibration changes from previous iteration outputs
                if iterNumber > 1 and os.path.exists(sofname):
                    with open(sofname, encoding="utf-8") as fp:
                        existing_content = fp.read()
                    if sof_content == existing_content:
                        _blog(
                            "Block already processed with identical calibrations, skipping."
                        )
                        cpt += 1
                        continue
                    else:
                        # Block will be re-processed — upgrade log level
                        _blog = log.info
                        log.info(f"[cyan]Re-processing Block #{i_block + 1} : {rbname}")
                        log.info(
                            f"Block #{i_block + 1} ({red_block['recipes']}): "
                            "calibrations updated."
                        )
                        overwritei = 1

                # Write SOF file
                if os.path.exists(sofname):
                    if overwritei:
                        log.debug("Overwriting existing .sof files")
                        with open(sofname, "w", encoding="utf-8") as fp:
                            fp.write(sof_content)
                    else:
                        if print_sof_status:
                            _blog(
                                "✅ sof file already exists (consider using --overwrite if needed)."
                            )
                else:
                    log.debug(
                        "sof file "
                        + Path(sofname).name
                        + " does not exist. Creating it..."
                    )
                    with open(sofname, "w", encoding="utf-8") as fp:
                        fp.write(sof_content)

                band_to_print = "LM" if "HAWAII" in red_block["tplstart"] else "N"
                if os.path.exists(outputDir):
                    if print_sof_status:
                        _blog(
                            "outputDir " + Path(outputDir).name + " already exists..."
                        )
                        _blog("Remove any previous logfile...")
                    try:
                        os.remove(os.path.join(outputDir, ".logfile"))
                    except Exception:
                        log.debug(".logfile does not exist or are already deleted.")

                    if os.listdir(outputDir) == []:
                        _blog("outputDir is empty, 🚀 block will be processed.")
                    else:
                        if print_sof_status:
                            _blog("outputDir already exists and is not empty...")

                        if overwritei:
                            log.warning("Overwriting existing files.")
                        else:
                            if print_sof_status:
                                log.warning(
                                    "outputDir exists (consider using --overwrite)."
                                )
                            continue
                else:
                    p = Path(outputDir)
                    path_to_print = p.relative_to(p.parents[1])
                    _blog(
                        f"Creates: [magenta]{path_to_print}[/magenta] ({band_to_print} band)."
                    )
                    os.mkdir(outputDir)

                listNewParams = remove_double_parameter(
                    red_block["param"].replace("/", " --")
                )

                if custom_recipes_dir is None:
                    cmd = (
                        "esorex --output-dir="
                        + outputDir
                        + " "
                        + red_block["recipes"]
                        + " "
                        + listNewParams
                        + " "
                        + sofname
                        + "%"
                        + resol
                    )
                else:
                    cmd = (
                        "esorex --output-dir="
                        + outputDir
                        + " "
                        + " --recipe-dir="
                        + str(custom_recipes_dir)
                        + " "
                        + red_block["recipes"]
                        + " "
                        + listNewParams
                        + " "
                        + sofname
                        + "%"
                        + resol
                    )
                listIterNumber[cpt] = iterNumber
                red_block["iter"] = iterNumber
                cptToProcess += 1
                listCmdEsorex.append((cmd, i_block + 1, red_block["recipes"]))
            else:
                cptStatusZero += 1
                _blog("Block not ready to be processed (missing calibrations).")
            cpt += 1

        # Compact summary for iter > 1
        if iterNumber > 1:
            n_reprocess = len(listCmdEsorex)
            if n_reprocess > 0:
                log.info(f"Iteration {iterNumber}: {n_reprocess} block(s) to process.")
            else:
                log.info(
                    f"Iteration {iterNumber}: all {cptStatusOne} block(s) unchanged."
                )
        else:
            log.info(
                f"Iteration {iterNumber}: {cptStatusOne} block(s) ready, "
                f"{cptStatusZero} block(s) not ready."
            )

        if not check_blocks:
            if listCmdEsorex != [] and not skip_calib_iter:
                n_tasks = len(listCmdEsorex)
                tasks = [
                    (cmd, block_num, recipe) for cmd, block_num, recipe in listCmdEsorex
                ]

                console.print(
                    f"\n[bold cyan]Launching {n_tasks} esorex task(s) "
                    f"on {nbCore} core(s)...[/]"
                )
                results = []
                failed = []
                with Progress(
                    SpinnerColumn(), *Progress.get_default_columns(), console=console
                ) as progress:
                    ptask = progress.add_task("[cyan]Running esorex...", total=n_tasks)
                    with Pool(processes=nbCore) as pool:
                        for (
                            block_index,
                            success,
                            recipe,
                            err_path,
                        ) in pool.imap_unordered(run_esorex, tasks):
                            results.append((block_index, success))
                            if success:
                                progress.console.print(
                                    f"  [green]✅ Block #{block_index}[/] {recipe}"
                                )
                            else:
                                progress.console.print(
                                    f"  [red]❌ Block #{block_index}[/] {recipe} "
                                    f"— see {Path(err_path).name}"
                                )
                                failed.append((block_index, recipe, err_path))
                            progress.advance(ptask)

                if failed:
                    console.print(f"\n[bold red]{len(failed)} block(s) failed:[/]")
                    for block_index, recipe, err_path in failed:
                        console.print(f"  Block #{block_index} ({recipe}): {err_path}")

        # Add MDFC Fluxes to CALIB_RAW_INT and TARGET_RAW_INT
        base_path = Path(repIter)
        list_oifits_files = [
            p
            for p in base_path.glob("*.rb/*_RAW_INT*")
            if p.name.endswith((".fits", ".fits.gz"))
        ]
        add_mdfc_fluxes(list_oifits_files, vizier_cat_mdfc)

        if show_blocs_status(listCmdEsorex, iterNumber, list_red_blocks, check_blocks):
            break

        if iterNumber >= MAX_SAFETY_ITER:
            log.warning(
                f"Reached safety limit of {MAX_SAFETY_ITER} iterations. Stopping."
            )
            break

    # --- Save pipeline report as SVG ---
    if save_report_svg:
        save_report(dirResult)
