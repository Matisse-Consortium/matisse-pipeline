#!/usr/bin/env python3
"""
This file is part of the Matisse pipeline GUI series
Copyright (C) 2017- Observatoire de la Côte d'Azur

Created in 2016
@author: pbe, fmi

Edited in 2025
@contributor: aso

Automatic MATISSE pipeline !

Please contact florentin.millour@oca.eu for any question

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.

You can use, modify and/ or redistribute the software under the terms
of the CeCILL license as circulated by CEA, CNRS and INRIA at the
following URL "http://www.cecill.info". You have a copy of the licence
in the LICENCE.md file.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.
"""

# import argparse
from __future__ import annotations

import filecmp
import glob
import os
import shutil
import sys
from multiprocessing import Manager, Pool
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
from astropy.io import fits
from astropy.io.fits import getheader
from astroquery.vizier import Vizier
from rich.panel import Panel

# --- MATISSE logging and console (imported from core) ---
from rich.progress import Progress

from matisse_pipeline.core.lib_auto_pipeline import (
    matisseAction,
    matisseCalib,
    matisseRecipes,
    matisseType,
)
from matisse_pipeline.core.utils.common import remove_double_parameter
from matisse_pipeline.core.utils.io_utils import resolve_raw_input
from matisse_pipeline.core.utils.log_utils import (
    console,
    iteration_banner,
    log,
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
        - cmd (list[str]): The full EsoRex command to execute, as a list of arguments.
        - block_index (int): The index of the current processing block.
        - lock (multiprocessing.Lock): A shared lock used to safely write or
          print messages from multiple parallel processes.

    Returns
    -------
    tuple
        A tuple `(block_index, success)` where:
        - block_index (int): The index of the processed block.
        - success (bool): True if the command completed successfully, False otherwise.

    Notes
    -----
    This function is typically executed inside a multiprocessing pool, for example:
    ```
    with Pool(processes=nb_core) as pool:
        results = pool.map(run_esorex, tasks)
    ```
    Each task must contain the `(cmd, block_index, lock)` tuple.

    """
    cmd, block_index, lock = args

    spl = cmd.split("%")
    cmd = spl[0].strip()

    with lock:
        console.print(f"[cyan] 🚀 Start esorex on block #{block_index}[/]")
        console.print(cmd)
        sys.stdout.flush()

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

    # Run the command in the proper directory and redirect output
    full_cmd = f"cd {workdir}; {cmd} > {out} 2> {err}"
    ret = os.system(full_cmd)

    status = "✅" if ret == 0 else "❌"
    with lock:
        console.print(
            f"[green]Block {block_index} completed {status}[/]"
            if ret == 0
            else f"[red]Block {block_index} failed {status}[/]"
        )

    return block_index, ret == 0


def run_pipeline(
    dirRaw: str,
    dirCalib: str = "",
    dirResult: str = "",
    nbCore: int = 1,
    resol: str = "",
    paramL: str = "",
    paramN: str = "",
    overwrite: int = 0,
    maxIter: int = 1,
    skipL: bool = False,
    skipN: bool = False,
    tplstartsel: str = "",
    tplidsel: str = "",
    spectralBinning: str = "",
    check_blocks: bool = False,
    check_calib: bool = False,
):
    """Main function to run MATISSE automatic pipeline."""

    # --- Section header ---
    section("Starting raw data reduction")

    # --- Setup Vizier to collect MATISSE magnitudes ---
    v = Vizier(columns=["med-Lflux", "med-Mflux", "med-Nflux"], catalog="II/361")

    # 1) Resolve raw input into a normalized list of FITS files

    try:
        list_raw, input_origin = resolve_raw_input(dirRaw)
    except FileNotFoundError as err:
        # console.print("[red]No raw data found in the specified directory.[/]")
        log.error(str(err))
        raise SystemExit(1) from err

    log.info(f"Discovered {len(list_raw)} FITS files from {input_origin}.")
    log.info(f"Example file: {list_raw[0]}")

    # 2) Resolve calibration directory
    if dirCalib:
        p = Path(dirCalib)
        if p.is_dir():
            listArchive = glob.glob(dirCalib + "/*.fits")
            log.info(f"Calibration directory explicitly provided: {p}")
        else:
            listArchive = []

    # --- Start file processing (progress shown below) ---
    # -> Sort listRaw using template ID and template start
    section("Processing FITS Headers")

    allhdr = []
    failed_files = []
    with Progress() as progress:
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

    # =====================================================
    # === STAGE 3: REDUCTION BLOCK CREATION ===
    # =====================================================
    section("Reduction Block Discovery")

    # Determination of the number of Reduction Blocks
    keyTplStart: list[str] = []
    listIterNumber: list[int] = []
    list_target: list[str] = []  # NEW: add target lists
    log.info("Determining the number of reduction blocks...")

    with Progress() as progress:
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

    iterNumber = 0
    while True:
        iterNumber += 1
        # log.info(f"Iteration number: {iterNumber}")
        console.print()
        # log.info(f":repeat: [bold cyan]Iteration {iterNumber}[/]")
        iteration_banner(iterNumber)
        if iterNumber > 1:
            listIter: list[str] = []
            log.info("Listing files from previous iteration...")
            for iter in range(iterNumber - 1):
                repIterPrev = dirResult + "/Iter" + str(iter + 1)
                listRepIter = [
                    os.path.join(repIterPrev, f)
                    for f in os.listdir(repIterPrev)
                    if os.path.isdir(os.path.join(repIterPrev, f))
                ]
                for elt in listRepIter:
                    listIter = listIter + [
                        os.path.join(elt, f)
                        for f in os.listdir(elt)
                        if os.path.isfile(os.path.join(elt, f)) and f[-5:] == ".fits"
                    ]

        log.info("Listing reduction blocks...")
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
                "iter": iter_num,
            }
            for _, iter_num in zip(keyTplStart, listIterNumber, strict=True)
        ]

        # Fill the list of raw data in the Reduction Blocks List
        log.info("Listing files in the reduction blocks...")
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
            tag = matisseType(hdr)
            list_red_blocks[keyTplStart.index(stri)]["input"].append(
                (filename, tag, hdr)
            )

        # Fill the list of actions,recipes,params in the Reduction Blocks List
        log.info("Listing actions in the reduction blocks...")
        for red_block in list_red_blocks:
            # hdr = red_block["input"][0][2]
            hdr = cast(dict[str, str], red_block["input"][0][2])
            chip = hdr["HIERARCH ESO DET CHIP NAME"]
            keyTplStartCurrent = hdr["HIERARCH ESO TPL START"] + "." + chip
            if chip == "AQUARIUS":
                resolution = hdr["HIERARCH ESO INS DIN NAME"]
            if chip == "HAWAII-2RG":
                resolution = hdr["HIERARCH ESO INS DIL NAME"]

            action = matisseAction(hdr, red_block["input"][0][1])

            tel = ""
            if "TELESCOP" in hdr:
                tel = hdr["TELESCOP"]

            recipes, param = matisseRecipes(
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

        log.info("Searching for GRA4MAT data...")
        count_grav = 0
        for red_block in list_red_blocks:
            hdr = red_block["input"][0][2]
            keyTplStartCurrent = hdr["HIERARCH ESO TPL START"]
            for fileGV, hdrGV in zip(listGRA4MAT, listhdrGRA4MAT, strict=True):
                if hdrGV["HIERARCH ESO TPL START"] == keyTplStartCurrent:
                    red_block["input"].append((fileGV, "RMNREC", hdrGV))
                    count_grav += 1

        log.info(f"{count_grav} files identified as GRA4MAT.")

        # Fill the list of calib in the Reduction Blocks List from dirCalib
        log.info("Listing calibrations in the reduction blocks...")

        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Listing calibrations...",
                total=len(list_red_blocks),
            )
            for red_block in list_red_blocks:
                hdr = red_block["input"][0][2]
                calib, status = matisseCalib(
                    hdr,
                    red_block["action"],
                    listArchive,
                    red_block["calib"],
                    red_block["tplstart"].split(".")[0],
                )
                red_block["calib"] = calib
                red_block["status"] = status
                progress.advance(task)

        log.info(
            "Listing calibrations from previous iteration in the reduction blocks..."
        )
        if iterNumber > 1:
            for red_block in list_red_blocks:
                hdr = red_block["input"][0][2]
                calib, status = matisseCalib(
                    hdr,
                    red_block["action"],
                    listIter,
                    red_block["calib"],
                    red_block["tplstart"].split(".")[0],
                )
                red_block["calib"] = calib
                red_block["status"] = status

        if check_calib and iterNumber == maxIter:
            show_calibration_status(list_red_blocks, console)
            break

        # Create the SOF files
        log.info("Creating the sof files and directories...")
        repIter = dirResult + "/Iter" + str(iterNumber)
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
            log.info(f"[cyan]Starting Block #{i_block + 1} : {rbname}")
            overwritei = overwrite
            if red_block["status"] == 1:
                cptStatusOne += 1
                sofname = os.path.join(repIter, rbname + ".sof").replace(":", ":")
                outputDir = os.path.join(repIter, rbname + ".rb").replace(":", "_")
                print_sof_status = True
                if overwritei == 0:
                    if glob.glob(
                        os.path.join(outputDir, "*_RAW_INT_*.fits")
                    ) or glob.glob(os.path.join(outputDir, "IM_BASIC.fits")):
                        log.info("Block already processed.")
                        print_sof_status = False
                    else:
                        overwritei = 1
                        # NOTES (aso) :
                        # overwritei = 1 (to be checked: calibration files are
                        # systematically reprocessed when overwritei == 1).
                        # Do we actually want this behavior?

                if glob.glob(os.path.join(outputDir, "*FIELD.fits")):
                    skip_calib_iter = True
                elif glob.glob(os.path.join(outputDir, "SHIFT_MAP.fits")):
                    skip_calib_iter = True

                resol = "no res"
                if os.path.exists(sofname):
                    if overwritei:
                        log.warning("Overwriting existing .sof files")
                        fp = open(sofname, "w")
                        for frame, tag, hdr in red_block["input"]:
                            fp.write(f"{str(frame)} {tag}\n")
                            resol = hdr["HIERARCH ESO INS DIL NAME"]
                        for frame, tag in red_block["calib"]:
                            fp.write(f"{str(frame)} {tag}\n")
                        fp.close()
                    else:
                        if print_sof_status:
                            log.info(
                                "✅ sof file already exists (consider using --overwrite if needed)."
                            )
                else:
                    log.info(
                        "sof file "
                        + Path(sofname).name
                        + " does not exist. Creating it..."
                    )
                    fp = open(sofname, "w")
                    for frame, tag, hdr in red_block["input"]:
                        fp.write(f"{str(frame)} {tag}\n")
                        resol = hdr["HIERARCH ESO INS DIL NAME"]
                    for frame, tag in red_block["calib"]:
                        fp.write(f"{str(frame)} {tag}\n")
                    fp.close()

                if os.path.exists(outputDir):
                    if print_sof_status:
                        log.info(
                            "outputDir " + Path(outputDir).name + " already exists..."
                        )
                        # Remove any previous logfile
                        log.info("Remove any previous logfile...")
                    try:
                        os.remove(os.path.join(outputDir, ".logfile"))
                    except Exception:
                        log.info(".logfile does not exist or are already deleted.")

                    if os.listdir(outputDir) == []:
                        if check_blocks:
                            log.info("Block ready to be proceeded.")
                            red_block["status"] = -2
                    else:
                        if print_sof_status:
                            log.info("outputDir already exists and is not empty...")

                        if overwritei:
                            log.warning("Overwriting existing files.")
                        else:
                            if print_sof_status:
                                log.warning(
                                    "outputDir exists (consider using --overwrite)."
                                )
                            continue
                else:
                    log.info(f"Create outputDir ({Path(outputDir)}).")
                    os.mkdir(outputDir)

                listNewParams = remove_double_parameter(
                    red_block["param"].replace("/", " --")
                )

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
                if iterNumber > 1:
                    sofnamePrev = (
                        repIterPrev
                        + "/"
                        + red_block["recipes"]
                        + "."
                        + red_block["tplstart"]
                        + ".sof"
                    )
                    if os.path.exists(sofnamePrev):
                        if filecmp.cmp(sofname, sofnamePrev):
                            print(
                                "Reduction Blocks already processed during previous iteration"
                            )
                            print("Remove directory : " + outputDir)
                            shutil.rmtree(outputDir)
                        else:
                            listIterNumber[cpt] = iterNumber
                            red_block["iter"] = iterNumber
                            cptToProcess += 1
                            listCmdEsorex.append(cmd)
                    else:
                        listIterNumber[cpt] = iterNumber
                        red_block["iter"] = iterNumber
                        cptToProcess += 1
                        listCmdEsorex.append(cmd)
                else:
                    listIterNumber[cpt] = iterNumber
                    red_block["iter"] = iterNumber
                    cptToProcess += 1
                    listCmdEsorex.append(cmd)
            else:
                cptStatusZero += 1
            cpt += 1

        if not check_blocks:
            if listCmdEsorex != [] and not skip_calib_iter and iterNumber <= maxIter:
                with Manager() as manager:
                    lock = manager.Lock()  # shared lock between processes
                    tasks = [(cmd, i + 1, lock) for i, cmd in enumerate(listCmdEsorex)]
                    with Pool(processes=nbCore) as pool:
                        results = pool.map(run_esorex, tasks)

                console.print("\n[bold cyan]=== ESOREX Parallel Summary ===[/]")
                for block_index, success in results:
                    console.print(
                        f"Block {block_index}: {'✅ OK' if success else '❌ ERROR'}"
                    )

        if check_blocks:
            console.print(
                Panel.fit(
                    f"[green]Done:[/] {cptStatusOne - cptToProcess}  |  [yellow]To process:[/] {cptToProcess}  |  [red]Failed:[/] {cptStatusZero}",
                    title="[bold]ESOREX[/]",
                    border_style="green",
                ),
                justify="center",
            )

        # Add MDFC Fluxes to CALIB_RAW_INT and TARGET_RAW_INT
        list_oifits_files = glob.glob(repIter + "/*.rb/*_RAW_INT*.fits")
        for oifits_filename in list_oifits_files:
            hdu = fits.open(oifits_filename, mode="update")
            targetname = hdu[0].header["ESO OBS TARG NAME"]
            try:
                result = v.query_region(targetname, radius="20s")
                fluxL = result[0][0][0]
                fluxM = result[0][0][1]
                fluxN = result[0][0][2]
                hdu[0].header["HIERARCH PRO MDFC FLUX L"] = (
                    fluxL,
                    "Flux (Jy) in L band from MDFC catalog",
                )
                hdu[0].header["HIERARCH PRO MDFC FLUX M"] = (
                    fluxM,
                    "Flux (Jy) in M band from MDFC catalog",
                )
                hdu[0].header["HIERARCH PRO MDFC FLUX N"] = (
                    fluxN,
                    "Flux (Jy) in N band from MDFC catalog",
                )
                hdu.flush()
            except Exception:
                try:
                    log.warning("Object " + targetname + " not found in MDFC catalog.")
                except NameError:
                    log.warning("There is no target name.")
            hdu.close()

        if show_blocs_status(
            listCmdEsorex, iterNumber, maxIter, list_red_blocks, check_blocks
        ):
            log.info(f"end {len(listCmdEsorex)} {len(list_raw)}")
            break
