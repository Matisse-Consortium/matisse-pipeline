#!/usr/bin/env python3
# LEGACY FILE — DO NOT MODIFY
# This module is kept for backward compatibility with old data reduction workflows.
# Excluded from linters and pre-commit hooks.
"""
This file is part of the Matisse pipeline GUI series
Copyright (C) 2017- Observatoire de la Côte d'Azur

Created in 2016
@author: pbe, fmi

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

import argparse
import filecmp
import glob
import os
import shutil
import sys
from multiprocessing.pool import Pool

import numpy as np
from astropy.io import fits
from astropy.io.fits import getheader
from astroquery.vizier import Vizier
from libAutoPipeline import *
from tqdm import tqdm


# Run esorex recipes
def runEsorex(cmd):
    spl = cmd.split("%")
    cmd = spl[0]
    print(cmd)
    #    resol = spl[1];
    #    print(resol)
    sys.stdout.flush()
    item = cmd.split()
    out = item[-1] + ".log"
    err = item[-1] + ".err"
    val = item[-1].split(".")
    # print("Running (Recipes : ",item[2],", TplStart : ",val[1],", Detector : ",val[2],")")
    val = item[1].split("=")
    os.system("cd " + val[1] + ";" + cmd + " > " + out + " 2> " + err)


# ------------------------------------------------------------------------------


def removeDoubleParameter(p):
    listP = p.split(" ")
    paramName = []
    paramsNew = ""
    for elt in listP:
        idx = elt.find("=")
        if elt[0:idx] not in paramName and elt != "":
            paramName.append(elt[0:idx])
            paramsNew = paramsNew + " " + elt
    return paramsNew


# ------------------------------------------------------------------------------


def mat_autoPipeline(
    dirRaw="",
    dirResult="",
    dirCalib="",
    nbCore=0,
    resol=0,
    paramL="",
    paramN="",
    overwrite=0,
    maxIter=0,
    skipL=0,
    skipN=0,
    tplstartsel="",
    tplidsel="",
    spectralBinning="",
):
    v = Vizier(columns=["med-Lflux", "med-Mflux", "med-Nflux"], catalog="II/361")
    # Print meaningful error messages if something is wrong in the command line
    print("------------------------------------------------------------------------")
    if dirRaw == "":
        print("ERROR: You have to specifiy a Raw Data Directory or a list of raw file")
        sys.exit(0)
    else:
        print("%-40s" % ("Raw Data Directory or file list:",), dirRaw)
    if dirCalib == "":
        dirCalib = dirRaw
        print(
            "Info: Calibration Directory not specified. We use the default directory "
            + dirCalib
        )
    print("%-40s" % ("Calibration Directory:",), dirCalib)
    if dirResult == "":
        dirResult = os.getcwd()
        print("Info : Results Directory not specified. We use the current directory")
    print("%-40s" % ("Results Directory:",), dirResult)
    if nbCore == 0:
        nbCore = 1
        print("Info : Number of Cores not specified. We use " + str(nbCore) + " core")
    print("%-40s" % ("Number of Cores:",), nbCore)
    if maxIter == 0:
        maxIter = 1
        print(
            "Info : Maximum Number of Iteration not specified. We fix it to "
            + str(maxIter)
        )
    print("%-40s" % ("Maximum Number of Iteration:",), maxIter)

    print("------------------------------------------------------------------------")
    if "[" not in dirRaw:
        listRaw = glob.glob(dirRaw + "/MATIS*.fits")
        print("Raw directory given")
    else:
        print("List of raw files given")
        listRaw = eval(dirRaw)
    if dirCalib != "":
        listArchive = glob.glob(dirCalib + "/*.fits")
    else:
        listArchive = []

    # Sort listRaw using template ID and template start
    print("Sorting files according to constraints...")
    allhdr = []
    for filename in tqdm(
        listRaw, unit=" files", unit_scale=False, desc="Working on files"
    ):
        try:
            allhdr.append(getheader(filename, 0))
        except:
            print("\nWARNING: corrupt file!")

    listRawSorted = []
    allhdrSorted = []
    listRes = []
    listGRA4MAT = []
    listhdrGRA4MAT = []
    for hdr, filename in zip(allhdr, listRaw, strict=False):
        chip = ""
        if "RMNREC" in hdr["HIERARCH ESO DPR TYPE"]:
            if "GRAVITY" in hdr["HIERARCH ESO DEL FT SENSOR"]:
                listGRA4MAT.append(filename)
                listhdrGRA4MAT.append(hdr)

        if "HIERARCH ESO TPL START" in hdr and "HIERARCH ESO DET CHIP NAME" in hdr:
            tplid = hdr["HIERARCH ESO TPL ID"]
            tplstart = hdr["HIERARCH ESO TPL START"]
            chip = hdr["HIERARCH ESO DET CHIP NAME"]

        if skipL == 0 and chip == "HAWAII-2RG":
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

        if skipN == 0 and chip == "AQUARIUS":
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

    # Determination of the number of Reduction Blocks
    keyTplStart = []
    listIterNumber = []
    print("Determining the number of reduction blocks...")

    for hdr, filename, _res in zip(allhdr, listRaw, listRes, strict=False):
        try:
            tplstart = hdr["HIERARCH ESO TPL START"]
            chipname = hdr["HIERARCH ESO DET CHIP NAME"]
        except:
            print("WARNING, " + filename + " is not a valid MATISSE fits file!")
            continue
        # Reduction blocks are defined by template start and detector name
        temp = tplstart + "." + chipname
        keyTplStart.append(temp)

    # Put LOW first, then MED then HIGH
    keyTplStart2 = sorted(set(keyTplStart))
    for idx2, ikey2 in enumerate(keyTplStart2):
        # print(idx2)
        # print(ikey2)
        idx = np.where([ikey == ikey2 for ikey in keyTplStart])
        idx = idx[0][0]
        # print(idx)
        # put high res data at the end
        if listRes[idx] == "HIGH":
            listRes[idx] = "zHIGH"
        keyTplStart2[idx2] = listRes[idx] + "%" + keyTplStart2[idx2]
    keyTplStart = sorted(set(keyTplStart2))
    keyTplStart = list([it.split("%")[1] for it in keyTplStart])

    for _ in keyTplStart:
        listIterNumber.append(0)
    print("Found " + str(len(keyTplStart)) + " reduction blocks.")

    iterNumber = 0
    while True:
        iterNumber += 1
        print("")
        print("Iteration ", iterNumber)
        print("-----------------------")
        if iterNumber > 1:
            listIter = []
            print("listing stuff...")
            for iter in range(iterNumber - 1):
                repIterPrev = dirResult + "/Iter" + str(iter + 1)
                listRepIter = [
                    os.path.join(repIterPrev, f)
                    for f in os.listdir(repIterPrev)
                    if os.path.isdir(os.path.join(repIterPrev, f))
                ]
                print("listing files from previous iteration...")
                for elt in listRepIter:
                    listIter = listIter + [
                        os.path.join(elt, f)
                        for f in os.listdir(elt)
                        if os.path.isfile(os.path.join(elt, f)) and f[-5:] == ".fits"
                    ]

        print("listing reduction blocks...")
        listRedBlocks = []
        # Reduction Blocks List Initialization
        cpt = 0
        for _elt in keyTplStart:
            listRedBlocks.append(
                {
                    "action": " ",
                    "recipes": " ",
                    "param": " ",
                    "input": [],
                    "calib": [],
                    "status": 0,
                    "tplstart": " ",
                    "iter": listIterNumber[cpt],
                }
            )
            cpt += 1
        # Fill the list of raw data in the Reduction Blocks List
        print("listing files in the reduction blocks...")
        for hdr, filename in zip(allhdr, listRaw, strict=False):
            if "RMNREC" in hdr["HIERARCH ESO DPR TYPE"]:
                print("WARNING, " + filename + " is a RMNREC file!")
                continue
            else:
                try:
                    chipname = hdr["HIERARCH ESO DET CHIP NAME"]
                    stri = hdr["HIERARCH ESO TPL START"] + "." + chipname
                except:
                    print("WARNING, " + filename + " is not a valid MATISSE fits file!")
                    continue
            tag = matisseType(hdr)
            listRedBlocks[keyTplStart.index(stri)]["input"].append([filename, tag, hdr])

        # Fill the list of actions,recipes,params in the Reduction Blocks List
        print("listing actions in the reduction blocks...")
        for elt in listRedBlocks:
            hdr = elt["input"][0][2]
            chip = hdr["HIERARCH ESO DET CHIP NAME"]
            keyTplStartCurrent = hdr["HIERARCH ESO TPL START"] + "." + chip
            if chip == "AQUARIUS":
                resolution = hdr["HIERARCH ESO INS DIN NAME"]
            if chip == "HAWAII-2RG":
                resolution = hdr["HIERARCH ESO INS DIL NAME"]

            action = matisseAction(hdr, elt["input"][0][1])
            if "TELESCOP" in hdr:
                tel = hdr["TELESCOP"]
            else:
                tel = ""
            recipes, param = matisseRecipes(
                action, hdr["HIERARCH ESO DET CHIP NAME"], tel, resolution
            )
            elt["action"] = action
            elt["recipes"] = recipes
            if action == "ACTION_MAT_RAW_ESTIMATES":
                if hdr["HIERARCH ESO DET CHIP NAME"] == "AQUARIUS":
                    if spectralBinning != "":
                        paramN += " --spectralBinning=" + spectralBinning
                    else:
                        paramN += " --spectralBinning=7"

                    if paramN == "":
                        elt["param"] = param
                    else:
                        elt["param"] = paramN + " " + param
                else:
                    if spectralBinning != "":
                        paramL += " --spectralBinning=" + spectralBinning
                    else:
                        paramL += " --spectralBinning=5"

                    if paramL == "":
                        elt["param"] = param
                    else:
                        elt["param"] = paramL + " " + param
            else:
                elt["param"] = param
            elt["tplstart"] = keyTplStartCurrent

            # Fill with GRA4MAT data
        print("Searching GRA4MAT data...")
        for elt in listRedBlocks:
            hdr = elt["input"][0][2]
            keyTplStartCurrent = hdr["HIERARCH ESO TPL START"]
            for fileGV, hdrGV in zip(listGRA4MAT, listhdrGRA4MAT, strict=False):
                if hdrGV["HIERARCH ESO TPL START"] == keyTplStartCurrent:
                    elt["input"].append([fileGV, "RMNREC", hdrGV])

        # Fill the list of calib in the Reduction Blocks List from dirCalib
        print("listing calibrations in the reduction blocks...")
        for elt in tqdm(
            listRedBlocks, unit=" block", unit_scale=False, desc="Working on"
        ):
            hdr = elt["input"][0][2]
            calib, status = matisseCalib(
                hdr,
                elt["action"],
                listArchive,
                elt["calib"],
                elt["tplstart"].split(".")[0],
            )
            # print('calib = ',calib)
            elt["calib"] = calib
            elt["status"] = status
        print("done.")

        # Fill the list of calib in the Reduction Blocks List from dirResult Iter i-1
        print("listing calibrations from previous iteration in the reduction blocks...")
        if iterNumber > 1:
            for elt in listRedBlocks:
                hdr = elt["input"][0][2]
                calib, status = matisseCalib(
                    hdr,
                    elt["action"],
                    listIter,
                    elt["calib"],
                    elt["tplstart"].split(".")[0],
                )
                elt["calib"] = calib
                elt["status"] = status
            print("done.")

        # Create the SOF files
        print("creating the sof files and directories...")
        repIter = dirResult + "/Iter" + str(iterNumber)
        if os.path.isdir(repIter) == True:
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
        for elt in listRedBlocks:
            overwritei = overwrite
            if elt["status"] == 1:
                cptStatusOne += 1

                filelist = os.listdir(repIter)
                rbname = elt["recipes"] + "." + elt["tplstart"]
                sofname = os.path.join(repIter, rbname + ".sof").replace(":", ":")
                outputDir = os.path.join(repIter, rbname + ".rb").replace(":", "_")

                if overwritei == 0:
                    print("\nTesting if last reduction went through...")
                    if glob.glob(
                        os.path.join(outputDir, "*_RAW_INT_*.fits")
                    ) or glob.glob(os.path.join(outputDir, "IM_BASIC.fits")):
                        print("Yes!")

                    else:
                        overwritei = 1

                resol = "no res"
                if os.path.exists(sofname):
                    print("sof file " + sofname + " already exists...")
                    if overwritei:
                        print("WARNING: Overwriting existing files")

                        fp = open(sofname, "w")
                        for frame, tag, hdr in elt["input"]:
                            fp.write(frame + " " + tag + "\n")
                            # print(frame, hdr['HIERARCH ESO INS DIL NAME'])
                            resol = hdr["HIERARCH ESO INS DIL NAME"]
                        for frame, tag in elt["calib"]:
                            fp.write(frame + " " + tag + "\n")
                        fp.close()
                    else:
                        print(
                            "WARNING: sof file exists. Skipping... (consider using --overwrite)"
                        )
                        # continue;
                else:
                    print("sof file " + sofname + " does not exist. Creating it...")
                    fp = open(sofname, "w")
                    for frame, tag, hdr in elt["input"]:
                        fp.write(frame + " " + tag + "\n")
                        # print(frame, hdr['HIERARCH ESO INS DIL NAME'])
                        resol = hdr["HIERARCH ESO INS DIL NAME"]
                    for frame, tag in elt["calib"]:
                        fp.write(frame + " " + tag + "\n")
                    fp.close()

                if os.path.exists(outputDir):
                    print("outputDir " + outputDir + " already exists...")
                    # Remove any previous logfile
                    print("Remove any previous logfile...")
                    try:
                        os.remove(os.path.join(outputDir, ".logfile"))
                    except Exception:
                        print("Nothing to remove...")
                    if os.listdir(outputDir) == []:
                        print("outputDir is empty, continuing...")
                    else:
                        print("outputDir already exists and is not empty...")
                        if overwritei:
                            print("WARNING: Overwriting existing files")
                        else:
                            print(
                                "WARNING: outputDir exists. Skipping... (consider using --overwrite)\n"
                            )
                            continue
                else:
                    print(
                        "outputDir " + outputDir + " does not exist. Creating it...\n"
                    )
                    os.mkdir(outputDir)

                listNewParams = removeDoubleParameter(elt["param"].replace("/", " --"))

                # cmd="esorex --output-dir="+outputDir+" "+elt['recipes']+" "+elt['param'].replace("/"," --")+" "+sofname+"%"+resol;
                cmd = (
                    "esorex --output-dir="
                    + outputDir
                    + " "
                    + elt["recipes"]
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
                        + elt["recipes"]
                        + "."
                        + elt["tplstart"]
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
                            elt["iter"] = iterNumber
                            cptToProcess += 1
                            listCmdEsorex.append(cmd)
                    else:
                        listIterNumber[cpt] = iterNumber
                        elt["iter"] = iterNumber
                        cptToProcess += 1
                        listCmdEsorex.append(cmd)
                else:
                    listIterNumber[cpt] = iterNumber
                    elt["iter"] = iterNumber
                    cptToProcess += 1
                    listCmdEsorex.append(cmd)
            else:
                cptStatusZero += 1
            cpt += 1
        print("%-40s" % ("Reduction Blocks to process:",), cptToProcess)

        if listCmdEsorex != [] and iterNumber <= maxIter:
            # Create a process pool with a maximum of 10 worker processes
            pool = Pool(processes=nbCore)
            # Map our function to a data set - number 1 through 20
            pool.map(runEsorex, listCmdEsorex)
            pool.close()
            pool.join()

        print("%-40s" % ("Reduction Blocks processed:",), cptStatusOne)
        print("%-40s" % ("Reduction Blocks not processed:",), cptStatusZero)

        # Add MDFC Fluxes to CALIB_RAW_INT and TARGET_RAW_INT
        listOifitsFiles = glob.glob(repIter + "/*.rb/*_RAW_INT*.fits")
        for elt in listOifitsFiles:
            hdu = fits.open(elt, mode="update")
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
            except:
                try:
                    print("Object " + targetname + " not found in MDFC catalog")
                except:
                    print("There is no target name")
            hdu.close()

        print("Iter number", iterNumber, len(listCmdEsorex))
        if listCmdEsorex == [] or iterNumber == maxIter:
            print(" ")
            print("No more iteration to do")
            print("-----------------------")
            print(" ")
            print("Processing summary:")
            print(" ")
            for elt in listRedBlocks:
                if elt["status"] == 1:
                    msg = "Processing done at iteration " + str(elt["iter"])
                else:
                    if elt["action"] == "NO-ACTION":
                        msg = "Data not taken into account by the Pipeline"
                    else:
                        msg = "Reduction Block not processed - Missing calibration"
                tplstart, detector = elt["tplstart"].split(".")
                print(
                    "%-24s" % (tplstart,),
                    "%-14s" % (detector,),
                    "%-30s" % (elt["action"],),
                    msg,
                )

            break


# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting...")

    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Automatic MATISSE pipeline implementation, allowing one to reduce entire nights of raw data into oifits files."
    )

    # --------------------------------------------------------------------------
    parser.add_argument(
        "dirRaw", default="", help="The path to the directory containing your raw data."
    )

    # --------------------------------------------------------------------------
    parser.add_argument("--dirCalib", default="", help="Calibration Map Path")

    # --------------------------------------------------------------------------
    parser.add_argument(
        "--dirResult", default="", help="Results Path (default is current directory)"
    )

    # --------------------------------------------------------------------------
    parser.add_argument(
        "--nbCore", default=0, type=int, help="Number Of Cores (default 1)"
    )

    # --------------------------------------------------------------------------
    parser.add_argument("--tplID", default="", help="template ID")

    # --------------------------------------------------------------------------
    parser.add_argument("--tplSTART", default="", help="template start")

    # --------------------------------------------------------------------------
    parser.add_argument(
        "--overwrite", default=0, help="overwrite existing files", action="store_true"
    )

    # --------------------------------------------------------------------------
    parser.add_argument(
        "--skipL", default=0, help="skip L band data", action="store_true"
    )

    # --------------------------------------------------------------------------
    parser.add_argument(
        "--skipN", default=0, help="skip N band data", action="store_true"
    )

    # --------------------------------------------------------------------------
    parser.add_argument(
        "--resol",
        default="",
        help="reduce only a given spectral resolution. Can be any of LOW, MED or HIGH",
    )
    # --------------------------------------------------------------------------
    parser.add_argument(
        "--spectralBinning", default="", help="Bin spectrally the data to improve SNR"
    )

    # --------------------------------------------------------------------------
    parser.add_argument(
        "--maxIter", default=0, help="Maximum Number of Iteration (default 1)"
    )

    # --------------------------------------------------------------------------
    parser.add_argument(
        "--paramN",
        default="/useOpdMod=TRUE",
        help="recipes parameters for N band (default /useOpdMod=TRUE for the UTs, and /replaceTel=3/useOpdMod=TRUE for the ATs)",
    )

    # --------------------------------------------------------------------------
    parser.add_argument(
        "--paramL",
        default="/tartyp=57/useOpdMod=FALSE",
        help="recipes parameters for LM band (default /tartyp=57/useOpdMod=FALSE/compensate=[pb,nl,if,rb,bp,od]/hampelFilterKernel=10)",
    )

    # --------------------------------------------------------------------------

    try:
        args = parser.parse_args()
    except:
        print(
            "\n\033[93mRunning mat_autoPipeline.py --help to be kind with you:\033[0m\n"
        )
        parser.print_help()
        print(
            "\n     Example : python mat_autoPipeline.py /data/2018-05-19 --skipN --resol=LOW --nbCore=2 --paramN=/useOpdMod=TRUE/corrFlux=TRUE --paramL=/cumulBlock=TRUE"
        )
        sys.exit(0)

    args.dirRaw = os.path.abspath(args.dirRaw) + "/"

    mat_autoPipeline(
        args.dirRaw,
        args.dirResult,
        args.dirCalib,
        args.nbCore,
        args.resol,
        args.paramL,
        args.paramN,
        args.overwrite,
        int(args.maxIter),
        args.skipL,
        args.skipN,
        args.tplSTART,
        args.tplID,
        args.spectralBinning,
    )
