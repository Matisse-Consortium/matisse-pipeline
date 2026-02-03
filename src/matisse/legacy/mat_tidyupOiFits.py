#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of the Matisse pipeline GUI series
Copyright (C) 2017- Observatoire de la Côte d'Azur

Created on Fri Mar 16 19:38:46 2018
@author: fmillour

Get your oifits files in one directory

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.

You can use, modify and/ or redistribute the software under the terms
of the CeCILL license as circulated by CEA, CNRS and INRIA at the
following URL "http://www.cecill.info". You have a copy of the licence
in the LICENCE.md file.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.
"""

import os
#import wx
import sys
import argparse
from tqdm import tqdm
#from tqdm import trange
#import colorama
from astropy.io import fits
#from mat_fileDialog import mat_FileDialog
from shutil import copyfile
from fnmatch import fnmatch

###############################################################################
# Function to remove all spaces from a given string
def removeSpaces(string):

    # To keep track of non-space character count
    count = 0

    list = []

    # Traverse the given string. If current character
    # is not space, then place it at index 'count++'
    for i in xrange(len(string)):
        if string[i] != ' ':
            list.append(string[i])

    return toString(list)

###############################################################################

def change_oifitsFile_name(oifits):
    direc = os.path.dirname(oifits)

    #print("looking if the file is a MATISSE oifits")
    # First check this is indeed an oifits file
    try:
        hdu     = fits.getheader(oifits)
        if hdu['HIERARCH ESO PRO CATG'] == 'CALIB_RAW_INT' or\
           hdu['HIERARCH ESO PRO CATG'] == 'TARGET_RAW_INT':
            #print('yay let\'s do it!')

            try:
                targ = hdu['HIERARCH ESO OBS TARG NAME']
            except:
                targ = hdu['OBJECT']

            try:
                stationsConfig = hdu['HIERARCH ESO ISS CONF STATION1']+hdu['HIERARCH ESO ISS CONF STATION2']+hdu['HIERARCH ESO ISS CONF STATION3']+hdu['HIERARCH ESO ISS CONF STATION4']
            except:
                stationsConfig = 'noConf'


            chipType = hdu['HIERARCH ESO DET CHIP TYPE'];
            if chipType == 'IR-LM':
                resol = hdu['HIERARCH ESO INS DIL NAME'];
            elif chipType == 'IR-N':
                resol = hdu['HIERARCH ESO INS DIN NAME'];
            try:
                chop = hdu['HIERARCH ESO ISS CHOP ST']
            except:
                chop = 'F'
            if (chop == 'F'):
                chopMode='noChop'
            else:
                chopMode='Chop'


            newName = os.path.join(direc,
                                hdu['HIERARCH ESO TPL START'].replace(':','') +
                                '_' + targ.replace(" ","") +
                                '_' + stationsConfig +
                                '_' + chipType +
                                '_' + resol +
                                '_' + hdu['HIERARCH ESO INS BCD1 NAME'] +
                                '_' + hdu['HIERARCH ESO INS BCD2 NAME'] +
                                '_' + chopMode +
                                  '.fits');

            #print("renaming "+os.path.basename(oifits)+" into " +os.path.basename(newName))
            #copyfile(src, dst)
            os.rename(oifits, newName)
    except:
        do_nothing = 1;
        print("AaaaAaAaAAAAAaaargh I'm dead!")

###############################################################################

if __name__ == '__main__':
    print("Starting mat_tidyupOiFits...")

    #--------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='A small utility tool to gather all MATISSE oifits files in a single directory.')

    #--------------------------------------------------------------------------
    parser.add_argument('oiFits', default="",  \
    help='The path to the file or directory containing your oifits data.')


    try:
        args = parser.parse_args()
    except:
        print("\n\033[93mRunning mat_tidyupOiFits.py --help to be kind with you:\033[0m\n")
        parser.print_help()
        print("\n     Example : python mat_tidyupOiFits.py /data/2018-05-19_OIFITS")
        sys.exit(0)

    if os.path.isfile(args.oiFits):
        print("Reading file "+args.oiFits+"...")
        dic = change_oifitsFile_name(args.oiFits)

    elif os.path.isdir(args.oiFits):
        cwd = os.getcwd()
        # Generate a new directory name: The directory where the fits files are + the extension _OIFITS
        newdir = os.path.join(cwd, os.path.basename(os.path.abspath(args.oiFits))+"_OIFITS");
        try:
            print(newdir+" already exists...")
            os.stat(newdir)
        except:
            print("Creating directory "+newdir)
            os.mkdir(newdir)

        print("Oifits files will be copied in that directory.")

        filecounter = 0
        allfiles = [];
        for root,subfolders,files in os.walk(args.oiFits):
            for fil in files:
                allfiles.append(os.path.join(root,fil))
                filecounter += 1;

        print("Number of files to treat:",len(allfiles))

        #print(os.path.join(args.oiFits,"*.fits*"))
        for fil in tqdm(allfiles, total=filecounter, unit=" files", unit_scale=False, desc="Working on files"):
        #for fil in allfiles:
            matchfilestoavoid = ["TARGET_CAL_0*","OBJ_CORR_FLUX_0*",
                                 "OI_OPDWVPO_*","PHOT_BEAMS_*",
                                 "CALIB_CAL_0*","RAW_DPHASE_*",
                                 "matis_eop*","nrjReal*",
                                 "DSPtarget*","nrjImag*",
                                 "fringePeak*","BSreal*","BSimag*"]

            fifil = os.path.basename(fil)

            if fifil.endswith('fits'):
                #print(fifil)
                for i in matchfilestoavoid:
                    if fnmatch(fifil, i):
                        #print("Breeaaaaaak!")
                        break;

                try:
                    #print("copying file to the right path...")
                    hdu    = fits.getheader(fil)
                    if hdu['HIERARCH ESO PRO CATG'] == 'CALIB_RAW_INT' or\
                       hdu['HIERARCH ESO PRO CATG'] == 'TARGET_RAW_INT':
                        copyfile(fil,
                                 os.path.join(newdir,fifil))
                        change_oifitsFile_name(os.path.join(newdir,fifil))
                except:
                    do_nothing = 1;
                    #print("Not a fits file!")

    print("I made my job!")
