#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of the Matisse pipeline GUI series
Copyright (C) 2017- Observatoire de la Côte d'Azur

Created on Wed Jul 11 23:06 2018
@author: fmillour

Get your calibration maps in one directory

This software is a computer program whose purpose is to show oifits
files from the MATISSE instrument.

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.

You can use, modify and/ or redistribute the software under the
terms of the CeCILL license as circulated by CEA, CNRS and INRIA at
the following URL "http://www.cecill.info". You have a copy of the
licence in the LICENCE.md file.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.
"""

import os
import sys
from astropy.io import fits
from mat_fileDialog import mat_FileDialog
from shutil import copyfile


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

def change_calibmap_name(calibmap):
    direc = os.path.dirname(calibmap)

    # First check this is indeed a calib map file
    try:
        hdu = fits.getheader(calibmap)
        if hdu['HIERARCH ESO PRO CATG'] == 'OBS_FLATFIELD':
            print('yay let\'s do it!')
            print(hdu['HIERARCH ESO DET SEQ1 DIT'])
            if hdu['HIERARCH ESO DET CHIP NAME'] == 'HAWAII-2RG':
                # PIL
                obstype = hdu['HIERARCH ESO INS PIL NAME']
                resol   = hdu['HIERARCH ESO INS DIL NAME']
            elif hdu['HIERARCH ESO DET CHIP NAME'] == 'AQUARIUS':
                obstype = hdu['HIERARCH ESO INS PIN NAME']
                resol   = hdu['HIERARCH ESO INS DIN NAME']


            newName = os.path.join(direc,
                                hdu['DATE-OBS'].replace(':','_') +
                                '_' + hdu['HIERARCH ESO PRO CATG'] +
                                '_' + hdu['HIERARCH ESO DET CHIP NAME'] +
                                '_' + hdu['HIERARCH ESO DET READ CURNAME'] +
                                '_DC9_' + str(hdu['HIERARCH ESO DET CLDC1 DC9']) +
                                '_DIT_' + str(hdu['HIERARCH ESO DET SEQ1 DIT']) +
                                '_' + obstype + '_' + resol +
                                   '.fits');


            print("renaming "+calibmap+" into " +newName)
            #copyfile(src, dst)
            os.rename(calibmap, newName)
        else:
            print("Not a calibmap file!")
    except:
        print("Not a fits file!")

if __name__ == '__main__':
    # Get command line arguments
    listArg   = sys.argv
    name_file = []
    for elt in listArg:
        if ('--help' in elt):
            print( "Usage: Tidyup_calibmap.py <directory>")
            sys.exit(0)
        elif len(listArg) == 2:
            name_file = sys.argv[1]
            print(name_file)

    if not name_file:
        print("Error need file or directory")
    else:

        if os.path.isfile(name_file):
            print("Reading file "+name_file+"...")
            dic = change_calibmapFile_name(name_file)

        elif os.path.isdir(name_file):
            newdir = os.path.join(name_file,"CALIBMAP");
            try:
                print(newdir+" already exists...")
                os.stat(newdir)
            except:
                print("Creating directory "+newdir)
                os.mkdir(newdir)
            print("Listing files in directory")

            print(os.path.join(name_file,"**/*.fits*"))
            for root,subfolders,files in os.walk(name_file):
                for fileToTreat in files:
                    #print(fileToTreat)
                    if (fileToTreat.startswith('OBS_FLATFIELD') and (fileToTreat.endswith('fits')) ):
                        try:
                            hdu    = fits.getheader(os.path.join(root,fileToTreat))
                            if hdu['HIERARCH ESO PRO CATG'] == 'OBS_FLATFIELD':
                                print('Found an calibmap file. Copying it...')
                                #print(fileToTreat)
                                fil = os.path.basename(fileToTreat)

                                copyfile(os.path.join(root,fileToTreat),
                                         os.path.join(newdir,fil))
                                change_calibmap_name(os.path.join(newdir,fileToTreat))
                        except:
                            print("Not a proper fits file!")

print("I made my job!")
