#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is part of the Matisse pipeline GUI series
Copyright (C) 2017- Observatoire de la Côte d'Azur

Created on 2019 June 24th
@author: F. Millour

Redo the data flagging properly (apply it to oifits files)

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.

You can use, modify and/ or redistribute the software under the terms
of the CeCILL license as circulated by CEA, CNRS and INRIA at the
following URL "http://www.cecill.info". You have a copy of the licence
in the LICENCE.md file.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.
"""

# Import stuff
import os
import argparse
from fnmatch import fnmatch
from tqdm import tqdm
import astropy
from astropy.io import fits
import numpy as np
from shutil import copyfile
from os import walk
import glob
import matplotlib.pyplot as plt

###############################################################################

def reflagData(oifits, keepOldFlags=0, debug=0):

    if debug:
        print(oifits)

    data=fits.open(oifits, mode='update')

    # Read whether it is a calibrator or a science star
    catg = data[0].header['ESO PRO CATG']
    #print(catg)

    if debug:
        print("reading data")

    WLEN = data['OI_WAVELENGTH'].data['EFF_WAVE']

    # Read VIS2 data
    VIS2     = data['OI_VIS2'].data['VIS2DATA']
    VIS2ERR  = data['OI_VIS2'].data['VIS2ERR']
    VIS2FLAG = data['OI_VIS2'].data['FLAG']

    if debug:
        print("Computing visibility")
    vis    = np.sqrt(np.abs(VIS2)) * np.sign(VIS2)
    viserr = 0.5* VIS2ERR / (vis+(vis==0));

    wlmin = (2.94e-6, 4.5e-6, 8e-6);
    wlmax = (4.1e-6,  5.0e-6, 12.5e-6);

    if debug:
        print("Getting old vis flags")
    #if(keepOldFlags==1):
    #    flag = ~bool(VIS2FLAG);
    #else:
    #    flag = np.array(1,np.shape(VIS2FLAG));


    flag = np.array(VIS2);

    if debug:
        print("Computing new vis flags")
    flag = (VIS2 > 0. - VIS2ERR) &\
        (VIS2 < 1. + VIS2ERR) &\
        (VIS2 > -0.1)         &\
        (VIS2 < 1.1)          &\
        (VIS2ERR > 0)         &\
        (VIS2ERR < 0.1)       &\
        (vis > 0. - viserr)   &\
        (vis < 1. + viserr)   &\
        (vis > -0.1)          &\
        (vis < 1.1)           &\
        (viserr > 0)          &\
        (viserr < 0.1)

    # Discard any visibility below V2=0.04 (20% contrast)
    if catg=='CALIB_RAW_INT':
        flag = flag & (VIS2>0.04)

    if debug:
        print("Filtering wavelength")
    # Filter interbands
    wlmask = np.zeros(np.shape(WLEN),dtype=bool)
    for i,wl in enumerate(wlmin):
        wlmaski = ((WLEN < wlmax[i]) &\
                   (WLEN > wlmin[i]))
        wlmask = wlmask | wlmaski;
    flag = flag & wlmask;

    # Check there are useful data left from flagging, exit if not
    if(np.count_nonzero(flag) < len(flag.flatten())/2):
        return 1;

    if debug:
        print("Inverting flags")
    # print(flag)
    flag = ~flag

    if debug:
        print("Setting flags")
    data['OI_VIS2'].data['FLAG'] = flag

    ############################################################

    if debug:
        print("Getting CP flags")
    # Read CP data
    CP     = data['OI_T3'].data['T3PHI']
    CPERR  = data['OI_T3'].data['T3PHIERR']
    CPFLAG = data['OI_T3'].data['FLAG']
    CPSTA  = data['OI_T3'].data['STA_INDEX']
    STAIDX = data['OI_ARRAY'].data['STA_INDEX']
    STAPOS = data['OI_ARRAY'].data['STAXYZ']

    if debug:
        print("bla")
        print(STAIDX)
        print(STAPOS)
        print(CPSTA);

        print("bla")
        STAPOSX = np.zeros(np.shape(CPSTA));
        STAPOSY = np.zeros(np.shape(CPSTA));

        print(np.shape(STAPOSX))

        for i,ista in enumerate(CPSTA):
            for j,jsta in enumerate(ista):
                print(i,j)
                STAPOSX[i,j] =  STAPOS[jsta==STAIDX,0];
                STAPOSY[i,j] =  STAPOS[jsta==STAIDX,1];
        #print(i ,ista)
        #print(np.where(CPSTA==ista, STAPOS, STAPOS))
        #STAPOSX = STAPOS[np.where(CPSTA==ista),0]

        print(STAPOSX);
        print(STAPOSY);

        print("bla")
    #STAPOSX = STAPOS[CPSTA[STAIDX]]
    #STAPOSY =

    #if(keepOldFlags==1):
    #    flag3 = ~CPFLAG;
    #else:
    #    flag3 = 1;

    if debug:
        print("Computing new flags")
    flag3 = (CPERR > 0.) & (CPERR < 60)
    flag3 = flag3 & wlmask;

    # Flag also visibilities with bad flag
    flag = ~flag
    flag3 = flag3 & (flag.sum(axis=0,keepdims=1)>0);

    # Discard any closure phase above 10 degrees
    if catg=='CALIB_RAW_INT':
        flag3 = flag3 & (abs(CP) < 10)

    if debug:
        print("Redundant closure phase filtering")
    # Check the redundant closure phase
        nclos = np.shape(CP)[0] // 4;
        print(np.shape(CP)[0])
        cp    = CP * np.pi/180.;
        cperr = CPERR * np.pi/180.;
    #print("2")

    #sgnc = np.ones(np.shape(CPSTA),dtype=np.float64);
        sgn = np.sign(np.prod(np.diff(CPSTA,axis=1),axis=1));
        sgn = np.ones(np.shape(sgn));
        sgn[::4]  = sgn[::4];
        sgn[1::4] = sgn[1::4];
        sgn[2::4] = -sgn[2::4];
        sgn[3::4] = -sgn[3::4];

        print("sign");
        print(sgn)

        phaz  = np.exp(np.expand_dims(sgn,axis=1) * 1j * cp);
        print("3a")
        #print(phaz);
        print(np.shape(phaz))
        for i in np.arange(nclos):
            #print("toto");
            BS = np.conj((phaz[4*i]) * (phaz[4*i+1]) * (phaz[4*i+2]));
            BSERR = np.sqrt(cperr[4*i]**2 + cperr[4*i+1]**2 + cperr[4*i+2]**2)
            print(BS.imag)
            CP_compare = np.arctan2(BS.imag, BS.real);
            plt.figure()
            plt.errorbar(WLEN,CP_compare,BSERR,   color="red");
            plt.errorbar(WLEN,sgn[4*i+3]*cp[4*i+3], cperr[4*i+3],color="blue");
            plt.ylim(np.max(np.array((-np.pi,np.min(cp[4*i+3])))),
                     np.min(np.array((np.pi,np.max((cp[4*i+3]))))));
            #plt.show()

            bsd = BS * np.conj(phaz[4*i+3]);
            cpdiff = np.arctan2(bsd.imag, bsd.real);

            errcomp = BSERR + cperr[i];

            flagi = np.abs(cpdiff) > np.abs(errcomp);

    if debug:
        print("Inverting flag")
    # print(flag)
    flag3 = ~flag3;

    data['OI_T3'].data['FLAG'] = flag3;

    #print("Saving fiel")
    data.flush();
    data.close();

    return 0;


###############################################################################

if __name__ == '__main__':
    print("Starting mat_reflagData...")

    #--------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='A small utility tool to reflag MATISSE data.')

    #--------------------------------------------------------------------------
    parser.add_argument('oiFits', default="", nargs='+',  \
    help='The path to the file or directory containing your oifits data.')


    try:
        args = parser.parse_args()
    except:
        print("\n\033[93mRunning mat_reflagData.py --help to be kind with you:\033[0m\n")
        parser.print_help()
        print("\n     Example : python mat_tidyupOiFits.py /data/2018-05-19_OIFITS")
        sys.exit(0)

    print(args.oiFits);

    # Test if argument is a file
    if(len(args.oiFits)==1):
        if os.path.isfile(args.oiFits[0]):
            print("Reading file "+args.oiFits[0]+"...")
            reflagData(args.oiFits[0])

        # Test if argument is a directory
        elif os.path.isdir(args.oiFits[0]):
            print("Working on directory "+args.oiFits[0]+"...")

            cwd = os.getcwd()

            newdir = os.path.join(cwd, os.path.basename(os.path.abspath(args.oiFits[0]))+"_REFLAGGED");

            try:
                print(newdir+" already exists...")
                os.stat(newdir)
            except:
                print("Creating directory "+newdir)
                os.mkdir(newdir)

            print("Oifits files will be copied and reflagged in that directory.")

            filecounter = 0
            allfiles    = [];
            for root,subfolders,files in os.walk(args.oiFits[0]):
                for fil in files:
                    allfiles.append(os.path.join(root,fil))
                    filecounter += 1;

            print("Number of files to treat:",len(allfiles))

            for fil in tqdm(allfiles, total=filecounter, unit=" files", unit_scale=False, desc="Working on files"):
                matchfilestoavoid = ["TARGET_CAL_0*","OBJ_CORR_FLUX_0*",
                                     "OI_OPDWVPO_*","PHOT_BEAMS_*",
                                     "CALIB_CAL_0*","RAW_DPHASE_*",
                                     "matis_eop*","nrjReal*",
                                     "DSPtarget*","nrjImag*",
                                     "fringePeak*","BSreal*","BSimag*",
                                     "*Image0.fits","*LFF.fits"]

                fifil = os.path.basename(fil)

                if fifil.endswith('fits'):
                    for i in matchfilestoavoid:
                        if fnmatch(fifil, i):
                            break;

                    copyfile(fil,
                             os.path.join(newdir,fifil))
                    remFile = reflagData(os.path.join(newdir,fifil))
                    #print(remFile)
                    if(remFile):
                        os.remove(os.path.join(newdir,fifil))

    print("I made my job!")
