#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of the Matisse pipeline GUI series
Copyright (C) 2017- Observatoire de la Côte d'Azur

Created on Thu Dec  6 22:22:19 2018
@author: fmillour

Displays UV coverage

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.

You can use, modify and/ or redistribute the software under the terms
of the CeCILL license as circulated by CEA, CNRS and INRIA at the
following URL "http://www.cecill.info". You have a copy of the licence
in the LICENCE.md file.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.
"""

import wx
import argparse
import sys
import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from   astropy.io import fits
from mat_fileDialog import mat_FileDialog, identifyFile
from libShowOifits import open_oi

###############################################################################

def open_hdr(oi_file):
    try:
        hdu = fits.open(oi_file)
    except IOError:
        print(("Unable to read fits file: " + oi_file))
        return {}

    hdr = hdu[0].header

    return hdr

###############################################################################

def get_UV(oi_file):

    BX = []
    BY = []

    dic = open_oi(oi_file);
    res = open_hdr(oi_file)

    if(res=={}):
        return 0,0,0,0,0,0;

    try:
        target_name = dic['TARGET']
    except:
        print("WARNING: No TARGET name. Trying to get it from the header...")
        try:
            target_name = res['HIERARCH ESO OBS TARG NAME']
        except:
            print("WARNING: Did not find TARGET name...")
            target_name = "Dummy target"

    try:
        typ    = res['HIERARCH ESO PRO CATG']
    except:
        print("WARNING: No Product category keyword")
        typ = "-"

    try:
        instru = res['INSTRUME']
    except:
        try:
            instru = res['HIERARCH ESO INS MODE']
        except:
            print('WARNING: unknown instrument!')

    if   instru == 'MATISSE':
        ntels = 4;
    elif instru == 'MIDI':
        ntels = 2;
    elif instru == 'AMBER':
        ntels = 3;
    elif instru == 'MIRCX':
        ntels = 6;
    else:
        try:
            ntels = res['HIERARCH ESO DET NTEL']
        except:
            print('WARNING: NTEL not set. Setting it to 1.')
            ntels=1;

    #read in priority the keywords
    try:
        BX = dic['VIS2']['U'];
        BY = dic['VIS2']['V'];
    except:
        print('No data in OI_VIS2 table. Trying ESO keywords...')
        base=0;
        for i in np.arange(1,ntels+1):
            for j in np.arange(i+1,ntels+1):
                tel1 = i;
                tel2 = j;
                base+=1;

                blen = (res['HIERARCH ESO ISS PBL'+str(tel1)+str(tel2)+' START']+res['HIERARCH ESO ISS PBL'+str(tel1)+str(tel2)+' END'])/2
                bang = (res['HIERARCH ESO ISS PBLA'+str(tel1)+str(tel2)+' START']+res['HIERARCH ESO ISS PBLA'+str(tel1)+str(tel2)+' END'])/2

                bx = blen * np.sin(bang * np.pi / 180.)
                by = blen * np.cos(bang * np.pi / 180.)

                BX = np.append(BX, bx)
                BY = np.append(BY, by)

    #read in priority the keywords
    try:
        WLEN = dic['WLEN']
    except:
        print('No wavelength table found. Trying some other tricks');
        WLEN = res['EFF_WAVE'];

    #print(len(BX))
    #print(len(BY))
    #print(target_name)
    #print(len(typ))
    #print(ntels)

    return BX,BY,WLEN,target_name[0], typ, ntels

###############################################################################

def get_UVs(files):
    BX   = []
    BY   = []
    TARG = []
    WLEN = []

    for file in tqdm(files, unit="file", unit_scale=False, desc="Working on"):
        bx, by, wl, targ, typ, ntels = get_UV(file)
        if(bx[0]==0):
            continue;

        if typ == 'CALIB_RAW_INT':
            continue

        if targ == '-':
            continue

        print(np.shape(BX));
        BX = np.append(BX, bx, axis=0)
        BY = np.append(BY, by, axis=0)

        TARG = np.append(TARG, targ)

        WLEN =  np.append(WLEN,wl)

    return BX, BY, WLEN, TARG, ntels;

###############################################################################

def plot_UV(BX, BY, WLEN, TARG, ntels, marker='o', markersize=4, color="red",title='True', freq='m'):

    BX0 = np.copy(BX);
    if freq=='as':
        BX = BX[:,None]/WLEN * np.pi/180/3600;
        BY = BY[:,None]/WLEN * np.pi/180/3600;
    if freq=='rad':
        BX = BX[:,None]/WLEN;
        BY = BY[:,None]/WLEN;

    uniques = set(TARG)

    if len(uniques) < 4:
        nwiny = len(uniques)
        nwinx = 1
    else:
        nwin = int(np.sqrt(len(uniques)))
        nwiny = nwin;
        nwinx = nwin+1;

    print(np.shape(BX0));

    for i,base in enumerate(BX0):
        for j,uni in enumerate(uniques):
            #print('bla')
            #print(ntels,i)
            #print(TARG[int(i/(ntels*(ntels-1)/2))])
            if uni == TARG[0]:
                idx = j

        try:
            ax = plt.subplot(nwinx,nwiny,idx+1)
            ax.axis('equal')
            aruni = np.array(list(uniques))
            #print(aruni)
            if title=='True':
                ax.set_title(aruni[idx])
            #print(aruni[idx])

            if freq=='m':
                ax.set_xlabel('Baseline (m)')
                ax.set_ylabel('Baseline (m)')
            else:
                ax.set_xlabel('Frequency (cycles/'+freq+')')
                ax.set_ylabel('Frequency (cycles/'+freq+')')


            ax.plot( BX[i], BY[i], marker=marker, markersize=markersize,   color=color)
            ax.plot(-BX[i],-BY[i], marker='o',    markersize=markersize-2, color=color)
        except:
            print("ouille !")

    ax.plot(0,0, marker='+', markersize=markersize, color='black')

###############################################################################

if __name__ == '__main__':
    print("Plotting UV...")

    #--------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='(u,v) plane plot.')
    #--------------------------------------------------------------------------
    parser.add_argument('in_dir', metavar='in_dir', type=str,
                        help='The path to the directory containing your oifits data.',
                        default='.')
    #--------------------------------------------------------------------------
    parser.add_argument('--showtitle', metavar='showtitle',
                        help='Display a title with the name of the star or not (True or False).', default=True)
    #--------------------------------------------------------------------------
    parser.add_argument('--pdf',   action="store_true",
                        help='Create a pdf file for each target.')
    #--------------------------------------------------------------------------
    parser.add_argument('--color', metavar='color', type=str,
                        help='Plot color.',
                        default='red')
    #--------------------------------------------------------------------------
    parser.add_argument('--freq', metavar='freq', type=str,
                        help='Type of scale to use. Can be any of m (meters), as (cycles/arcseconds) or rad (cycles/radians).',
                        default='m')
    #--------------------------------------------------------------------------

    try:
        args = parser.parse_args()
    except:
        print("\n\033[93mRunning mat_showUV.py --help to be kind with you:\033[0m\n")
        parser.print_help()
        sys.exit(0)

    ########################################################################

    files = glob.glob(args.in_dir+"/*.fits*")

    #print(files)

    plt.figure(1)
    plt.clf();

    BX, BY, WLEN, TARG, ntels = get_UVs(files)
    plot_UV(BX, BY, WLEN, TARG, ntels,title=args.showtitle,color=args.color,
            freq=args.freq)

    print("Done!")
    plt.show();
