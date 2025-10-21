#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of the Matisse pipeline GUI series
Copyright (C) 2018- Observatoire de la Côte d'Azur

Created in 2016
@author: P. Berio

Automatic MATISSE calibration and Merge!

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.

You can use, modify and/ or redistribute the software under the terms
of the CeCILL license as circulated by CEA, CNRS and INRIA at the
following URL "http://www.cecill.info". You have a copy of the licence
in the LICENCE.md file.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.
"""

from   subprocess import call
import argparse
import glob
import os
import sys
from tqdm import tqdm
import numpy as np
from astropy.io import fits
from multiprocessing.pool import Pool


#------------------------------------------------------------------------------

def make_sof(input_dir, output_dir, band, timespan=0.04):

    SOFFILE = []
    TPLST = []
    CH = []
    DI = []
    MJD = []
    files =  glob.glob(input_dir+'/*'+band+'*.fits')
    #print input_dir#, files

    DIC = []
    # First read all files
    print("Scanning files in "+input_dir+" ...")
    for f in files:
        hdr     = fits.open(f)[0].header
        dic = {'hdr': hdr}
        DIC.append(dic)

    tplstartDone=[]
    for i,f in enumerate(files):
        hdri = DIC[i]['hdr']
        obstype = hdri['ESO PRO CATG']
        if obstype == 'TARGET_RAW_INT':
            mjd  = hdri['MJD-OBS']
            chip = hdri['ESO DET CHIP NAME']
            dit  = hdri['ESO DET SEQ1 DIT']
            tplstart = hdri['ESO TPL START']
            if (tplstart not in tplstartDone):
                tplstartDone.append(tplstart)
                filename  = os.path.basename(f)
                name, ext = os.path.splitext(filename)
                token=name.split('_')
                str=token[0]+'_'+token[1]+'_'+token[2]+'_'+token[3]+'_'+token[4]
                fname ='%s/%s_cal_oifits.sof'%(output_dir, str)
                SOFFILE.append(fname)
                TPLST.append(tplstart)
                CH.append(chip)
                DI.append(dit)
                MJD.append(mjd)
    for j,fname in enumerate(SOFFILE):
        tplstart=TPLST[j]
        chip=CH[j]
        dit=DI[j]
        FILE = []
        OBSTYPE = []
        mjd=MJD[j]
        for i,f in enumerate(files):
            hdri = DIC[i]['hdr']
            obstype = hdri['ESO PRO CATG']

            if obstype == 'TARGET_RAW_INT':
                if (hdri['ESO TPL START'] == tplstart):
                    FILE.append(f)
                    OBSTYPE.append("TARGET_RAW_INT")


            if obstype == 'CALIB_RAW_INT':
                mjdc  = hdri['MJD-OBS']
                chipc = hdri['ESO DET CHIP NAME']
                ditc  = hdri['ESO DET SEQ1 DIT']
                dif = mjd - mjdc
                absdif = np.abs(dif)
                if chip == chipc and dit == ditc :
                    if (absdif < float(timespan)):
                        FILE.append(f)
                        OBSTYPE.append("CALIB_RAW_INT")
        soffile = open(fname, 'w')
        for i,fil in enumerate(FILE):
            soffile.write('{} \t {} \n'.format("../"+fil,OBSTYPE[i]))
        soffile.close()

    return SOFFILE

#------------------------------------------------------------------------------

if __name__ == '__main__':
    print("Starting...")

    #--------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Wrapper to run the calibration steps of the MATISSE DRS on a given directory containong raw OIFITS files.')

    #--------------------------------------------------------------------------
    parser.add_argument('in_dir', metavar='in_dir', type=str, \
    help='The path to the directory containing your oifits data.', default=None)

    #--------------------------------------------------------------------------
    parser.add_argument('-o', '--out_dir', dest='out_dir', metavar='Out Dir', type=str, default=None, \
    help='The path to the directory you want results to be stored in (defaults to current directory).')

    #--------------------------------------------------------------------------
    #parser.add_argument('-i', '--in_dir', dest='in_dir', metavar='Working Directory', type=str, \
    #help='The path to the directory containing your oifits data.')

    #--------------------------------------------------------------------------
    parser.add_argument('--timespan', dest='timespan', metavar='Timespan of calibs', type=str, default='.', \
    help='The time search interval for selecting calibrators around the science star')
    #--------------------------------------------------------------------------

    try:
        args = parser.parse_args()
    except:
        print("\n\033[93mRunning mat_autoCalib.py --help to be kind with you:\033[0m\n")
        parser.print_help()
        sys.exit(0)

    if args.out_dir == None:
        args.out_dir = os.path.dirname(args.in_dir) + "_CALIBRATED"

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # N band
    #######################################################################
    #----------------------------------------------------------------------
    #---- Make the SOF files ----------------------------------------------
    if (args.timespan=='.'):
        targsof = make_sof(args.in_dir, args.out_dir,'-N')
    else:
        targsof = make_sof(args.in_dir, args.out_dir, '-N', args.timespan)

    #--------------------------------------------------------------------------
    #----- Run the Recipes ----------------------------------------------------
    os.chdir(args.out_dir)
    for isof in tqdm(targsof, unit="file", unit_scale=False, desc="Calibrating"):
        #call("esorex --output-dir=%s mat_cal_oifits %s >> log.log"%(args.out_dir,isof), shell=True)
        cmd="esorex mat_cal_oifits --cumulBlock=TRUE ../"+isof+" >> log.log"
        #print(cmd)
        call(cmd, shell=True)
        token1=isof.split('/')
        token2=token1[1].split('_')
        name=token2[0]+'_'+token2[1]+'_'+token2[2]+'_'+token2[3]+'_'+token2[4]
        print(name)
        # Rename files
        resultFiles = glob.glob('TARGET_CAL_INT_????.fits')
        for idx,f in enumerate(resultFiles):
            hdr = fits.open(f)[0].header
            if (hdr['ESO ISS CHOP ST'] == 'F'):
                os.rename(f, name+"_"+hdr['ESO CFG BCD MODE']+'_nochop.fits')
            else:
                os.rename(f, name+"_"+hdr['ESO CFG BCD MODE']+'_chop.fits')
        os.rename('TARGET_CAL_INT_noBCD.fits', name+'.fits')

    # cleanup intermediate files
    try :
        os.remove("CAL_CPHASE.fits")
        os.remove("CAL_DPHASE.fits")
        os.remove("CAL_VIS.fits")
    except:
        pass
    os.chdir("../")

    # LM band
    #######################################################################
    #----------------------------------------------------------------------
    #---- Make the SOF files ----------------------------------------------
    if (args.timespan=='.'):
        targsof = make_sof(args.in_dir, args.out_dir,'-LM')
    else:
        targsof = make_sof(args.in_dir, args.out_dir, '-LM', args.timespan)

    #--------------------------------------------------------------------------
    #----- Run the Recipes ----------------------------------------------------
    os.chdir(args.out_dir)
    for isof in tqdm(targsof, unit="file", unit_scale=False, desc="Calibrating"):
        #call("esorex --output-dir=%s mat_cal_oifits %s >> log.log"%(args.out_dir,isof), shell=True)
        cmd="esorex mat_cal_oifits --cumulBlock=TRUE ../"+isof+" >> log.log"
        #print(cmd)
        call(cmd, shell=True)
        token1=isof.split('/')
        token2=token1[1].split('_')
        name=token2[0]+'_'+token2[1]+'_'+token2[2]+'_'+token2[3]+'_'+token2[4]
        print(name)
        # Rename files
        resultFiles = glob.glob('TARGET_CAL_INT_????.fits')
        for idx,f in enumerate(resultFiles):
            hdr = fits.open(f)[0].header
            if (hdr['ESO ISS CHOP ST'] == 'F'):
                os.rename(f, name+"_"+hdr['ESO CFG BCD MODE']+'_nochop.fits')
            else:
                os.rename(f, name+"_"+hdr['ESO CFG BCD MODE']+'_chop.fits')
        os.rename('TARGET_CAL_INT_noBCD.fits', name+'.fits')

    # cleanup intermediate files
    try :
        os.remove("CAL_CPHASE.fits")
        os.remove("CAL_DPHASE.fits")
        os.remove("CAL_VIS.fits")
    except:
        pass
