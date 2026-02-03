#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of the Matisse pipeline GUI series
Copyright (C) 2018- Observatoire de la Côte d'Azur

Created in 2016
@author: J. Isbell, F. Millour

Automatic MATISSE calibration !

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

def findClosestCal(DIC,i,way=0):
    hdri = DIC[i]['hdr']
    mjd  = hdri['MJD-OBS']
    bcd1 = hdri['ESO INS BCD1 NAME']
    bcd2 = hdri['ESO INS BCD2 NAME']
    chip = hdri['ESO DET CHIP NAME']
    dit  = hdri['ESO DET SEQ1 DIT']
    try:
        chop = hdri['ESO ISS CHOP ST']
    except:
        chop = 'F'
        print("error")


    goodCalIdx=[]
    mjds=[]

    for j,DICj in enumerate(DIC):
        hdrj = DIC[j]['hdr']
        obstypec = hdrj['ESO PRO CATG']
        mjdc  = hdrj['MJD-OBS']
        bcd1c = hdrj['ESO INS BCD1 NAME']
        bcd2c = hdrj['ESO INS BCD2 NAME']
        chipc = hdrj['ESO DET CHIP NAME']
        ditc  = hdrj['ESO DET SEQ1 DIT']
        try:
            chopc = hdrj['ESO ISS CHOP ST']
        except:
            chopc = 'F'
            print("error")

        if obstypec == 'CALIB_RAW_INT' and bcd1 == bcd1c and bcd2 == bcd2c and chip == chipc and dit == ditc and chop == chopc:
            goodCalIdx.append(j)
            mjds.append(mjdc)
    if len(mjds)!=0:
        diff=np.array(mjds)-mjd

        if way==0: #closet cal
            idx=np.argmin(np.abs(diff))
        if way==1: #closet cal after
            if mjd<np.max(mjds):
                idx=np.where(diff>0,diff,np.inf).argmin()
            else:
                idx=-1
        if way==-1:  #closet cal before
            if mjd>np.min(mjds):
                idx=np.where(diff<0,diff,-np.inf).argmax()
            else:
                idx=-1

        if idx!=-1:
            return goodCalIdx[idx]
        else:
            return -1
    else:
        return -1

#------------------------------------------------------------------------------

def make_sof(input_dir, output_dir, timespan=0.1,interpType="MEAN"):

    SOFFILE = [];

    allfiles = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir,f))]
    #files    = [os.path.join(input_dir,f) for f in allfiles if '.fits' and 'LAMP' not in f]
    files = [os.path.join(input_dir,f) for f in allfiles if '.fits' in f and 'LAMP' not in f]


    DIC = []
    # First read all files
    print("Scanning files in "+input_dir+" ...")
    for f in files:
        hdr     = fits.open(f)[0].header
        dic = {'hdr': hdr}
        DIC.append(dic)
        #try:
    for i,f in enumerate(files):

            hdri = DIC[i]['hdr']
            obstype = hdri['ESO PRO CATG']

            if obstype == 'TARGET_RAW_INT' or obstype == 'TARGET_FLUXCAL_INT':
                mjd  = hdri['MJD-OBS']
                bcd1 = hdri['ESO INS BCD1 NAME']
                bcd2 = hdri['ESO INS BCD2 NAME']
                chip = hdri['ESO DET CHIP NAME']
                dit  = hdri['ESO DET SEQ1 DIT']
                try:
                    chop = hdri['ESO ISS CHOP ST']
                except:
                    chop = 'F'
                    print("error")

                filename  = os.path.basename(f)
                name, ext = os.path.splitext(filename)

                fname ='%s/%s_cal_oifits.sof'%(output_dir, name)

                soffile = open(fname, 'w')
                SOFFILE.append(fname)
                soffile.write('{} \t {} \n'.format(f,'TARGET_RAW_INT'))
                calcount = 0;

                if interpType=="NEAREST":
                    j=findClosestCal(DIC,i,way=0)
                    fcal=files[j]
                    soffile.write('{} \t {} \n'.format(fcal,'CALIB_RAW_INT'))

                elif interpType=="LINEAR":
                    jm=findClosestCal(DIC,i,way=-1)
                    jp=findClosestCal(DIC,i,way=1)
                    if jm!=-1:
                        fcal=files[jm]
                        soffile.write('{} \t {} \n'.format(fcal,'CALIB_RAW_INT'))


                    if jp!=-1:
                        fcal=files[jp]
                        soffile.write('{} \t {} \n'.format(fcal,'CALIB_RAW_INT'))



                else:# interpType="MEAN"
                    for j,fcal in enumerate(files):
                        if fcal != f:
                            hdrj = DIC[j]['hdr']
                            obstypec = hdrj['ESO PRO CATG']
                            mjdc  = hdrj['MJD-OBS']
                            bcd1c = hdrj['ESO INS BCD1 NAME']
                            bcd2c = hdrj['ESO INS BCD2 NAME']
                            chipc = hdrj['ESO DET CHIP NAME']
                            ditc  = hdrj['ESO DET SEQ1 DIT']
                            try:
                                chopc = hdrj['ESO ISS CHOP ST']
                            except:
                                chopc = 'F'
                                print("error")
                            dif = mjd - mjdc
                            absdif = np.abs(dif)
                            if obstypec == 'CALIB_RAW_INT' and bcd1 == bcd1c and bcd2 == bcd2c and chip == chipc and dit == ditc and chop == chopc:
                                if (absdif < float(timespan)/24.):
                                    soffile.write('{} \t {} \n'.format(fcal,obstypec))
                                    calcount+=1

                soffile.close()

            fmat = obstype[:-1]
            if 'RAW_INT' not in fmat:
                f = '#' + f
        #except:
         #   continue
    return SOFFILE

#------------------------------------------------------------------------------

if __name__ == '__main__':
    print("Starting...")

    #--------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Wrapper to run the calibration steps of the MATISSE DRS on a given directory containing raw OIFITS files.')

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
                        help='The time search interval (in hours) for selecting calibrators around the science star (only used for --interpType=MEAN)')
    #--------------------------------------------------------------------------
    parser.add_argument('--interpType', default="MEAN", \
    help='interpolation type of the Transfer Function : should be either MEAN(default), NEAREST or LINEAR ')
    #--------------------------------------------------------------------------

    try:
        args = parser.parse_args()
    except:
        print("\n\033[93mRunning mat_autoCalib.py --help to be kind with you:\033[0m\n")
        parser.print_help()
        sys.exit(0)

    if args.out_dir == None:
        cwd = os.getcwd()
        args.out_dir = os.path.join(cwd, os.path.basename(os.path.abspath(args.in_dir))+"_CALIBRATED");

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)


    #----------------------------------------------------------------------
    #---- Make the SOF files ----------------------------------------------
    if (args.timespan=='.'):
        targsof = make_sof(args.in_dir, args.out_dir, 3, interpType=args.interpType)
    else:
        targsof = make_sof(args.in_dir, args.out_dir, args.timespan,interpType=args.interpType)

    #--------------------------------------------------------------------------
    #----- Run the Recipes ----------------------------------------------------
    for isof in tqdm(targsof, unit="file", unit_scale=False, desc="Calibrating"):
        if args.interpType=="LINEAR":
            #add="--tfInterp=2" # COMMENTED as the tfInterp=2 option give weird results for now
            add=""
        else:
            add=""
        #print("esorex --output-dir=%s  mat_cal_oifits %s %s>> log.log"%(args.out_dir,add,isof))
        try :
            call("esorex --output-dir=%s  mat_cal_oifits %s %s>> log.log"%(args.out_dir,add,isof), shell=True)
        except:
            print("error on execution. Possible reason is: spaces in folder names (not allowed).")

        # Create a process pool with a maximum of 10 worker processes
        #pool = Pool(processes=8)
        # Map our function to a data set - number 1 through 20
        #pool.map(runEsorex, listCmdEsorex)

        name, ext = os.path.splitext(isof)
        #print(name)

        # Rename files
        resultFiles = glob.glob(args.out_dir+'/TARGET_CAL_INT_????.fits',recursive=False)
        for idx,fi in enumerate(resultFiles):
            #print("renaming",fi, name+"_"+str(idx+1)+'.fits')
            os.rename(fi, name+"_"+str(idx)+'.fits')

    os.chdir(args.out_dir)
    # cleanup intermediate files
    try :
        os.system("rm -f CALCPHASE_*.fits")
        os.system("rm -f CALDPHASE_*.fits")
        os.system("rm -f CALVIS_*.fits")
        os.system("rm -f TARGET_CAL_INT_*.fits")
    except:
        pass
    #os.remove("CALIBRATED/*.sof")
