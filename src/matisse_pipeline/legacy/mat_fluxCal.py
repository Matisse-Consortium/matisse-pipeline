#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from shutil import copyfile
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.special import j0,j1
from scipy.interpolate import interp1d
import math
import os
from astroquery.simbad import Simbad
from numpy.polynomial.polynomial import polyval
from astropy.convolution import Gaussian1DKernel,Box1DKernel,convolve
import scipy.stats
from libFluxCal import *
import argparse
import sys
import importlib
import imp
import shutil
from operator import itemgetter


###################### Main ################################################################

if __name__ == '__main__':
    print("Starting...")

    #--------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Spectrophotometric calibration of MATISSE total and correlated spectra.')

    #--------------------------------------------------------------------------
    parser.add_argument('dir_oifits', default="",  \
    help='The path to the directory containing the MATISSE oifits files of the science target and of the spectrophotometric calibrator')

    #--------------------------------------------------------------------------
    #parser.add_argument('dir_caldatabases', default="",  \
    #help='The path to the directory containing the calibrator synthetic spectra databases')

    #--------------------------------------------------------------------------
    parser.add_argument('--sciname', default="",  \
    help='Name of the science target (as indicated in the oifits file name).')

    #--------------------------------------------------------------------------
    parser.add_argument('--calname', default="",  \
    help='Name of the spectrophotometric calibrator (as indicated in the oifits file name). If empty, the closest calibrator in time will be used.')

    #--------------------------------------------------------------------------
    parser.add_argument('--mode', default="flux", \
                        help='Type of MATISSE spectra you want to calibrate. "flux": total spectra or "corrflux": correlated spectra.')
    #--------------------------------------------------------------------------
    parser.add_argument('--band',default='LM', \
                        help='Spectral band to calibrate (either "LM" or "N").')
    #--------------------------------------------------------------------------
    parser.add_argument('--airmassCorr',  \
                        help='Do airmass correction between the science and the calibrator', action='store_true')
    #--------------------------------------------------------------------------
    parser.add_argument('--timespan', default=1.,type=float,  \
    help='Maximum time difference (in h) between the SCI and CAL observations.')
    #--------------------------------------------------------------------------

    try:
        args = parser.parse_args()
    except:
        print("\n\033[93mRunning mat_fluxCal.py --help to be kind with you:\033[0m\n")
        #parser.print_help()
        print("\n This routine can produce two types of calibrated oifits files depending on the selected mode (either 'flux' or 'corrflux':\n")
        print("\n - ***_calflux.fits: only total flux is calibrated (incoherently processed oifits file expected) and stored in the OI_FLUX table (FLUXDATA column).\n")
        print("\n - ***_calcorrflux.fits: only correlated fluxes are calibrated (coherently processed oifits file expected) and stored in the OI_VIS table (VISAMP column).\n")
        print("\n Example of calibration of the total flux of a MATISSE science oifits files with a specified calibrator and an airmass correction in LM band:\n")
        print(" mat_fluxCal.py dir --sciname='sci' --calname='cal' --mode='flux' --band='LM' --airmassCorr\n")
        sys.exit(0)

    #-----------------------------------------
    #Path to the calibrators spectra databases
    #-----------------------------------------
    #a=imp.find_module("libFluxCal")
    #dir_caldatabases=os.path.dirname(a[1])+'/calib_spec_databases'
    a=importlib.util.find_spec("libFluxCal")
    dir_caldatabases=os.path.dirname(a.origin)+'/calib_spec_databases'
    #dir_caldatabases='/data/users/ama/dev_python/tools/mat_tools/calib_spec_databases'

    #---------------------
    # Oifits files sorting
    #---------------------
    if args.calname == "":  #No calibrator name specified
        print('No calibrator specified. The closest calibrator in time will thus be used.')
        calfiles=[]
        args.dir_oifits = os.path.abspath(args.dir_oifits)+"/"
        if args.band == 'LM':
            scifiles=glob.glob(args.dir_oifits+'*'+args.sciname+'*_IR-LM*Chop*.fits')
            list_files=glob.glob(args.dir_oifits+'*_IR-LM*Chop*.fits')
            for f in list_files:
                hdu_f=fits.open(f)
                catg_f=hdu_f[0].header['HIERARCH ESO PRO CATG']
                if catg_f == 'CALIB_RAW_INT':
                    calfiles.append(f)
        elif args.band == 'N':
            scifiles=glob.glob(args.dir_oifits+'*'+args.sciname+'*_IR-N*Chop*.fits')
            list_files=glob.glob(args.dir_oifits+'*_IR-N*Chop*.fits')
            for f in list_files:
                hdu_f=fits.open(f)
                catg_f=hdu_f[0].header['HIERARCH ESO PRO CATG']
                if catg_f == 'CALIB_RAW_INT':
                    calfiles.append(f)
    else:
        args.dir_oifits = os.path.abspath(args.dir_oifits)+"/"
        if args.band == 'LM':
            scifiles=glob.glob(args.dir_oifits+'*'+args.sciname+'*_IR-LM*Chop*.fits')
            calfiles=glob.glob(args.dir_oifits+'*'+args.calname+'*_IR-LM*Chop*.fits')
        elif args.band == 'N':
            scifiles=glob.glob(args.dir_oifits+'*'+args.sciname+'*_IR-N*Chop*.fits')
            calfiles=glob.glob(args.dir_oifits+'*'+args.calname+'*_IR-N*Chop*.fits')

    nfiles_sci=np.size(scifiles)
    nfiles_cal=np.size(calfiles)
    list_of_dicts_sci=[]
    list_of_dicts_cal=[]
    if args.band == 'LM':
        id_disp='DIL'
    else:
        id_disp='DIN'

    for i in range(nfiles_sci):
        hdul_sci=fits.open(scifiles[i])
        hdr_sci=hdul_sci[0].header
        dateobs_sci=hdr_sci['MJD-OBS']
        bcd_pos1=hdr_sci['HIERARCH ESO INS BCD1 ID']
        bcd_pos2=hdr_sci['HIERARCH ESO INS BCD2 ID']
        chop_status=hdr_sci['HIERARCH ESO ISS CHOP ST']
        tpl_start=hdr_sci['HIERARCH ESO TPL START']
        res=hdr_sci['HIERARCH ESO INS '+id_disp+' NAME']
        filename=scifiles[i].split("/")[-1]
        dic_sci = {'DATE_OBS':dateobs_sci}
        dic_sci['TPL_START']=tpl_start
        dic_sci['FILENAME']=filename
        dic_sci['BCD']=bcd_pos1+'-'+bcd_pos2
        dic_sci['CHOP_ST']=chop_status
        dic_sci['RES']=res
        list_of_dicts_sci.append(dic_sci)

    for i in range(nfiles_cal):
        hdul_cal=fits.open(calfiles[i])
        hdr_cal=hdul_cal[0].header
        dateobs_cal=hdr_cal['MJD-OBS']
        bcd_pos1=hdr_cal['HIERARCH ESO INS BCD1 ID']
        bcd_pos2=hdr_cal['HIERARCH ESO INS BCD2 ID']
        chop_status=hdr_cal['HIERARCH ESO ISS CHOP ST']
        tpl_start=hdr_cal['HIERARCH ESO TPL START']
        res=hdr_cal['HIERARCH ESO INS '+id_disp+' NAME']
        filename=calfiles[i].split("/")[-1]
        dic_cal = {'DATE_OBS':dateobs_cal}
        dic_cal['TPL_START']=tpl_start
        dic_cal['FILENAME']=filename
        dic_cal['BCD']=bcd_pos1+'-'+bcd_pos2
        dic_cal['CHOP_ST']=chop_status
        dic_cal['RES']=res
        list_of_dicts_cal.append(dic_cal)

    sorted_list_of_dicts_sci = sorted(list_of_dicts_sci,key=itemgetter('DATE_OBS'))
    sorted_scifiles=[]
    sorted_scitime=[]
    sorted_bcd_sci=[]
    sorted_res_sci=[]
    sorted_chop_sci=[]
    for dic in sorted_list_of_dicts_sci:
        files = dic['FILENAME']
        tpl_start_sci = dic['TPL_START']
        scitime = dic['DATE_OBS']
        bcd_sci = dic['BCD']
        res_sci = dic['RES']
        chop_sci = dic['CHOP_ST']
        #print('files={0}').format(files)
        sorted_scifiles.append(files)
        sorted_scitime.append(scitime)
        sorted_bcd_sci.append(bcd_sci)
        sorted_res_sci.append(res_sci)
        sorted_chop_sci.append(chop_sci)
        #sorted_tpl_start_sci.append(files)
    sorted_list_of_dicts_cal = sorted(list_of_dicts_cal,key=itemgetter('DATE_OBS'))
    sorted_calfiles=[]
    sorted_caltime=[]
    sorted_bcd_cal=[]
    sorted_res_cal=[]
    sorted_chop_cal=[]
    for dic in sorted_list_of_dicts_cal:
        files = dic['FILENAME']
        tpl_start_cal = dic['TPL_START']
        caltime = dic['DATE_OBS']
        bcd_cal = dic['BCD']
        res_cal = dic['RES']
        chop_cal = dic['CHOP_ST']
        #print('files={0}').format(files)
        sorted_calfiles.append(files)
        sorted_caltime.append(caltime)
        sorted_bcd_cal.append(bcd_cal)
        sorted_res_cal.append(res_cal)
        sorted_chop_cal.append(chop_cal)
        #sorted_tpl_start_cal.append(files)



    #-----------------
    # Flux calibration
    #-----------------

    nfiles_sci=np.size(sorted_scifiles)
    nfiles_cal=np.size(sorted_calfiles)
    sorted_scitime=np.array(sorted_scitime)
    sorted_caltime=np.array(sorted_caltime)
    sorted_bcd_sci=np.array(sorted_bcd_sci)
    sorted_bcd_cal=np.array(sorted_bcd_cal)
    sorted_res_sci=np.array(sorted_res_sci)
    sorted_res_cal=np.array(sorted_res_cal)
    sorted_chop_sci=np.array(sorted_chop_sci)
    sorted_chop_cal=np.array(sorted_chop_cal)

    if args.mode == 'flux':
       outputdir=args.dir_oifits+'calflux/'
    elif args.mode == 'corrflux':
        outputdir=args.dir_oifits+'calcorrflux/'
    if (not os.path.isdir(outputdir)):
        os.mkdir(outputdir)
    for i in range(nfiles_sci):
        ind_bcd=np.where(sorted_bcd_cal == sorted_bcd_sci[i])
        ind_res=np.where(sorted_res_cal[ind_bcd[0][:]] == sorted_res_sci[i])
        ind_chop=np.where(sorted_chop_cal[ind_bcd[0][ind_res]] == sorted_chop_sci[i])
        delta_time=np.abs(sorted_scitime[i]-sorted_caltime[ind_bcd[0][ind_res][ind_chop]])
        try:
            ind=np.argmin(delta_time)
        except:
            print('No calibrator file could be found to calibrate the following SCI file: '+sorted_scifiles[i])
        if delta_time[ind]*24. < args.timespan:
            if args.mode == 'flux':
                outputfile=sorted_scifiles[i].replace(".fits","")+'_calflux.fits'
                #outputfile=sorted_scifiles[i].split(".")[0]+'_calflux.fits'
            elif args.mode == 'corrflux':
                outputfile=sorted_scifiles[i].replace(".fits","")+'_calcorrflux.fits'
                #outputfile=sorted_scifiles[i].split(".")[0]+'_calcorrflux.fits'
            print('-------------------------------------------------------------------------------------------------------------------')
            print('Delta time between SCI and CAL is shorter than the specified timespan (',args.timespan,' h) for the following pair:')
            print('Sci = {0}'.format(sorted_scifiles[i]))
            print('Cal = {0}'.format(sorted_calfiles[ind_bcd[0][ind_res][ind_chop][ind]]))
            #print('Calibration performed with the following calibrator spectra database = {0}'.format(dir_caldatabases))
            calfile=sorted_calfiles[ind_bcd[0][ind_res][ind_chop][ind]]
            fluxcal(args.dir_oifits+sorted_scifiles[i],args.dir_oifits+calfile,outputdir+outputfile, dir_caldatabases, mode=args.mode,output_fig_dir='',match_radius=25.0,do_airmass_correction=args.airmassCorr)
        else:
            print('------------------------------------------------------------------------------------------------------------')
            print('Delta time between SCI and CAL exceeds the specified timespan (',args.timespan,' h) for the following pair:')
            print('Sci = {0}'.format(sorted_scifiles[i]))
            print('Cal = {0}'.format(sorted_calfiles[ind_bcd[0][ind_res][ind_chop][ind]]))
            print('No calibration performed')
            continue
