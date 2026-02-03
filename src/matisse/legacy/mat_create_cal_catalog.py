import numpy as np
from astropy.io import fits
import os
import argparse
import sys
from argparse import RawTextHelpFormatter

def create_catalog(filetext,filefits):
    f = open('catalog.txt','r')
    tname=[]
    traj=[]
    tdej=[]
    tlmag=[]
    tmmag=[]
    tnmag=[]
    tuddk=[]
    tuddl=[]
    tuddm=[]
    tuddn=[]
    teldd=[]
    for line in f:
        if ( ("#" in line) == False):
            column=line.split(',')
            tname.append(column[0])
            traj.append(column[1])
            tdej.append(column[2])
            if ( "NULL" in column[3] ):
                tlmag.append(None)
            else:
                tlmag.append(float(column[3]))
            if ( "NULL" in column[4] ):
                tmmag.append(None)
            else:
                tmmag.append(float(column[4]))
            if ( "NULL" in column[5] ):
                tnmag.append(None)
            else:
                tnmag.append(float(column[5]))
            tuddk.append(float(column[6]))
            tuddl.append(float(column[7]))
            tuddm.append(float(column[8]))
            tuddn.append(float(column[9]))
            teldd.append(float(column[10]))
    f.close()


    hdu=fits.BinTableHDU.from_columns(
        [fits.Column(name='NAME',format='33A',array=np.asarray(tname)),
         fits.Column(name='RAJ2000',format='14A',array=np.asarray(traj)),
         fits.Column(name='DEJ2000', format='14A', array=np.asarray(tdej)),
         fits.Column(name='LMAG', format='E', array=np.asarray(tlmag)),
         fits.Column(name='MMAG', format='E', array=np.asarray(tmmag)),
         fits.Column(name='NMAG', format='E', array=np.asarray(tnmag)),
         fits.Column(name='UDDK', format='E', array=np.asarray(tuddk)),
         fits.Column(name='UDDL', format='E', array=np.asarray(tuddl)),
         fits.Column(name='UDDM', format='E', array=np.asarray(tuddm)),
         fits.Column(name='UDDN', format='E', array=np.asarray(tuddn)),
         fits.Column(name='E_LDD', format='E', array=np.asarray(teldd))])
    hdu.header['EXTNAME']='JSDC2.fits#1'
    hdr=fits.Header()
    hdr['HIERARCH ESO PRO CATG']='JSDC_CAT'
    hdr['HIERARCH ESO PRO TECH']='CATALOG'
    hdr['HIERARCH ESO PRO TYPE']='JMMC'
    hdr['HIERARCH ESO PRO SCIENCE']=False
    hdr['INSTRUME']='MATISSE'
    primary_hdu = fits.PrimaryHDU(header=hdr)
    hdul = fits.HDUList([primary_hdu, hdu])
    if (os.path.exists(filefits)):
        os.remove(filefits)
    hdul.writeto(filefits)




if __name__ == '__main__':
    print("Starting...")

    #--------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Create Calibrator Catalog in JSDC format.', formatter_class=RawTextHelpFormatter)
    #--------------------------------------------------------------------------
    parser.add_argument('filetxt', metavar='filetxt', type=str,
                        help='''Name fo the input ASCII file containing informations on the calibrators.
Example of input ASCII file:
#NAME            , RAJ2000       , DEJ2000        , LMAG   , MMAG    , NMAG   , UDDK    , UDDL    , UDDM    , UDDN     , E_LDD
#-----------------------------------------------------------------------------------------------------------------------------
IRAS 08534-2405  , 08 55 41.0745 , -24 17 32.3837 , NULL   , NULL    , NULL   , 3.6     , 3.6     , 3.6     , 3.6      , 0.35
                        ''', \
                        default='')
    #--------------------------------------------------------------------------
    parser.add_argument('filefits', metavar='filefits', type=str,
                        help='Name fo the output FITS file.',
                        default='')
    #--------------------------------------------------------------------------
    try:
        args = parser.parse_args()
    except:
        print("\n\033[93mRunning mat_create_cal_catalog.py --help to be kind with you:\033[0m\n")
        parser.print_help()
        sys.exit(0)

    create_catalog(args.filetxt,args.filefits)
