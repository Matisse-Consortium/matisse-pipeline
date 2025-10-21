import numpy as np
from astropy.io import fits
import os


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
filename="catalog.fits"
if (os.path.exists(filename)):
    os.remove(filename)
hdul.writeto(filename)
