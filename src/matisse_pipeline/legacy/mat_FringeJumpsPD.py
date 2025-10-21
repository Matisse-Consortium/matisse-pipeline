import sys
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
from astropy.time import Time




def smooth_tab(tab,n_smooth):
    kernel=np.ones(n_smooth)/n_smooth
    tab_smoothed=np.convolve(tab,kernel,mode='same')
    return tab_smoothed



def mat_phaseDelay(filename):
    hdu=fits.open(filename)
    ext=(hdu[0].header['HIERARCH ESO DPR TYPE']).split(',')


    if ext[1]!='RMNREC':
        print('this is not an RMN gravity file')
        exit()

    coherence=hdu['IMAGING_DATA_FT'].data['COHERENCE'].astype(float)
    GD=hdu['IMAGING_DATA_FT'].data['GD'].astype(float)* 2.15 * 25 / (2. * np.pi)
    acqstart = hdu[0].header['HIERARCH ESO PCR ACQ START']
    t = Time(acqstart)
    time = t.mjd + hdu['IMAGING_DATA_FT'].data['TIME'] * 1E-6 / 86400.
    for i in range(4):
        vars()['lambda'+str(i+1)]=coherence[:,16*i:16*(i+1)]
        fluxReal= vars()['lambda'+str(i+1)][:,4:10]
        fluxImag= vars()['lambda'+str(i+1)][:,10:16]

        vars()['fluxComplex'+str(i+1)]=fluxReal+1j*fluxImag
        for ii in range(6):
            vars()['fluxComplex'+str(i+1)][:,ii]=smooth_tab(vars()['fluxComplex'+str(i+1)][:,ii],40)
        vars()['fluxComplexConj'+str(i+1)]=np.conj(vars()['fluxComplex'+str(i+1)])
    prod=0
    somme=0
    for i in range(3):
        prod=prod+vars()['fluxComplex'+str(i+1)]*vars()['fluxComplexConj'+str(i+2)]
        somme=somme+vars()['fluxComplex'+str(i+1)]


    GD=np.angle(prod)*2.15*25/(2*np.pi)
    PD=np.angle(somme)*2.15/(2*np.pi)
    hdu.close()

    base_Grav = ['34','24','14', '23', '13', '12']
    for b in range(6):
        plt.subplot(6,1,b+1)
        plt.plot(PD[:,b])
        plt.ylabel(base_Grav[b])

    plt.show()

if  __name__== '__main__' :


    arg=sys.argv

    if (arg[1]=="--help" or arg[1]== "-h"):
        print( "mat_fringe_flux_plot script to visualize flux vs time on fringes exposures")
        print( "Usage : filename [-options]")
        print( "options :")
        print( " --sky skyfilename")
        print( " --unchop")
    else:
        filename=arg[1]

        mat_groupeDelay(filename)
