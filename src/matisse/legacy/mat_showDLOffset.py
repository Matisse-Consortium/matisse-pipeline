#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
This file is part of the Matisse pipeline GUI series
Copyright (C) 2017- Observatoire de la Côte d'Azur

Created on Sun Apr 14 14:16:01 2019
@author: ame

DL offset display

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




import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import sys
np.set_printoptions(threshold=np.inf)

"""
file="MATISSE_OBS_RMNREC104_0055.fits"
"""

def checkifrmn(f):
    h=fits.open(f)
    try:
        dprtyp=h[0].header['HIERARCH ESO DPR TYPE']
        tracker=h[0].header['HIERARCH ESO DEL FT SENSOR']
        if tracker=='MATISSE' and (dprtyp=='STD,RMNREC' or dprtyp=='OBJECT,RMNREC'):
            return 1
        else:
            return 0
    except:
        print('missing keywords, not able to check if file is rmnrec')
        print('will not continue')
        sys.exit(0)



def mat_showDLOffset(filename,removeAvg=True,relative=False):
    data=fits.open(filename)
    fig, ax=plt.subplots(4,1, sharex=True,sharey=removeAvg)
    fig.suptitle("{0}".format(filename))
    fig.text(0.01,0.5,"Offset ($\mu$m)",rotation=90,horizontalalignment='center',size="15")
    iplot=0
    offset0=[0]
    idx=np.where(data["OPD_DL1"].data['TIME']!=0)
    print(idx)
    time=np.take( data["OPD_DL1"].data['TIME'],idx)[0]/1e6

    for i in range(6):
        if np.sum(data["DL{0}".format(i+1)].data['position'])!=0:
             offset=np.take(data["OPD_DL{0}".format(i+1)].data['rtOffset'],idx)[0]*1e6
             avg=np.mean(offset)
             if removeAvg:
                 offset=offset-avg
             if offset0[0]==0:
                 offset0=offset
             if (relative):
                offset=offset-offset0
             ax[iplot].plot(time,offset,marker="+",linestyle='',color='red')
             #ax[i].set_ylim()
             ax[iplot].set_ylabel("DL{0}".format(i+1))
             ax[iplot].text(-3,avg,"DL{0}".format(i+1),horizontalalignment='left',verticalalignment='center',color='black',size="20")
             print(iplot, 'DL{}'.format(i+1),avg)
             iplot+=1

    ax[3].set_xlabel("Time (s)",size="15")
    ax[3].set_xlim(-4,np.max(time))
    plt.show()
    print(data["OPD_DL1"].data['TIME'])


if  __name__== '__main__' :


    arg=sys.argv

    if (arg[1]=="--help" or arg[1]== "-h"):
        print( "mat_plotRmnecOpd script to visualize OD offset from RMNREC file")
        print( "Usage : filename [-options]")
        print( "options :")
        print( " --removeAvg [True False]")
        print( " --relative [True False]")
    else:
        filename=arg[1]
        if not(checkifrmn(filename)):
            print('not an rmn file (or maybe the tracker was not MATISSE) \n please check again the filename, the tracker (del.ft.sensor) and its dpr.type')
            sys.exit(0)

        narg=len(arg)
        removeAvg=False
        relative=False

        for i in range(2,narg):
            if (arg[i] == '--removeAvg' ):
                removeAvg=arg[i+1]
            if (arg[i] == '--relative' ):
                relative=arg[i+1]

        mat_showDLOffset(filename,removeAvg=True,relative=False)
