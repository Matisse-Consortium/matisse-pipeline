#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
This file is part of the Matisse pipeline GUI series
Copyright (C) 2017- Observatoire de la Côte d'Azur

Created on Tue Sep 06 09:04:52 2016
@author: pbe

MATISSE bad pixel map display

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

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages


def checkfisbadmap(f):
    h=fits.open(f)
    try:
        machin=h[0].header['HIERARCH ESO PRO CATG']
        if machin=='BADPIX':
            return 1
        else:
            return 0
    except:
        print('missing keywords in header, impossible to check if file is badpixel map')
        print('byebye')
        exit()



f=sys.argv[1]

if not(checkfisbadmap(f)):
    print('this is not a bad pixel map you are trying to display')
    exit()


hdulist = fits.open(f)#'C:\Users\pbe\Desktop\Python-TestPlan\MATISSE_GEN_cal_shift_L_SPCAL_OPEN_HIGH_210_0001.fits')
nBeam=5
szSpectral=[]
cornerSpectral=[]
data=[]
for i in range(nBeam):
    szSpectral.append([hdulist['IMAGING_DETECTOR'].data['NAXIS'][i+8][0],hdulist['IMAGING_DETECTOR'].data['NAXIS'][i+8][1]])
    cornerSpectral.append([hdulist['IMAGING_DETECTOR'].data['CORNER'][i+8][0],hdulist['IMAGING_DETECTOR'].data['CORNER'][i+8][1]])

data.append(hdulist['IMAGING_DATA'].data['DATA9'][0])
data.append(hdulist['IMAGING_DATA'].data['DATA10'][0])
data.append(hdulist['IMAGING_DATA'].data['DATA11'][0])
data.append(hdulist['IMAGING_DATA'].data['DATA12'][0])
data.append(hdulist['IMAGING_DATA'].data['DATA13'][0])
hdulist.close()


imgFullFrame=np.zeros((2048,2048),dtype=np.float)

for i in range(nBeam):
    imgFullFrame[cornerSpectral[i][1]-1:cornerSpectral[i][1]-1+szSpectral[i][1],cornerSpectral[i][0]-1:cornerSpectral[i][0]-1+szSpectral[i][0]]=data[i]
plt.figure(1)
plt.imshow(imgFullFrame[1000:1100,1000:1100])

hdulist = fits.open('C:\Users\pbe\Desktop\Python-TestPlan\matisse_bpm_det1_rid2.fits')
bpm=hdulist[0].data
hdulist.close()
plt.figure(2)
plt.imshow(bpm[1000:1100,1000:1100])


hdulist = fits.open('C:\Users\pbe\Desktop\Python-TestPlan\imageraw.fits')
raw=hdulist[0].data
imgraw=np.zeros((2048,2048),dtype=np.float)
imgraw[41:41+1960,750:750+625]=raw[:,300:300+625]
hdulist.close()
