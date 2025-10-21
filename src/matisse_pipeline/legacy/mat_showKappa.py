#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
This file is part of the Matisse pipeline GUI series
Copyright (C) 2017- Observatoire de la Côte d'Azur

Created on Fri Jul 22 09:59:00 2016
@author: pbe

Displays a Kappa matrix fits file content

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
import wx
from mat_fileDialog import mat_FileDialog


def show_kappa(filename):

    #if filename is void
    #dlg = mat_FileDialog(None,"Choose a Kappa matrix file")
    #filename = dlg.GetPaths()

    hdulist = fits.open(filename)

    nBeam          = 4
    regName        = []
    szSpectral     = []
    coefKappa      = []
    errCoefKappa   = []
    cornerSpectral = []
    for i in range(nBeam):
        iReg = hdulist['KAPPA_MATRIX'].data['REGION'][i]
        ix   = np.where(hdulist['IMAGING_DETECTOR'].data['REGION'][:] == iReg)
        regName.append(hdulist['IMAGING_DETECTOR'].data['REGNAME'][ix[0]][0])
        szSpectral.append(hdulist['IMAGING_DETECTOR'].data['NAXIS'][ix[0]][0][1])
        coefKappa.append(hdulist['KAPPA_MATRIX'].data['MATRIX'][i])
        errCoefKappa.append(hdulist['KAPPA_MATRIX'].data['ERROR'][i])
        cornerSpectral.append(hdulist['IMAGING_DETECTOR'].data['CORNER'][ix[0]][0][1])

    detName=hdulist[0].header['HIERARCH ESO DET CHIP NAME']
    curName=hdulist[0].header['HIERARCH ESO DET READ CURNAME']
    if (detName == 'AQUARIUS'):
        specResolution=hdulist[0].header['HIERARCH ESO INS DIN ID']
        filterType=hdulist[0].header['HIERARCH ESO INS FIN ID']
        polarType=hdulist[0].header['HIERARCH ESO INS PON ID']
    else:
        specResolution=hdulist[0].header['HIERARCH ESO INS DIL ID']
        filterType=hdulist[0].header['HIERARCH ESO INS FIL ID']
        polarType=hdulist[0].header['HIERARCH ESO INS POL ID']

    dispCoef0=hdulist[0].header['HIERARCH PRO DISP COEF0']
    dispCoef1=hdulist[0].header['HIERARCH PRO DISP COEF1']
    dispCoef2=hdulist[0].header['HIERARCH PRO DISP COEF2']

    x=np.arange(szSpectral[0])+cornerSpectral[0]
    wavelength=dispCoef0+dispCoef1*x+dispCoef2*x*x

    configType=detName+"/"+curName+" --- "+specResolution+"/"+filterType+"/"+polarType
    hdulist.close()

    f, axarr = plt.subplots(3, sharex=True, figsize=(8,8))
    f.suptitle('KAPPA MATRIX',fontsize=20)
    for i in range(nBeam):
    #    axarr[0].plot(wavelength,coefKappa[i][0:szSpectral[i]],label=regName[i],marker=(4,i,0),linestyle='-')
         #axarr[0].plot(wavelength,coefKappa[i][0:szSpectral[i]],label=regName[i],linestyle='-')
        axarr[0].errorbar(wavelength,coefKappa[i][0:szSpectral[i]],yerr=errCoefKappa[i][0:szSpectral[i]],label=regName[i],linestyle='-')
    axarr[0].plot([wavelength[0],wavelength[szSpectral[0]-1]],[1.5,1.5],linestyle='--',color='black')
    axarr[0].plot([wavelength[0],wavelength[szSpectral[0]-1]],[3,3],linestyle='--',color='black')
    if (wavelength[szSpectral[0]-1] > 10):
        axarr[0].plot([8,8],[0,4.5],linestyle='--',color='black')
        axarr[0].plot([13,13],[0,4.5],linestyle='--',color='black')
    else :
        axarr[0].plot([2.8,2.8],[0,4.5],linestyle='--',color='black')
        axarr[0].plot([5,5],[0,4.5],linestyle='--',color='black')
    axarr[0].set_ylim([0,4.5])
    axarr[0].set_xlim([wavelength[0],wavelength[szSpectral[0]-1]])
    axarr[0].set_ylabel('Ratio')
    axarr[0].set_title(configType,fontsize=12)

    for i in range(nBeam):
        print(np.mean(coefKappa[i][szSpectral[i]+85:szSpectral[i]+135]))
        print(np.std(coefKappa[i][szSpectral[i]+85:szSpectral[i]+135]))
    #    axarr[1].plot(wavelength,coefKappa[i][szSpectral[i]:2*szSpectral[i]],label=regName[i],marker=(4,i,0),linestyle='-')
        #axarr[1].plot(wavelength,coefKappa[i][szSpectral[i]:2*szSpectral[i]],label=regName[i],linestyle='-')
        axarr[1].errorbar(wavelength,coefKappa[i][szSpectral[i]:2*szSpectral[i]],yerr=errCoefKappa[i][szSpectral[i]:2*szSpectral[i]],label=regName[i],linestyle='-')
    if (wavelength[szSpectral[0]-1] > 10):
        axarr[1].plot([8,8],[1,11],linestyle='--',color='black')
        axarr[1].plot([13,13],[1,11],linestyle='--',color='black')
    else :
        axarr[1].plot([2.8,2.8],[1,11],linestyle='--',color='black')
        axarr[1].plot([5,5],[1,11],linestyle='--',color='black')
    axarr[1].set_ylim([1,11])
    axarr[0].set_xlim([wavelength[0],wavelength[szSpectral[0]-1]])
    #axarr[1].legend(prop={'size':8},loc=2)
    axarr[1].set_ylabel('Zoom')

    for i in range(nBeam):
    #    axarr[2].plot(wavelength,coefKappa[i][2*szSpectral[i]:3*szSpectral[i]],label=regName[i],marker=(4,i,0),linestyle='-')
        #axarr[2].plot(wavelength,coefKappa[i][2*szSpectral[i]:3*szSpectral[i]],label=regName[i],linestyle='-')
        axarr[2].errorbar(wavelength,coefKappa[i][2*szSpectral[i]:3*szSpectral[i]],yerr=errCoefKappa[i][2*szSpectral[i]:3*szSpectral[i]],label=regName[i],linestyle='-')
    if (wavelength[szSpectral[0]-1] > 10):
        axarr[2].plot([8,8],[-100,100],linestyle='--',color='black')
        axarr[2].plot([13,13],[-100,100],linestyle='--',color='black')
    else :
        axarr[2].plot([2.8,2.8],[-100,100],linestyle='--',color='black')
        axarr[2].plot([5,5],[-100,100],linestyle='--',color='black')
    axarr[2].set_ylim([-30,20])
    axarr[0].set_xlim([wavelength[0],wavelength[szSpectral[0]-1]])
    axarr[2].legend(prop={'size':8},loc=1)
    axarr[2].set_ylabel('Shift')
    axarr[2].set_xlabel('Wavelength ($\mu$m)')

    # pp=PdfPages('C:\Users\pbe\Desktop\Doc-TestPlan\kappa_N_low.pdf')
    pp=PdfPages('test.pdf')
    pp.savefig(f)
    pp.close()

app = wx.App()
openFileDialog = mat_FileDialog(None, 'Open a file',"lmk,")
if openFileDialog.ShowModal() == wx.ID_OK:
    print( openFileDialog.GetPaths())
    filename = openFileDialog.GetPaths()[0]

show_kappa(filename)

openFileDialog.Destroy()
app.MainLoop()
app.Destroy()
