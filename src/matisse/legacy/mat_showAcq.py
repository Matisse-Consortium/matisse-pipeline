#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of the Matisse pipeline GUI series
Copyright (C) 2017- Observatoire de la Côte d'Azur

Created on Fri May 11 05:35:18 2018

Acquisition display

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
import os,time,stat
from datetime import datetime
import sys
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy import ndimage
from mpl_toolkits import mplot3d
import fnmatch
import numpy as np
from astropy.io import fits
from scipy.optimize import minimize
import matplotlib.patches as patches
from scipy import ndimage
import scipy.optimize as opt
import argparse
from tqdm import tqdm
import subprocess
# import commands

def getSkyFile(filename):
    h=fits.open(filename)
    print('new version de getskyfile')
    splitt=filename.split('/')
    if len(splitt)==1:
        c=os.getcwd()
    else:
        c='/'.join(splitt[0:-1])
        print(c)

    dprtype=h[0].header['HIERARCH ESO DPR TYPE']
    if dprtype=='SKY' or dprtype=='SEARCH,SKY':
        print('you are trying to plot a SEARCH/SKY acquisition, please choose a target file')
        sys.exit(0)

    if dprtype=='STD' or dprtype=='OBJECT':
        tpl=h[0].header['HIERARCH ESO TPL START']
        detec=h[0].header['HIERARCH ESO DET CHIP NAME']
        print('no chopping')
        cmd='dfits '+c+'/*.fits | fitsort dpr.type tpl.start det.chip.name | grep '+detec+'| grep '+tpl+' | grep SKY '
        test=os.system(cmd)
        if test==256:
            print('missing sky for this unchopped acquisiiton')
            print('can not continue')
            sys.exit(0)
        else:
            p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            out, err = p.communicate()
            out=out.decode("utf-8")
            skyfile=(out.split('\n')[0].split('\t')[0])




    return str(skyfile)





def reference(detecteur):
    a=1
    b=1
    try:
        blabla=np.loadtxt('mtmcfgINS_REF_IMG.cfg',usecols=0,dtype=str)
        print('using updated ref pos, found the config file')
        trouver=1
    except:
        print('using latest stored ref pos, file not found')
        trouver=0
    if trouver==1:
        keywords=np.loadtxt('mtmcfgINS_REF_IMG.cfg',usecols=0,dtype=str)[11:]
        values=np.loadtxt('mtmcfgINS_REF_IMG.cfg',usecols=1,dtype=str)[11:]
        valuues=np.zeros(len(values),dtype=float)
        for vv,v in enumerate(values):
            valuues[vv]=float(v[:-2])
        dicoref=dict(zip(keywords,valuues))

        if detecteur=='AQUARIUS':
            m10x=(dicoref['OCS.DET2.IMG.REFX1']-1)*a
            m9x=(dicoref['OCS.DET2.IMG.REFX2']-1)*a
            m12x=(dicoref['OCS.DET2.IMG.REFX3']-1)*a
            m13x=(dicoref['OCS.DET2.IMG.REFX4']-1)*a
            m10y=(dicoref['OCS.DET2.IMG.REFY1']-1-100)*b
            m9y=(dicoref['OCS.DET2.IMG.REFY2']-1-100)*b
            m12y=(dicoref['OCS.DET2.IMG.REFY3']-1-100)*b
            m13y=(dicoref['OCS.DET2.IMG.REFY4']-1-100)*b
        if detecteur=='HAWAII-2RG':
            voie=['10','9','13','12']
            m10x=(dicoref['OCS.DET1.IMG.REFX1']-1)*a
            m9x=(dicoref['OCS.DET1.IMG.REFX2']-1)*a
            m13x=(dicoref['OCS.DET1.IMG.REFX3']-1)*a
            m12x=(dicoref['OCS.DET1.IMG.REFX4']-1)*a
            m10y=(dicoref['OCS.DET1.IMG.REFY1']-1)*b
            m9y=(dicoref['OCS.DET1.IMG.REFY2']-1)*b
            m13y=(dicoref['OCS.DET1.IMG.REFY3']-1)*b
            m12y=(dicoref['OCS.DET1.IMG.REFY4']-1)*b


    else:
        if detecteur=='AQUARIUS':
            m10x=(35.3-1)*a
            m9x=(41.3-1)*a
            m12x=(41.9-1)*a
            m13x=(37.2-1)*a
            m10y=(8-1)*b
            m9y=(8-1)*b
            m12y=(8-1)*b
            m13y=(8-1)*b
        if detecteur=='HAWAII-2RG':
            m10x=(72.9-1)*a
            m9x=(72.2-1)*a
            m13x=(74.7-1)*a
            m12x=(75.4-1)*a
            m10y=(11.9-1)*b
            m9y=(11.8-1)*b
            m13y=(11.6-1)*b
            m12y=(11.8-1)*b

    return(m9x,m9y,m10x,m10y,m12x,m12y,m13x,m13y)

def dimension(detecteur):
    if detecteur == 'HAWAII-2RG':
        DLx=19.68
        DLy=4.92
        tolxG=5.1
        tolyG=1.3
        return(DLx,DLy,tolxG,tolyG)
    else:
        DLx=31.5
        DLy=7.87
        tolxO=5.1
        tolyO=1.3
        tolxG=3
        tolyG=0.8
        return(DLx,DLy,tolxG,tolyG,tolxO,tolyO)


def gaussian(center_x, center_y, width_x, width_y,height,background):
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp( -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)+background

def file_is_empty(path):
    return os.stat(path).st_size==0

def mat_showAcq(filename,pdf=False):
    a=1
    b=1
    sky=[]
    target=[]
    undy=[]
    compteursky=0
    compteurtarget=0
    compteurundy=0


    hdu=fits.open(filename)


    TorS=hdu['IMAGING_DATA'].data['TARTYP']
    etoile=hdu[0].header['HIERARCH ESO OBS TARG NAME']
    tplstart=hdu[0].header['HIERARCH ESO TPL START']
    tracker=hdu[0].header['HIERARCH ESO DEL FT SENSOR']
    detecteur=hdu[0].header['HIERARCH ESO DET CHIP NAME']
    print("Processing file {0}...".format(filename))




    for j in range(np.size(TorS)):
        if TorS[j] == 'S':
            sky=np.append(sky,int(j))
            compteursky=compteursky+1
        elif TorS[j] == 'T':
            target=np.append(target,int(j))
            compteurtarget=compteurtarget+1
        else:
            undy=np.append(undy,int(j))
            compteurundy=compteurundy+1


    sky=list(sky)
    target=list(target)
    undy=list(undy)
    sky=[int(sky[m]) for m in range(compteursky) ]
    target=[int(target[mm]) for mm in range(compteurtarget)]
    undy=[int(undy[mmm]) for mmm in range(compteurundy)]
    csky=np.where(np.diff(sky)!=1)
    ctarget=np.where(np.diff(target)!=1)
    csky=list(np.append(-1,csky[0]))
    ctarget=list(np.append(-1,ctarget[0]))

    if len(csky)!=len(ctarget):
        del sky[-1:]
    if ctarget[-1:]!=target[-1:]:
        ctarget=np.append(ctarget,len(np.diff(target)))



    plt.figure(1,figsize=(15,7))
    for i in tqdm(range(len(csky)-1)):

        vecsky=sky[csky[i]+1:csky[i+1]+1]
        vectarget=target[ctarget[i]+1:ctarget[i+1]+1]
        if detecteur=='HAWAII-2RG':
            frame9=hdu['IMAGING_DATA'].data['DATA9'][vectarget,:,:].astype(float)
            frame10=hdu['IMAGING_DATA'].data['DATA10'][vectarget,:,:].astype(float)
            frame12=hdu['IMAGING_DATA'].data['DATA12'][vectarget,:,:].astype(float)
            frame13=hdu['IMAGING_DATA'].data['DATA13'][vectarget,:,:].astype(float)
            sky9=hdu['IMAGING_DATA'].data['DATA9'][vecsky,:,:].astype(float)
            sky10=hdu['IMAGING_DATA'].data['DATA10'][vecsky,:,:].astype(float)
            sky12=hdu['IMAGING_DATA'].data['DATA12'][vecsky,:,:].astype(float)
            sky13=hdu['IMAGING_DATA'].data['DATA13'][vecsky,:,:].astype(float)
            back9=np.mean(sky9)
            back10=np.mean(sky10)
            back12=(np.mean(sky12))
            back13=(np.mean(sky13))
        else:
            frame9=hdu['IMAGING_DATA'].data['DATA9'][vectarget,100:120,:].astype(float)
            frame10=hdu['IMAGING_DATA'].data['DATA10'][vectarget,100:120,:].astype(float)
            frame12=hdu['IMAGING_DATA'].data['DATA12'][vectarget,100:120,:].astype(float)
            frame13=hdu['IMAGING_DATA'].data['DATA13'][vectarget,100:120,:].astype(float)
            sky9=hdu['IMAGING_DATA'].data['DATA9'][vecsky,100:120,:].astype(float)
            sky10=hdu['IMAGING_DATA'].data['DATA10'][vecsky,100:120,:].astype(float)
            sky12=hdu['IMAGING_DATA'].data['DATA12'][vecsky,100:120,:].astype(float)
            sky13=hdu['IMAGING_DATA'].data['DATA13'][vecsky,100:120,:].astype(float)
            back9=np.mean(sky9)
            back10=np.mean(sky10)
            back12=(np.mean(sky12))
            back13=(np.mean(sky13))


        titreaqua=['IP7','IP5','IP1','IP3']
        titrehawa=['IP3','IP1','IP5','IP7']
        if detecteur=='HAWAII-2RG':
            titretitre=titrehawa
        else:
            titretitre=titreaqua
        refs=reference(detecteur)
        voie=['9','10','12','13']
        for kkk in range(4):
            img=np.mean(vars()['frame'+voie[kkk]],axis=0)-np.mean(vars()['sky'+voie[kkk]],axis=0)
            params = (refs[2*kkk+1],refs[2*kkk],1,4,img.max(),vars()['back'+voie[kkk]])
            errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(img.shape)) - img)
            params, success = opt.leastsq(errorfunction, params)
            g = gaussian(*params)
            im=g(*np.indices(np.shape(img)))
            bx=params[1]*a
            by=params[0]*b

            dims=dimension(detecteur)


            plt.subplot(2,2,kkk+1)
            plt.scatter(bx*a,by*b ,marker='o',color="blue")

            plt.text(bx*a,by*b , str(i), color="red", fontsize=12)
            axes=plt.gca()
            axes.add_artist(patches.Ellipse((refs[2*kkk],refs[2*kkk+1]), dims[0]*a, dims[1]*b , 0,edgecolor = 'magenta', fill=False, zorder = 2))
            axes.add_artist(patches.Ellipse((refs[2*kkk],refs[2*kkk+1]), dims[2]*a, dims[3]*b , 0,edgecolor = 'green', fill=False, zorder = 2))
            if detecteur=='AQUARIUS':
                axes.add_artist(patches.Ellipse((refs[2*kkk],refs[2*kkk+1]), dims[4]*a, dims[5]*b , 0,edgecolor = 'orange', fill=False, zorder = 2))
            plt.xlim(refs[2*kkk]-dims[0]*a/2-2*a,refs[2*kkk]+dims[0]*a/2+2*a)
            plt.ylim(refs[2*kkk+1]-dims[1]*b/2-2*b ,refs[2*kkk+1]+dims[1]*b/2+2*b)
            plt.ylabel(titretitre[kkk])
            plt.suptitle(etoile+' '+tplstart+' '+tracker+' '+detecteur)

    if pdf==False:
        plt.show()
    else:
        savename="{0}_acq.pdf".format(filename.split(".fits")[0])
        print("Saving plot to {0}".format(savename))
        plt.savefig(savename,bbox_inches = "tight")


def mat_showAcq_nochop(filename,pdf=False):


    skyfile=getSkyFile(filename)

    h=fits.open(filename)
    hd=fits.open(skyfile)
    a=1
    b=1
    detecteur=h[0].header['HIERARCH ESO DET CHIP NAME']
    refs=reference(detecteur)
    dims=dimension(detecteur)
    etoile=h[0].header['HIERARCH ESO OBS TARG NAME']
    tplstart=h[0].header['HIERARCH ESO TPL START']
    tracker=h[0].header['HIERARCH ESO DEL FT SENSOR']

    voie=['9','10','12','13']

    titreaqua=['IP7','IP5','IP1','IP3']
    titrehawa=['IP3','IP1','IP5','IP7']
    if detecteur=='HAWAII-2RG':
        titretitre=titrehawa
    else:
        titretitre=titreaqua

    for kkk in range(4):
        data=np.mean(h['IMAGING_DATA'].data['DATA'+voie[kkk]].astype(float),axis=0)
        sky=np.mean(hd['IMAGING_DATA'].data['DATA'+voie[kkk]].astype(float),axis=0)
        img= data-sky
        back=np.mean(sky)

        params = (refs[2*kkk+1],refs[2*kkk],1,4,img.max(),back)
        errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(img.shape)) - img)
        params, success = opt.leastsq(errorfunction, params)
        g = gaussian(*params)
        im=g(*np.indices(np.shape(img)))
        bx=params[1]
        by=params[0]
        plt.subplot(2,2,kkk+1)
        plt.scatter(bx,by ,marker='o',color="blue")
        axes=plt.gca()
        axes.add_artist(patches.Ellipse((refs[2*kkk],refs[2*kkk+1]), dims[0]*a, dims[1]*b , 0,edgecolor = 'magenta', fill=False, zorder = 2))
        axes.add_artist(patches.Ellipse((refs[2*kkk],refs[2*kkk+1]), dims[2]*a, dims[3]*b , 0,edgecolor = 'green', fill=False, zorder = 2))
        if detecteur=='AQUARIUS':
            axes.add_artist(patches.Ellipse((refs[2*kkk],refs[2*kkk+1]), dims[4]*a, dims[5]*b , 0,edgecolor = 'orange', fill=False, zorder = 2))

        plt.xlim(refs[2*kkk]-dims[0]*a/2-2*a,refs[2*kkk]+dims[0]*a/2+2*a)
        plt.ylim(refs[2*kkk+1]-dims[1]*b/2-2*b ,refs[2*kkk+1]+dims[1]*b/2+2*b)

        plt.ylabel(titretitre[kkk])
    plt.suptitle(etoile+' '+tplstart+' '+tracker+' '+detecteur)


    if pdf==False:
        plt.show()
    else:
        savename="{0}_acq.pdf".format(filename.split(".fits")[0])
        print("Saving plot to {0}".format(savename))
        plt.savefig(savename,bbox_inches = "tight")





if  __name__== '__main__' :
    parser = argparse.ArgumentParser(description='mat_showAcq.py : compute and show photocenter of MATISSE image acqusition')

    parser.add_argument('filename', default="",  \
    help='Name of the MATISSE image acquisition file to process')

    parser.add_argument('--pdf', default=0,  action='store_true',\
    help='save plot to pdf')


    try:
        args = parser.parse_args()
    except:
        print("\n\033[93mRunning mat_showAcq.py --help to be kind with you:\033[0m\n")
        parser.print_help()
        print("\n     Example : python mat_showAcq.py  MATIS.2019-11-07T06:25:21.184.fits")
        sys.exit(0)

    arg=sys.argv

    h=fits.open(args.filename)
    chopping=h[0].header['HIERARCH ESO ISS CHOP ST']
    dprtype=h[0].header['HIERARCH ESO DPR TYPE']
    if chopping=='T':
        print('chopping = T')
        mat_showAcq(args.filename,pdf=args.pdf)
    if dprtype=='SKY':
        print('\n\n')
        print('you are trying to plot a SKY acquisition, please choose a target file')
        print('\n\n')
        sys.exit(0)

    if dprtype=='SEARCH,SKY':
        print('\n\n')
        print('you are trying to plot a SKY acquisition, please choose a target file')
        print('\n\n')
        sys.exit(0)

    if chopping=='F' and (dprtype=='STD' or dprtype=='OBJECT'):
        mat_showAcq_nochop(args.filename,pdf=args.pdf)
