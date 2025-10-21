# -*- coding: utf-8 -*-
"""
This file is part of the Matisse pipeline GUI series
Copyright (C) 2017- Observatoire de la Côte d'Azur

Created on Tue Nov 19 13:50:34 2019
@author: ame

MATISSE BCD treatment tools

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

from astropy.io import fits
from scipy.stats import circstd, circvar
import numpy as np
import os


BCD=[[0,1,2,3,4,5],  #OUT-OUT (0)
     [0,1,4,5,2,3],  #OUT-IN  (1)
     [0,1,3,2,5,4],  #IN-OUT  (2)
     [0,1,5,4,3,2]]  #IN-IN  (3)

BCDsign=[[1,1,1,1,1,1],   #OUT-OUT (0)
         [1,-1,1,1,1,1],  #OUT-IN  (1)
         [-1,1,1,1,1,1], #IN-OUT  (2) => Pourquoi?
         [-1,-1,1,1,1,1]] #IN-IN  (3)

BCDcp=[[0, 1, 2, 3], #OUT-OUT (0)
       [3, 1, 2, 0], #OUT-IN  (1)
       [0, 2, 1, 3], #IN-OUT  (2)
       [3, 2, 1, 0]] #IN-IN  (3)


sta_index_cp=[[1,2,3],
           [0,1,2],
           [0,1,3],
           [0,2,3]]

uv1=[2,1,1,4]
uv2=[0,2,3,0]

BCDcpsign=[[ 1,  1,  1,  1], #OUT-OUT (0)
           [1,  -1,  -1, 1], #OUT-IN  (1)
           [ -1, 1, 1,  -1], #IN-OUT  (2)
           [-1, -1, -1, -1]] #IN-IN  (3)

BCDfluxL=[[0,1,3,2],
          [0,1,2,3],
          [1,0,3,2],
          [1,0,2,3]]

BCDfluxN=[[3,2,0,1],
          [3,2,1,0],
          [2,3,0,1],
          [2,3,1,0]]

################### Sorting oifits files by TPLSTART ##########################

def mat_sortByTplStart(oifitsListOrDir):
    data=[]
    if type(oifitsListOrDir)==type(""):
        oifitsListOrDir=[oifitsListOrDir+"/"+ filei for filei in os.listdir(oifitsListOrDir)]
    if type(oifitsListOrDir[0])==type(""):
        data=[fits.open(oifitsi) for oifitsi in oifitsListOrDir]
    else:
        data=oifitsListOrDir
    tplstart=[d[0].header["ESO TPL START"] for d in data]

    nfiles=len(data)
    tplstartList=[]
    sortData=[]
    for i in range(nfiles):

        if not(tplstart[i] in tplstartList):
            sortData.append([])
            tplstartList.append(tplstart[i] )
        idx=np.where(np.array(tplstartList)==tplstart[i])[0][0]
        sortData[idx].append(data[i])
    return tplstartList,sortData


################### Merging a list of oifits files ###########################


def mat_mergeOifits(oifitsList):
    data=[]
    if type(oifitsList[0])==type(""):
        data=[fits.open(oifitsi) for oifitsi in oifitsList]
    else:
        data=oifitsList

    nfile=len(data)


    for datai in data:
        mat_removeBCD(datai)

    #------------------- preparing merged container-------------

    extnames=[hdu.name for hdu in data[0]]
    avgFits=fits.HDUList([hdu.copy() for hdu in data[0]])

    #------------------------------OI_VIS2------------------------------------
    if "OI_VIS2" in extnames:

        nB=np.array([len(datai["OI_VIS2"].data) for datai in data])
        nBmin=6
        temp=mat_hduCutRows(data[0]["OI_VIS2"],nBmin)
        #mean of the square of vis2 to compute std(vis2) = sqrt(|<vis2>^2-<vis2^2>|)
        vis22=temp.data["VIS2DATA"]**2
        weight_vis2=1./temp.data["VIS2ERR"]**2
        vis2_weighted=temp.data["VIS2DATA"]/temp.data["VIS2ERR"]**2
        norm=1
        for ifile in range(nfile):
            nmod=nB[ifile]//nBmin
            for imod in range(nmod):
                if (ifile!=0) or (imod!=0):
                    for key in ["VIS2DATA","VIS2ERR","UCOORD","VCOORD","TIME","MJD","INT_TIME"]:
                        if len(np.shape(temp.data[key]))==2:
                            temp.data[key]= (temp.data[key]*norm + data[ifile]["OI_VIS2"].data[key][imod*nBmin:(imod+1)*nBmin,:])/(norm+1)
                        else:
                            temp.data[key]= (temp.data[key]*norm + data[ifile]["OI_VIS2"].data[key][imod*nBmin:(imod+1)*nBmin])/(norm+1)
                    vis22 = (vis22*norm + data[ifile]["OI_VIS2"].data["VIS2DATA"][imod*nBmin:(imod+1)*nBmin,:]**2)/(norm+1)
                    weight_vis2 = weight_vis2 + 1./(data[ifile]["OI_VIS2"].data["VIS2ERR"][imod*nBmin:(imod+1)*nBmin,:]**2)
                    vis2_weighted = vis2_weighted + data[ifile]["OI_VIS2"].data["VIS2DATA"][imod*nBmin:(imod+1)*nBmin,:]/(data[ifile]["OI_VIS2"].data["VIS2ERR"][imod*nBmin:(imod+1)*nBmin,:]**2)
                    norm+=1
        #temp.data["VIS2ERR"]= np.sqrt(temp.data["VIS2ERR"]**2+ np.abs(vis22- temp.data["VIS2DATA"]**2))/np.sqrt(norm)
        #temp.data["VIS2ERR"]= np.sqrt(temp.data["VIS2ERR"]**2/norm + np.abs(vis22- temp.data["VIS2DATA"]**2))
        temp.data["VIS2ERR"]= np.sqrt(1./weight_vis2 + np.abs(vis22- temp.data["VIS2DATA"]**2))
        temp.data["VIS2DATA"]= vis2_weighted/weight_vis2

        #temp.data["VIS2ERR"]=np.sqrt(np.abs(vis22- temp.data["VIS2DATA"]**2))/np.sqrt(norm)
        temp.data["INT_TIME"] *=norm
        avgFits["OI_VIS2"]=temp



    #----------------------------OI_VIS---------------------------------------
    if "OI_VIS" in extnames:

        nB=np.array([len(datai["OI_VIS"].data) for datai in data])
        nBmin=6
        temp=mat_hduCutRows(data[0]["OI_VIS"],nBmin)
        viscompl=temp.data["VISAMP"]*np.exp(np.complex(0,1)*np.deg2rad(temp.data["VISPHI"]))
        expvisphi=np.exp(np.complex(0,1)*np.deg2rad(temp.data["VISPHI"]))
        #expvisphi2=(np.exp(np.complex(0,1)*np.deg2rad(temp.data["VISPHI"])))**2
        #mean of the square of visamp to compute the std
        visampi2=temp.data["VISAMP"]**2
        weight_vis=1./temp.data["VISAMPERR"]**2
        vis_weighted=temp.data["VISAMP"]/temp.data["VISAMPERR"]**2
        visphi_arr=[]
        visphi_arr.append(np.deg2rad(temp.data["VISPHI"]))
        norm=1
        for ifile in range(nfile):
            nmod=nB[ifile]//nBmin
            for imod in range(nmod):
                if ((ifile!=0) or (imod!=0)):
                    #print('Vis: calcul moyenne')
                    for key in ["VISAMP","VISAMPERR","VISPHIERR","UCOORD","VCOORD","TIME","MJD","INT_TIME"]:
                        if len(np.shape(temp.data[key]))==2:
                            temp.data[key]= (temp.data[key]*norm + data[ifile]["OI_VIS"].data[key][imod*nBmin:(imod+1)*nBmin,:])/(norm+1)
                        else:
                            temp.data[key]= (temp.data[key]*norm + data[ifile]["OI_VIS"].data[key][imod*nBmin:(imod+1)*nBmin])/(norm+1)
                    visampi =data[ifile]["OI_VIS"].data["VISAMP"][imod*nBmin:(imod+1)*nBmin,:]
                    visampi2=(visampi2*norm+visampi**2)/(norm+1)
                    weight_vis = weight_vis + 1./(data[ifile]["OI_VIS"].data["VISAMPERR"][imod*nBmin:(imod+1)*nBmin,:]**2)
                    vis_weighted = vis_weighted + data[ifile]["OI_VIS"].data["VISAMP"][imod*nBmin:(imod+1)*nBmin,:]/(data[ifile]["OI_VIS"].data["VISAMPERR"][imod*nBmin:(imod+1)*nBmin,:]**2)
                    visphii =np.deg2rad(data[ifile]["OI_VIS"].data["VISPHI"][imod*nBmin:(imod+1)*nBmin,:])
                    #viscompl = (viscompl*norm + visampi*np.exp(np.complex(0,1)*visphii))/(norm+1)
                    expvisphi +=  np.exp(np.complex(0,1)*visphii)
                    #expvisphi2 +=  (np.exp(np.complex(0,1)*visphii))**2
                    visphi_arr.append(visphii)
                    norm+=1
        visphi_arr=np.array(visphi_arr)
        #temp.data["VISAMP"]=np.abs(viscompl)
        temp.data["VISPHI"]=np.rad2deg(np.angle(expvisphi))
        #temp.data["VISPHIERR"]/=np.sqrt(norm)  # no better estimation than that for now
        temp.data["VISPHIERR"]=np.rad2deg(np.sqrt((np.deg2rad(temp.data["VISPHIERR"]))**2/norm + circvar(visphi_arr,axis=0)))
        #temp.data["VISAMPERR"]/=np.sqrt(norm)  # no better estimation than that for now
        #temp.data["VISAMPERR"]= np.sqrt(temp.data["VISAMPERR"]**2/norm + np.abs(visampi2- temp.data["VISAMP"]**2))
        temp.data["VISAMPERR"]= np.sqrt(1./weight_vis + np.abs(visampi2- temp.data["VISAMP"]**2))
        temp.data["VISAMP"]= vis_weighted/weight_vis
        temp.data["INT_TIME"] *=norm
        avgFits["OI_VIS"]=temp

    #-----------------------------OI_T3----------------------------------------
    if "OI_T3" in extnames:

        nB=np.array([len(datai["OI_T3"].data) for datai in data])
        nBmin=4
        temp=mat_hduCutRows(data[0]["OI_T3"],nBmin)
        expt3phi=np.exp(np.complex(0,1)*np.deg2rad(temp.data["T3PHI"]))
        #expt3phi2=(np.exp(np.complex(0,1)*np.deg2rad(temp.data["T3PHI"])))**2
        t3phi_arr=[]
        t3phi_arr.append(np.deg2rad(temp.data["T3PHI"]))
        norm=1
        for ifile in range(nfile):
            nmod=nB[ifile]//nBmin
            for imod in range(nmod):
                if (ifile!=0) or (imod!=0):
                    for key in ["T3PHIERR","U1COORD","V1COORD","U2COORD","V2COORD","TIME","MJD","INT_TIME"]:
                        if len(np.shape(temp.data[key]))==2:
                            temp.data[key]= (temp.data[key]*norm + data[ifile]["OI_T3"].data[key][imod*nBmin:(imod+1)*nBmin,:])/(norm+1)
                        else:
                            temp.data[key]= (temp.data[key]*norm + data[ifile]["OI_T3"].data[key][imod*nBmin:(imod+1)*nBmin])/(norm+1)
                    expt3phi +=  np.exp(complex(0,1)*np.deg2rad(data[ifile]["OI_T3"].data["T3PHI"][imod*nBmin:(imod+1)*nBmin,:]))
                    #expt3phi2 +=  (np.exp(complex(0,1)*np.deg2rad(data[ifile]["OI_T3"].data["T3PHI"][imod*nBmin:(imod+1)*nBmin,:])))**2
                    #t3phi_B[imod,:,:]=np.deg2rad(data[ifile]["OI_T3"].data["T3PHI"][imod*nBmin:(imod+1)*nBmin,:])
                    t3phi_arr.append(np.deg2rad(data[ifile]["OI_T3"].data["T3PHI"][imod*nBmin:(imod+1)*nBmin,:]))
                    norm+=1
        t3phi_arr=np.array(t3phi_arr)
        temp.data["T3PHI"]=np.rad2deg(np.angle(expt3phi))
        temp.data["T3PHIERR"]=np.rad2deg(np.sqrt((np.deg2rad(temp.data["T3PHIERR"]))**2/norm + circvar(t3phi_arr,axis=0)))  # no better estimation than that for now
        #temp.data["T3PHIERR"]/=np.sqrt(norm)  # no better estimation than that for now
        temp.data["INT_TIME"] *=norm
        avgFits["OI_T3"]=temp

    #-----------------------------OI_FLUX--------------------------------------
    if "OI_FLUX" in extnames:
        nB=np.array([len(datai["OI_FLUX"].data) for datai in data])
        nBmin=4
        temp=mat_hduCutRows(data[0]["OI_FLUX"],nBmin)
        flux2=temp.data["FLUXDATA"]**2
        weight_flux=1./temp.data["FLUXERR"]**2
        flux_weighted=temp.data["FLUXDATA"]/temp.data["FLUXERR"]**2
        norm=1
        for ifile in range(nfile):
            nmod=nB[ifile]//nBmin
            for imod in range(nmod):
                if (ifile!=0) or (imod!=0):  #Problem here
                    for key in ["FLUXDATA","FLUXERR","TIME","MJD","INT_TIME"]:
                        if len(np.shape(temp.data[key]))==2:
                            temp.data[key]= (temp.data[key]*norm + data[ifile]["OI_FLUX"].data[key][imod*nBmin:(imod+1)*nBmin,:])/(norm+1)
                        else:
                            temp.data[key]= (temp.data[key]*norm + data[ifile]["OI_FLUX"].data[key][imod*nBmin:(imod+1)*nBmin])/(norm+1)
                    flux2 = (flux2*norm + data[ifile]["OI_FLUX"].data["FLUXDATA"][imod*nBmin:(imod+1)*nBmin,:]**2)/(norm+1)
                    weight_flux = weight_flux + 1./(data[ifile]["OI_FLUX"].data["FLUXERR"][imod*nBmin:(imod+1)*nBmin,:]**2)
                    flux_weighted = flux_weighted + data[ifile]["OI_FLUX"].data["FLUXDATA"][imod*nBmin:(imod+1)*nBmin,:]/(data[ifile]["OI_FLUX"].data["FLUXERR"][imod*nBmin:(imod+1)*nBmin,:]**2)
                    norm+=1
        #temp.data["FLUXDATA"]*=norm  # flux are added not averaged
        #temp.data["FLUXERR"]= np.sqrt(temp.data["FLUXERR"]**2/norm + np.abs(flux2- temp.data["FLUXDATA"]**2))
        #temp.data["FLUXERR"]*=np.sqrt(norm)  # =/srqt(norm)*norm  => no better estimation than that for now
        temp.data["FLUXERR"]= np.sqrt(1./weight_flux + np.abs(flux2- temp.data["FLUXDATA"]**2))
        temp.data["FLUXDATA"]= flux_weighted/weight_flux
        temp.data["INT_TIME"] *=norm

        avgFits["OI_FLUX"]=temp

    #-----------------------------TF2-------------------------------------------
    if "TF2" in extnames:

        nB=np.array([len(datai["TF2"].data) for datai in data])
        nBmin=6
        temp=mat_hduCutRows(data[0]["TF2"],nBmin)

        #mean of the square of vis2 to compute std(vis2) = sqrt(|<vis2>^2-<vis2^2>|)
        vis22=temp.data["TF2"]**2
        norm=1
        for ifile in range(nfile):
            nmod=nB[ifile]//nBmin
            for imod in range(nmod):
                if (ifile!=0) or (imod!=0):  #Problem here
                    for key in ["TF2","TIME","MJD","INT_TIME"]:
                        if len(np.shape(temp.data[key]))==2:
                            temp.data[key]= (temp.data[key]*norm + data[ifile]["TF2"].data[key][imod*nBmin:(imod+1)*nBmin,:])/(norm+1)
                        else:
                            temp.data[key]= (temp.data[key]*norm + data[ifile]["TF2"].data[key][imod*nBmin:(imod+1)*nBmin])/(norm+1)
                    vis22 = (vis22*norm + data[ifile]["TF2"].data["TF2"][imod*nBmin:(imod+1)*nBmin,:]**2)/(norm+1)
                    norm+=1
        #temp.data["TF2ERR"]=np.sqrt(np.abs(vis22- temp.data["TF2"]**2))/np.sqrt(norm)
        temp.data["TF2ERR"]= np.sqrt(temp.data["TF2ERR"]**2/norm + np.abs(vis22- temp.data["TF2"]**2))
        temp.data["INT_TIME"] *=norm
        avgFits["TF2"]=temp


    return avgFits


####################### removing BCD in an oifits files ########################

def mat_removeBCD(oifits,saveFits=False):

    if type(oifits)==type(""):
        data=fits.open(oifits)
    else:
        data=oifits
    bcd1=data[0].header["ESO INS BCD1 NAME"]
    bcd2=data[0].header["ESO INS BCD2 NAME"]
    bcd=(bcd2 == "IN") + 2*(bcd1 == "IN")
    if bcd==0:
        #print("no bcd")
        return

    data[0].header["ESO INS BCD1 NAME"]="OUT"
    data[0].header["ESO INS BCD2 NAME"]="OUT"
    data[0].header["ESO INS BCD1 ID"]="OUT"
    data[0].header["ESO INS BCD2 ID"]="OUT"
    data[0].header["ESO INS BCD1 NO"]=1
    data[0].header["ESO INS BCD1 NO"]=1
    data[0].header["HIERARCH ESO DET BCD STATE"]=0
    data[0].header["HIERARCH ESO CFG BCD MODE"]="OUT-OUT"

    extnames=[hdu.name for hdu in data]

        #------------------OI_VIS2-------------------------
    if "OI_VIS2" in extnames:

        temp=data["OI_VIS2"].copy()

        nB=len(temp.data)
        for iB in range(nB):
            iB2=iB % 6
            shift0 = (int(iB) // 6)*6
            temp.data[iB2+shift0]=data["OI_VIS2"].data[BCD[bcd][iB2]+shift0]
            temp.data[iB2+shift0]["UCOORD"]*=BCDsign[bcd][iB2]
            temp.data[iB2+shift0]["VCOORD"]*=BCDsign[bcd][iB2]
            if BCDsign[bcd][iB2]==-1:
                temp.data[iB2+shift0]["STA_INDEX"]=np.flip(temp.data[iB2+shift0]["STA_INDEX"],axis=0)

        data["OI_VIS2"]=temp
    #------------------OI_VIS--------------------------
    if "OI_VIS" in extnames:

        temp=data["OI_VIS"].copy()

        nB=len(temp.data)
        for iB in range(nB):
            iB2=iB % 6
            shift0 = (iB // 6)*6
            temp.data[iB2+shift0]=data["OI_VIS"].data[BCD[bcd][iB2]+shift0]
            temp.data[iB2+shift0]["VISPHI"]*=BCDsign[bcd][iB2]
            temp.data[iB2+shift0]["UCOORD"]*=BCDsign[bcd][iB2]
            temp.data[iB2+shift0]["VCOORD"]*=BCDsign[bcd][iB2]
            if BCDsign[bcd][iB2]==-1:
                temp.data[iB2+shift0]["STA_INDEX"]=np.flip(temp.data[iB2+shift0]["STA_INDEX"],axis=0)

        data["OI_VIS"]=temp

    #------------------OI_T3---------------------------
    if "OI_T3" in extnames:
        sta_index=data["OI_ARRAY"].data["STA_INDEX"]
        temp=data["OI_T3"].copy()

        nB=len(temp.data)
        for iB in range(nB):
            iB2=iB % 4
            shift0 = (iB // 4)*4
            shift0V2 = (iB // 4)*6
            data["OI_T3"].data[BCDcp[bcd][iB2]+shift0]["T3PHI"]*=BCDcpsign[bcd][iB2]
            temp.data[iB2+shift0]=data["OI_T3"].data[BCDcp[bcd][iB2]+shift0]
            temp.data[iB2+shift0]["U1COORD"]=data["OI_VIS2"].data["UCOORD"][uv1[iB2]+shift0V2]
            temp.data[iB2+shift0]["V1COORD"]=data["OI_VIS2"].data["VCOORD"][uv1[iB2]+shift0V2]
            temp.data[iB2+shift0]["U2COORD"]=data["OI_VIS2"].data["UCOORD"][uv2[iB2]+shift0V2]
            temp.data[iB2+shift0]["V2COORD"]=data["OI_VIS2"].data["VCOORD"][uv2[iB2]+shift0V2]
            temp.data[iB2+shift0]["STA_INDEX"]=np.array([sta_index[sta_index_cp[iB2][i]] for i in range(3)])

        data["OI_T3"]=temp

    #------------------OI_FLUX-------------------------
    if "OI_FLUX" in extnames:
        if data[0].header["ESO DET CHIP NAME"]=="HAWAII-2RG":
            BCDflux=BCDfluxL
        else:
            BCDflux=BCDfluxN

        temp=data["OI_FLUX"].copy()

        nB=len(temp.data)
        for iB in range(nB):
            iB2=iB % 4
            shift0 = (iB // 4)*4
            temp.data[iB2+shift0]=data["OI_FLUX"].data[BCDflux[bcd][iB2]+shift0]

        data["OI_FLUX"]=temp


    #------------------TF2-------------------------
    if "TF2" in extnames:

        temp=data["TF2"].copy()

        nB=len(temp.data)
        for iB in range(nB):
            iB2=iB % 6
            shift0 = (iB // 6)*6
            temp.data[iB2+shift0]=data["TF2"].data[BCD[bcd][iB2]+shift0]
            if BCDsign[bcd][iB2]==-1:
                temp.data[iB2+shift0]["STA_INDEX"]=np.flip(temp.data[iB2+shift0]["STA_INDEX"],axis=0)

        data["TF2"]=temp

    #------------------END------------------------

    data[0].header["ESO INS BCD1 NAME"]="OUT"
    data[0].header["ESO INS BCD2 NAME"]="OUT"
    if saveFits==True:

        filenamein=data.filename()
        filenameout= filenamein.split(".fits")[0]+"_noBCD.fits"
        data.writeto(filenameout)

    return data


#=============================================================================


def mat_mergeByTplStart(something,save=False,verbose=True,dirOut="./MERGED",separateChopping=False):
    data=[]
    currentDir= os.path.abspath("./")
    if type(something)==type(""):
        #should be q directory but check first
        if not(os.path.isdir(something)):
            print("Error : {0} is not a directory",something)
            return
        currentDir=something
        something=[something+"/"+fi for fi in os.listdir(something) if ".fits" in fi]
    if type(something[0])==type(""):
        data=[fits.open(oifitsi) for oifitsi in something]
    else:
        data=something
    tplstart,sortedData=mat_sortByTplStart(data)
    ntpl=len(tplstart)

    if verbose:
        print("Number of TPLSTARTs : {0}".format(ntpl))

    mergedData=[]
    for itpl in range(ntpl):
        band=np.array([d[0].header["ESO DET NAME"] for d in sortedData[itpl]])
        chop=np.array([d[0].header["ESO ISS CHOP ST"] for d in sortedData[itpl]])
        #print(chop)

        #print(chop=='F')
        idxN=np.where(band=="MATISSE-N")[0]

        if separateChopping:
            idxLChop=np.where((band=="MATISSE-LM") &(chop=='T'))[0]
            idxLNonChop=np.where((band=="MATISSE-LM") &(chop=='F'))[0]
            idxL=np.array([])
        else:
            idxLChop=np.array([])
            idxLNonChop=np.array([])
            idxL=np.where(band=="MATISSE-LM")[0]



        if verbose:
            print("****** ({0}/{1}) TPLSTART={2} ******".format(itpl+1,ntpl,tplstart[itpl]))
            if separateChopping:
                print("Separating chopped and non-chopped exposures in LM-band")
                print("number of files to merge : {0} for LM non-chopped, {1} for LM chopped and {2} for N".format(len(idxLNonChop),len(idxLChop),len(idxN)))
            else:
                print("number of files to merge : {0} for LM and {1} for N".format(len(idxL),len(idxN)))

        for idxi in [idxL,idxLChop,idxLNonChop,idxN]:
            #print("idxi = {0}").format(idxi)
            if len(idxi!=0):
                datai=[sortedData[itpl][idata] for idata in idxi]
                filenames=[dataii.filename() for dataii in datai]
                #print(filenames)
                mergedi=mat_mergeOifits(datai)
                #print('mergedi = {0}').format(mergedi.filename())
                mergedData.append(mergedi)
                if save:
                    if not(os.path.exists(dirOut)):
                        os.mkdir(dirOut)
                    if separateChopping:
                        fileout=dirOut+"/"+os.path.basename(sortedData[itpl][idxi[0]].filename()).replace("_OUT","").replace("_IN","")
                    else:
                        fileout=dirOut+"/"+os.path.basename(sortedData[itpl][idxi[0]].filename()).replace("_OUT","").replace("_IN","").replace("_noChop","").replace("_Chop","")
                    if verbose:
                        print("Saving merged file to {0}".format(fileout))
                    mergedi.writeto(fileout,overwrite=True)

    return mergedData,data


#=============================================================================



def mat_hduCutRows(hdu,nrows):
    cols = hdu.data.columns
    #print('cols = {0}').format(cols)
    shape=np.shape(hdu.data[cols[0].name])
    newcols=[]
    for coli in cols:
        newcoli=fits.Column(name=coli.name,array=hdu.data[coli.name][0:nrows],unit=coli.unit,format=coli.format)
        newcols.append(newcoli)
    newhdu=fits.BinTableHDU.from_columns(fits.ColDefs(newcols))

    newhdu.header=hdu.header
    newhdu.update()
    return newhdu
