#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  This file is part of the Matisse pipeline GUI series
  Copyright (C) 2017- Observatoire de la Côte d'Azur

  Created on Tue Jul 16 12:28:03 2019
  @author: ame

  This script shows MATISSE oifits content on a neat ID card.

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
import libShowOifits as msoi
import os
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import argparse


inch=1/2.54



sta_name_list=np.array(['A0','A1','B0','B1','B2','B3','B4','B5','C0','C1','C2','C3','D0','D1','D2','E0','G0','G1','G2','H0','I1','J1','J2','J3','J4','J5','J6','K0','L0','M0','U1','U2','U3','U4'])

sta_pos_list=np.array([[-32.001,-48.013,-14.642,-55.812],[-32.001,-64.021, -9.434,-70.949],[-23.991,-48.019, -7.065,-53.212],[-23.991,-64.011, -1.863,-68.334],[-23.991,-72.011,  0.739,-75.899],
         [-23.991,-80.029,  3.348,-83.481],[-23.991,-88.013,  5.945,-91.030],[-23.991,-96.012,  8.547,-98.594],[-16.002,-48.013,  0.487,-50.607],[-16.002,-64.011,  5.691,-65.735],
         [-16.002,-72.019,  8.296,-73.307],[-16.002,-80.010, 10.896,-80.864],[  0.010,-48.012, 15.628,-45.397],[  0.010,-80.015, 26.039,-75.660],[  0.010,-96.012, 31.243,-90.787],
         [ 16.011,-48.016, 30.760,-40.196],[ 32.017,-48.0172,45.896,-34.990],[ 32.020,-112.010,66.716,-95.501],[ 31.995,-24.003, 38.063,-12.289],[ 64.015,-48.007, 76.150,-24.572],
         [ 72.001,-87.997, 96.711,-59.789],[ 88.016,-71.992,106.648,-39.444],[ 88.016,-96.005,114.460,-62.151],[ 88.016,  7.996,  80.628,36.193],[ 88.016,23.993,  75.424, 51.320],
         [ 88.016,47.987,  67.618, 74.009],[ 88.016,71.990,  59.810, 96.706],[ 96.002,-48.006,106.397,-14.165],[104.021,-47.998,113.977,-11.549],[112.013,-48.000,121.535, -8.951],
         [-16.000,-16.000, -9.925,-20.335],[ 24.000,24.000,  14.887, 30.502],[64.0013,47.9725, 44.915, 66.183],[112.000,8.000,  103.306,43.999]])

def _vltiplot(tels=np.array([]),baselines=np.array([]),symsize=2,color='k',tcolor=['k','k','k','k'],bcolor=['r','g','b','m','c','y'],labels=False,axe=None):

    UTs=sta_pos_list[-4:,:]
    ATs=sta_pos_list[:-4,:]

    if not(axe):
        axe=plt
    #plt.axis([np.min(sta_pos[:,2])-20,np.max(sta_pos[:,2])+20,np.min(sta_pos[:,3])-20,np.max(sta_pos[:,3])+20])
    axe.plot(UTs[:,2],UTs[:,3],marker='o',linestyle="",markersize=3*symsize,color=color)
    axe.plot(ATs[:,2],ATs[:,3],marker='o',linestyle="",markersize=symsize,color=color)

    if labels:
        for i in range(len(sta_name_list)):
            axe.text(sta_pos_list[i,2]+symsize,sta_pos_list[i,3]+symsize,sta_name_list[i],fontsize=2.5*symsize,color=color,horizontalalignment='center',zorder=1)


    if len(baselines)>0:
        for i in range(np.shape(baselines)[0]):
            tel1=np.where(sta_name_list == baselines[i,0])
            tel1x=sta_pos_list[tel1,2]
            tel1y=sta_pos_list[tel1,3]
            tel2=np.where(sta_name_list == baselines[i,1])
            tel2x=sta_pos_list[tel2,2]
            tel2y=sta_pos_list[tel2,3]
            axe.plot([tel1x[0,0],tel2x[0,0]],[tel1y[0,0],tel2y[0,0]],color=bcolor[i],zorder=2)

    if len(tels)>0:
         for i in range(len(tels)):
            tel=np.where(sta_name_list == tels[i])
            telx=sta_pos_list[tel,2]
            tely=sta_pos_list[tel,3]
            #print(telx)
            #print(tely)
            axe.scatter(telx[0,0],tely[0,0],c=tcolor[i],s=40*symsize,zorder=3)

def mat_showOiData(filename,wlRange=None,showErr=False,fig=None,visRange=None,visName='V2'):

    try:
        dic=msoi.open_oi(filename)
    except:
        print("Unable to open {0}".format(filename))
        return

    u=dic['VIS2']['U']
    v=dic['VIS2']['V']
    if visName=='V':
        v2=dic['VIS']['VISAMP']
        errv2=dic['VIS']['VISAMPERR']
    else:
        v2=dic['VIS2']['VIS2']
        errv2=dic['VIS2']['VIS2ERR']
    phi=dic['VIS']['DPHI']
    errphi=dic['VIS']['DPHIERR']
    cp=dic['T3']['CLOS']
    cperr=dic['T3']['CLOSERR']

    try:
        flux=dic['FLUX']['FLUX']
        corrFlux=False
    except:
        corrFlux=True
        v2=dic['VIS']['VISAMP']
        errv2=dic['VIS']['VISAMPERR']
    wl=dic['WLEN']*1e6

    nwl=np.size(wl)
    c=15
    dash=[[3*c,6*c],[1e-3,3*c,3*c,3*c],[1e-3,6*c,3*c,1e-3]]

    sta_index= dic['VIS2']['STA_INDEX']
    sta_name=[
             [dic['STA_NAME'][np.where(dic['STA_INDEX'] == sti[0])[0][0]],
              dic['STA_NAME'][np.where(dic['STA_INDEX'] == sti[1])[0][0]]]
             for sti in sta_index]

    if not(fig):
        fig=plt.figure(figsize=(29.7*inch,21*inch))

    plts={}

    cols=['r','g','b','m','c','y']

    colT3=['brm','gbc','gmy','cry']


    colTel=['yellow','magenta','cyan','grey']
    cols=['darkcyan','red','blue','darkmagenta','limegreen','y']
    colT3=['darkblue','k','darkred','darkgreen']

    if not corrFlux and visName=='V2':
        pltv2=[]
        for i in range(6):
            if i==0:
                plts['VIS2_{0}'.format(i)]=fig.add_axes([0.075, 0.1*(i+0.5), 0.22,0.1])
            else:
                plts['VIS2_{0}'.format(i)]=fig.add_axes([0.075, 0.1*(i+0.5), 0.22,0.1],sharex=plts['VIS2_0'],sharey=plts['VIS2_0'])
                plts['VIS2_{0}'.format(i)].get_xaxis().set_visible(False)
            pltv2.append(plts['VIS2_{0}'.format(i)])
        msoi.show_oi_vs_wlen(dic,key='VIS2',datatype="VIS2",plot_errorbars=showErr,subplotList=pltv2,colorList=cols)

        #plts['VIS2_{0}'.format(i)].plot(wl,v2[i,:],color=cols[i])


    else:
        pltcf=[]
        for i in range(6):
            if i==0:
                plts['VIS2_{0}'.format(i)]=fig.add_axes([0.075, 0.1*(i+0.5), 0.22,0.1])
            else:
                plts['VIS2_{0}'.format(i)]=fig.add_axes([0.075, 0.1*(i+0.5), 0.22,0.1],sharex=plts['VIS2_0'],sharey=plts['VIS2_0'])
                plts['VIS2_{0}'.format(i)].get_xaxis().set_visible(False)
            pltcf.append(plts['VIS2_{0}'.format(i)])
        msoi.show_oi_vs_wlen(dic,key='VIS',datatype="VISAMP",plot_errorbars=showErr,subplotList=pltcf,colorList=cols)
    pltphi=[]
    for i in range(6):
        if i==0:
            plts['PHI_{0}'.format(i)]=fig.add_axes([0.375, 0.1*(i+0.5), 0.22,0.1],sharex=plts['VIS2_0'])
        else:
            plts['PHI_{0}'.format(i)]=fig.add_axes([0.375, 0.1*(i+0.5), 0.22,0.1],sharex=plts['VIS2_0'],sharey=plts['PHI_0'])
            plts['PHI_{0}'.format(i)].get_xaxis().set_visible(False)
        pltphi.append(plts['PHI_{0}'.format(i)])
        #plts['PHI_{0}'.format(i)].plot(wl,phi[i,:],color=cols[i])

    msoi.show_oi_vs_wlen(dic,key='VIS',datatype="DPHI",plot_errorbars=showErr,subplotList=pltphi,colorList=cols)


    pltcp=[]
    for i in range(4):
        if i==0:
            plts['CP_{0}'.format(i)]=fig.add_axes([0.675, 0.1*(i+0.5), 0.22,0.1],sharex=plts['VIS2_0'])
        else:
            plts['CP_{0}'.format(i)]=fig.add_axes([0.675, 0.1*(i+0.5), 0.22,0.1],sharex=plts['VIS2_0'],sharey=plts['CP_0'])
            plts['CP_{0}'.format(i)].get_xaxis().set_visible(False)
        pltcp.append(plts['CP_{0}'.format(i)])
    msoi.show_oi_vs_wlen(dic,key='T3',datatype="CLOS",plot_errorbars=showErr,subplotList=pltcp,colorList=colT3)


    plts['FLUX']=fig.add_axes([0.075, 0.70, 0.22,0.15],sharex=plts['VIS2_0'])
    pltflx=[plts['FLUX']]*4
    if not(corrFlux):
        msoi.show_oi_vs_wlen(dic,key='FLUX',datatype="FLUX",plot_errorbars=showErr,subplotList=pltflx,colorList=colTel)

    ncols=np.size(u)//len(cols)
    plts['UV']=fig.add_axes([0.675, 0.53, 0.22,0.22*1.41])
    plts['UV'].set_aspect('equal', 'box')
    plts['UV'].scatter(u,v,c=cols*ncols,s=2)
    plts['UV'].scatter(-u,-v,c=cols*ncols,s=2)
    plts['UV'].set_xlabel('U (m)')
    plts['UV'].set_ylabel('V (m)')

    plts['VLTI']=fig.add_axes([0.305,0.65,0.32,0.32])
    plts['VLTI'].set_axis_off()
    plts['VLTI'].set_aspect('equal', 'box')
    _vltiplot(tels=dic['STA_NAME'],baselines=np.array(sta_name[0:6]),symsize=2,color='k',bcolor=cols,tcolor=colTel,labels=False,axe=plts['VLTI'])

    if (wlRange):
        plts['VIS2_0'].set_xlim(wlRange)

    plts['UV'].set_xlim(150,-150)
    plts['UV'].set_ylim(-150,150)

    if not(corrFlux):
        maxV2=np.max(v2)
        maxV2=maxV2 if maxV2<1.2 else 1.2
        minV2=np.min(v2)
        minV2=minV2 if minV2>-0.2 else -0.2
        plts['VIS2_0'].set_ylim(minV2,maxV2)

    if visRange!=None:
        plts['VIS2_0'].set_ylim(visRange[0],visRange[1])

    maxphi=np.max(phi)
    maxphi=maxphi if maxphi<180 else 180
    minphi=np.min(phi)
    v=minphi if minphi>-180 else -180
    plts['PHI_0'].set_ylim(minphi,maxphi)

    maxcp=np.max(cp)
    maxcp=maxcp if maxcp<180 else 180
    mincp=np.min(cp)
    v=mincp if mincp>-180 else -180
    plts['CP_0'].set_ylim(mincp,maxcp)


    for i in range(6):
        tel1=dic['STA_NAME'][np.where(dic['STA_INDEX'] == dic['VIS2']['STA_INDEX'][i,0])[0]][0]
        tel2=dic['STA_NAME'][np.where(dic['STA_INDEX'] == dic['VIS2']['STA_INDEX'][i,1])[0]][0]
        txt="{0}-{1}".format(tel1,tel2)
        plts["VIS2_{0}".format(i)].text(1.01,0.5,txt,transform=plts["VIS2_{0}".format(i)].transAxes,rotation=90,va='center')

    for i in range(4):
        tel1=dic['STA_NAME'][np.where(dic['STA_INDEX'] == dic['T3']['STA_INDEX'][i,0])[0]][0]
        tel2=dic['STA_NAME'][np.where(dic['STA_INDEX'] == dic['T3']['STA_INDEX'][i,1])[0]][0]
        tel3=dic['STA_NAME'][np.where(dic['STA_INDEX'] == dic['T3']['STA_INDEX'][i,2])[0]][0]
        txt="{0}-{1}-{2}".format(tel1,tel2,tel3)
        plts["CP_{0}".format(i)].text(1.01,0.5,txt,transform=plts["CP_{0}".format(i)].transAxes,rotation=90,va='center')


    if not(corrFlux) and visName=='V2':
        label="Square Visibility"
    elif not(corrFlux) and visName=='V':
        label="Visibility"
    else:
        label="Correlated flux"
    fig.text(0.02,0.35,label,rotation=90,horizontalalignment='center',verticalalignment='center')
    fig.text(0.32,0.35,"Differential Phase ($^o$)",rotation=90,horizontalalignment='center',verticalalignment='center')
    fig.text(0.62,0.25,"Closure Phase ($^o$)",rotation=90,horizontalalignment='center',verticalalignment='center')
    fig.text(0.185,0.01,"$\lambda$ ($\mu$m)",horizontalalignment='center',verticalalignment='center')
    fig.text(0.485,0.01,"$\lambda$ ($\mu$m)",horizontalalignment='center',verticalalignment='center')
    fig.text(0.785,0.01,"$\lambda$ ($\mu$m)",horizontalalignment='center',verticalalignment='center')

    wlmin=np.min(dic['WLEN'])*1e6
    wlmax=np.min(dic['WLEN'])*1e6
    if dic['BAND']=="LM":
        band=""
        if wlmin<4.2:
            band+="L"
        if wlmax>4.5:
            band+="M"
    else:
        band="N"

    filename=filename.split("/")[-1]

    fig.text(0.02,0.95,filename,fontsize=12)
    fig.text(0.03,0.92,"BAND = {0}".format(band))
    fig.text(0.03,0.90,"DISP = {0}".format(dic['DISP']))
    fig.text(0.03,0.88,"DIT = {0}s".format(dic['DIT']))
    fig.text(0.03,0.86,"BCD = {0}-{1}".format(dic['BCD1NAME'],dic['BCD2NAME']))




    telconf=""
    for el in dic['STA_NAME']:
        telconf+=el
        telconf+="-"
    telconf=telconf[:-1]

    fig.text(0.71,0.95,telconf,fontsize=20)

    seeing="{0:.2f}".format(dic['SEEING'])
    coherence="{0:.2f}".format(1000*dic['TAU0'])
    try:
        airmass="{0:.2f}".format(dic['HDR']['HIERARCH ESO ISS AIRM START'])
    except:
        print("WARNING: No airmass. Setting it to zero")
        airmass=0;
        fig.text(0.71,0.92,"Seeing = {0}\"".format(seeing))
    fig.text(0.71,0.90,"Coherence = {0}ms".format(coherence))
    fig.text(0.71,0.88,"Airmass = {0}".format(airmass))

    return fig


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Plot in the same figure VIS2, VISPHI, T3PHI, FLUX, and UV plan for MATISSE files."
      "\nCan produce pdf per file or merged pdf for all files in a folder.")

    parser.add_argument('filesOrDir', default="",help='A file, a list of files, or a directory')
    parser.add_argument('--wlRange', default=0, help='The min and max wavelengths [min,max] for the plot (microns)')
    parser.add_argument('--showErr', default=0, help='Plotting errors', action='store_true')
    parser.add_argument('--pdf', default=0, help='Create pdf(s) for the file(s)', action='store_true')
    parser.add_argument('--mergedpdf', default=0, help='Create a unique pdf for all the files', action='store_true')
    parser.add_argument('--visRange', default=0, help='The min and max for visibility (or CorrFLux)')
    parser.add_argument('--vis', default='V2', help='For visibility plot the incoherent squared visibility (V2) or the coherent visibility (V).')


    try:
        args = parser.parse_args()
    except:
        print("\n\033[93mRunning mat_showOiData.py --help to be kind with you:\033[0m\n")
        parser.print_help()
        print("\n     Example : python mat_showOiData.py . --mergedpdf")
        sys.exit(0)

    if args.wlRange!=0:
        wlRange=[float(val) for val in args.wlRange.replace("[","").replace("]","").split(",")]
        if  len(wlRange)!=2:
            print("Error : wlRange should be a pair of numbers not {0}".format(wlRange))
            parser.print_help()
            sys.exit(0)
    else:
        wlRange=None

    if args.visRange!=0:
        visRange=[float(val) for val in args.visRange.replace("[","").replace("]","").split(",")]
        if  len(visRange)!=2:
            print("Error : visRange should be a pair of numbers not {0}".format(visRange))
            parser.print_help()
            sys.exit(0)
    else:
        visRange=None

    showErr=args.showErr
    pdf=args.pdf
    merged=args.mergedpdf
    filesOrDir=args.filesOrDir
    if (pdf or merged):
        fig=plt.figure(figsize=(29.7*inch,21*inch))
    else:
        fig=None

    if os.path.isdir(filesOrDir):
        os.chdir(filesOrDir)
        dir0=os.path.basename(os.getcwd())
        filesOrDir=[fi for fi in os.listdir(filesOrDir) if ".fits" in fi]
    else:
        filesOrDir=[filesOrDir]
    nfiles=len(filesOrDir)

    if merged:
        pdfname=dir0+'_plots.pdf'
        pdf = PdfPages(pdfname)

    print("Processing {0} oifits files".format(len(filesOrDir)))
    for filei in tqdm(filesOrDir):
        fig=mat_showOiData(filei,wlRange=wlRange,showErr=showErr,fig=fig,visRange=visRange,visName=args.vis)
        if pdf and not(merged):
            pdfname=filei.split(".fits")[0]+".pdf"
            plt.savefig(pdfname)
            plt.clf()
            tqdm.write("saving to {0}".format(pdfname))
        elif merged:
            pdf.savefig(fig)
            plt.clf()
        else:
            fig=None
    if not(pdf or merged):
            plt.show()

    if merged:
        print("Saving plots to {0}".format(pdfname))
        pdf.close()
