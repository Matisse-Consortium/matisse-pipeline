#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of the Matisse pipeline GUI series
Copyright (C) 2017- Observatoire de la Côte d'Azur

Created on Tue Jul 22 2019
@author: fmillour

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.

You can use, modify and/ or redistribute the software under the terms
of the CeCILL license as circulated by CEA, CNRS and INRIA at the
following URL "http://www.cecill.info". You have a copy of the licence
in the LICENCE.md file.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.
"""
import argparse
import numpy as np
import sys
import os
import matplotlib
from astropy.io import fits


inch=1/2.54

if __name__ == '__main__':
    print("Starting...")

    #--------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Plots the transfer function of a night of MATISSE data from its oifits files.')

    #--------------------------------------------------------------------------
    parser.add_argument('in_dir', metavar='in_dir', type=str, \
    help='The path to the directory containing your oifits data.', default=None)
    parser.add_argument('--wlmin',  type=float, help='Minimum wavelength', default=0)
    parser.add_argument('--wlmax',  type=float, help='Maximum wavelength', default=0)

    #--------------------------------------------------------------------------
    parser.add_argument('--pdf',   action="store_true",
                        help='Create a pdf file for each target.')
    #--------------------------------------------------------------------------

    parser.add_argument('--showvis', action="store_true",
                        help='Plot visibilities instead of V squared.')

    #--------------------------------------------------------------------------
    parser.add_argument('--LM', action="store_true",
                        help='Only plot LM bands data')

    #--------------------------------------------------------------------------
    parser.add_argument('--N', action="store_true",
                        help='Only plot N band data')

    try:
        args = parser.parse_args()
    except:
        print("\n\033[93mRunning mat_ashowTransFunc.py --help to be kind with you:\033[0m\n")
        parser.print_help()
        sys.exit(0)
    #Pyplot needs to be imported after setting up the matplotlib backend.
    #The standard backend can produce  pdf but not if the script if
    #launched in background and the users disconnect himself.
    if args.pdf==True:
        matplotlib.use('PDF')
    else:
        pass
        #matplotlib.use('WXAgg')


    import matplotlib.pyplot as plt
    import libShowOifits as msoi

    print(matplotlib.get_backend())



    list_of_dicts          = msoi.open_oi_dir(args.in_dir)
    #filtered_list_of_dicts = msoi.filter_oi_list(list_of_dicts)

    wl0=[[3.,4.1],[8.5,10.5]]

    if args.LM==True:
        bands=["LM"]
    elif args.N==True:
        bands=["N"]
    else:
        bands=["LM","N"]

    for band in bands:

        if ((args.wlmin==0) | (args.wlmax==0)):
            if band=="LM":
                wl=wl0[0]
            else:
                wl=wl0[1]
        else:
            wl=[args.wlmin,args.wlmax]


        print("***************************{0}***************************".format(band))
        filtered_list_of_dicts = msoi.filter_oi_list(list_of_dicts,bands=[band])

        fig=plt.figure(figsize=(29.7*inch,19*inch))

        plts={}

        print("Setting V2 windows...")
        #####################################################
        pltv2=[]
        for i in range(6):
            if i==0:
                plts['VIS2_{0}'.format(i)]=fig.add_axes([0.075, 0.15*(i+0.5), 0.35,0.15])
            else:
                plts['VIS2_{0}'.format(i)]=fig.add_axes([0.075, 0.15*(i+0.5), 0.35,0.15],sharex=plts['VIS2_0'],sharey=plts['VIS2_0'])
                plts['VIS2_{0}'.format(i)].get_xaxis().set_visible(False)
            pltv2.append(plts['VIS2_{0}'.format(i)])

        print("Setting CP windows...")
        #####################################################
        pltcp=[]
        for i in range(4):
            if i==0:
                plts['CP_{0}'.format(i)]=fig.add_axes([0.575, 0.2*(i+0.5), 0.35,0.2],sharex=plts['VIS2_0'])
            else:
                plts['CP_{0}'.format(i)]=fig.add_axes([0.575, 0.2*(i+0.5), 0.35,0.2],sharex=plts['VIS2_0'],sharey=plts['CP_0'])
                plts['CP_{0}'.format(i)].get_xaxis().set_visible(False)
            pltcp.append(plts['CP_{0}'.format(i)])

        print("plotting...")
        #####################################################
        msoi.show_oi_vs_time(filtered_list_of_dicts ,wl,showvis=args.showvis,key="VIS2", datatype="VIS2",subplotList=pltv2,calColor='lightgray')

        msoi.show_oi_vs_time(filtered_list_of_dicts ,wl,showvis=args.showvis,key="TF2", datatype="TF2",subplotList=pltv2,calColor='blue')

        msoi.show_oi_vs_time(filtered_list_of_dicts ,wl,showvis=args.showvis,key="T3", datatype="CLOS",subplotList=pltcp)

        if args.pdf:
            pdfname = os.path.join(os.path.abspath( args.in_dir),"TransFunc_"+band+".pdf")
            plt.savefig(pdfname)
            plt.close(fig)
            print("saving to {0}".format(pdfname))

        if not(args.pdf):
            plt.show()
