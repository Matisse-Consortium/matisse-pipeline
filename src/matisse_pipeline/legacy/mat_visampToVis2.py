#!/usr/bin/env python3
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

import sys
import argparse
from astropy.io import fits
import os

#############################################################################

def mat_visampToVis2(something,save=False,verbose=True,dirOut="./VISAMP2VIS2"):
    data=[]
    if type(something)==type(""):
        #should be q directory but check first
        if not(os.path.isdir(something)):
            print("Error : {0} is not a directory".format(something))
            return
        something=[something+"/"+fi for fi in os.listdir(something) if ".fits" in fi]
    if type(something[0])==type(""):
        data=[fits.open(oifitsi) for oifitsi in something]
    else:
        data=something

    for di in data:
        visamp=di["OI_VIS"].data["VISAMP"]
        visamperr=di["OI_VIS"].data["VISAMPERR"]

        v2=di["OI_VIS"].data["VISAMP"]**2
        v2err=2*visamp*visamperr

        di["OI_VIS2"].data["VIS2DATA"]=v2
        di["OI_VIS2"].data["VIS2ERR"]=v2err
        di["OI_VIS2"].data["FLAG"]=di["OI_VIS"].data["FLAG"]

        dirOut=os.path.abspath((dirOut))

        if save:
               if not(os.path.exists(dirOut)):
                   os.mkdir(dirOut)

               fileout=os.path.join(dirOut,os.path.basename(di.filename()))

               if verbose:
                   print("Saving visampToVis2 file to {0}".format(fileout))
               di.writeto(fileout,overwrite=True)

    return data

#############################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Move MATISSE coherent" \
      "integration from OI_VIS VISAMP to OI_VIS2 VIS2DATA (including errors and flag)")
    parser.add_argument('dirIn', default="",help='Input directory')
    parser.add_argument('--dirOut', default="VISAMP2VIS2",help='Output directory')

    try:
        args = parser.parse_args()
    except:
        print("\n\033[93mRunning mat_visampToVis2.py --help to be kind with you:\033[0m\n")
        parser.print_help()
        print("\n     Example : python mat_visampToVis2.py . ")
        sys.exit(0)


    data=mat_visampToVis2(args.dirIn,save=True,dirOut=args.dirOut)
